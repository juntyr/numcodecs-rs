//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.85.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-pressio
//! [crates.io]: https://crates.io/crates/numcodecs-pressio
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-pressio
//! [docs.rs]: https://docs.rs/numcodecs-pressio/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_pressio
//!
//! libpressio codec wrapper for the [`numcodecs`] API.

use std::{
    borrow::Cow,
    collections::{BTreeMap, btree_map::Entry},
    sync::{Arc, Mutex, RwLock},
};

use ndarray::{Array, ArrayView, ArrayViewMut, CowArray, IxDyn};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig, StaticCodecVersion,
};
use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
/// Pressio codec which applies the identity function, i.e. passes through the
/// input unchanged during encoding and decoding.
pub struct PressioCodec {
    /// The Pressio compressor
    #[serde(flatten)]
    pub compressor: PressioCompressor,
    /// The codec's encoding format version. Do not provide this parameter explicitly.
    #[serde(default, rename = "_version")]
    pub version: StaticCodecVersion<1, 0, 0>,
}

/// Pressio compressor
pub struct PressioCompressor {
    // get_config clones the compressor, but we want the config to include the
    //  compressor metrics
    // so we make cheap shallow clones whenever possible and then later make
    //  the compressor unique with the clone-on-write `Arc::make_mut`
    // we pinky-promise to only lock the inner `Mutex` for immutable access
    //  when we have read-only access, otherwise we can go through
    //  `Mutex::get_mut`
    inner: RwLock<Arc<PressioCompressorInner>>,
}

impl Clone for PressioCompressor {
    #[expect(clippy::unwrap_used)]
    fn clone(&self) -> Self {
        Self {
            inner: RwLock::new(self.inner.read().unwrap().clone()),
        }
    }
}

struct PressioCompressorInner {
    compressor: Mutex<libpressio::PressioCompressor>,
    compressor_id: String,
    early_config: BTreeMap<String, PressioOption>,
}

impl Clone for PressioCompressorInner {
    #[expect(clippy::unwrap_used)]
    fn clone(&self) -> Self {
        let mut pressio = libpressio::Pressio::new().unwrap();
        let compressor = self.compressor.lock().unwrap();

        let mut compressor_clone = pressio.get_compressor(self.compressor_id.as_str()).unwrap();
        compressor_clone
            .set_name(compressor.get_name().unwrap())
            .unwrap();
        compressor_clone
            .set_options(&compressor.get_options().unwrap())
            .unwrap();

        std::mem::drop(compressor);

        Self {
            compressor: Mutex::new(compressor_clone),
            compressor_id: self.compressor_id.clone(),
            early_config: self.early_config.clone(),
        }
    }
}

impl Serialize for PressioCompressor {
    #[expect(clippy::too_many_lines)]
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        fn convert_from_pressio_options<E: serde::ser::Error>(
            options: impl Iterator<Item = (Option<String>, Option<libpressio::PressioOption>)>,
        ) -> Result<BTreeMap<String, PressioOption>, E> {
            let mut config = BTreeMap::new();

            for (name, option) in options {
                // skip invalid option names and values
                let (Some(name), Some(option)) = (name, option) else {
                    continue;
                };

                let value = match option {
                    libpressio::PressioOption::bool(Some(x)) => PressioOption::Bool(x),
                    libpressio::PressioOption::int8(Some(x)) => PressioOption::I8(x),
                    libpressio::PressioOption::int16(Some(x)) => PressioOption::I16(x),
                    libpressio::PressioOption::int32(Some(x)) => PressioOption::I32(x),
                    libpressio::PressioOption::int64(Some(x)) => PressioOption::I64(x),
                    libpressio::PressioOption::uint8(Some(x)) => PressioOption::U8(x),
                    libpressio::PressioOption::uint16(Some(x)) => PressioOption::U16(x),
                    libpressio::PressioOption::uint32(Some(x)) => PressioOption::U32(x),
                    libpressio::PressioOption::uint64(Some(x)) => PressioOption::U64(x),
                    libpressio::PressioOption::float32(Some(x)) => PressioOption::F32(x),
                    libpressio::PressioOption::float64(Some(x)) => PressioOption::F64(x),
                    libpressio::PressioOption::string(Some(x)) => PressioOption::String(x),
                    libpressio::PressioOption::vec_string(Some(x)) => PressioOption::VecString(x),
                    libpressio::PressioOption::dtype(Some(x)) => PressioOption::String(format!("{x}")),
                    libpressio::PressioOption::thread_safety(Some(x)) => PressioOption::String(format!("{x}")),
                    libpressio::PressioOption::data(Some(x)) => match x.clone_into_array() {
                        Option::None => continue,
                        Some(libpressio::PressioArray::Bool(x)) => PressioOption::DataBool(NdArray(x)),
                        Some(libpressio::PressioArray::Byte(x) | libpressio::PressioArray::U8(x)) => PressioOption::DataU8(NdArray(x)),
                        Some(libpressio::PressioArray::U16(x)) => PressioOption::DataU16(NdArray(x)),
                        Some(libpressio::PressioArray::U32(x)) => PressioOption::DataU32(NdArray(x)),
                        Some(libpressio::PressioArray::U64(x)) => PressioOption::DataU64(NdArray(x)),
                        Some(libpressio::PressioArray::I8(x)) => PressioOption::DataI8(NdArray(x)),
                        Some(libpressio::PressioArray::I16(x)) => PressioOption::DataI16(NdArray(x)),
                        Some(libpressio::PressioArray::I32(x)) => PressioOption::DataI32(NdArray(x)),
                        Some(libpressio::PressioArray::I64(x)) => PressioOption::DataI64(NdArray(x)),
                        Some(libpressio::PressioArray::F32(x)) => PressioOption::DataF32(NdArray(x)),
                        Some(libpressio::PressioArray::F64(x)) => PressioOption::DataF64(NdArray(x)),
                    },
                    libpressio::PressioOption::user_ptr(_)
                    | libpressio::PressioOption::unset
                    | _ /* non-exhaustive */ => continue,
                };

                let Some(nested_name) = name.strip_prefix('/') else {
                    // global option
                    if config.insert(name.clone(), value).is_some() {
                        return Err(serde::ser::Error::custom(format!(
                            "duplicate global option: `{name}`"
                        )));
                    }
                    continue;
                };

                // hierarchical option
                let mut parts = nested_name.split(':').peekable();

                let Some(first) = parts.next() else {
                    return Err(serde::ser::Error::custom(format!(
                        "invalid hierarchical config name `{name}`"
                    )));
                };
                let paths = first.split('/');

                if parts.peek().is_none() {
                    return Err(serde::ser::Error::custom(format!(
                        "invalid hierarchical config name `{name}`"
                    )));
                }
                let option_name = parts.map(String::from).collect::<Vec<_>>().join(":");

                let mut it = &mut config;
                for path in paths {
                    if let Entry::Vacant(entry) = it.entry(String::from(path)) {
                        entry.insert(PressioOption::Nested(BTreeMap::new()));
                    }

                    let Some(PressioOption::Nested(entry)) = it.get_mut(path) else {
                        return Err(serde::ser::Error::custom(format!(
                            "duplicate option nesting: `{path}` in `{name}`"
                        )));
                    };
                    it = entry;
                }
                if it.insert(option_name.clone(), value).is_some() {
                    return Err(serde::ser::Error::custom(format!(
                        "duplicate nested option: `{option_name}` in `{name}`"
                    )));
                }
            }

            Ok(config)
        }

        let inner = self.inner.read().map_err(serde::ser::Error::custom)?;
        let compressor = inner.compressor.lock().map_err(serde::ser::Error::custom)?;
        let options = compressor
            .get_options()
            .map_err(serde::ser::Error::custom)?;
        let metric_results = compressor
            .get_metric_results()
            .map_err(serde::ser::Error::custom)?;
        let name = compressor.get_name().map_err(serde::ser::Error::custom)?;

        let result = PressioCompressorBorrowedFormat {
            compressor_id: inner.compressor_id.as_str(),
            early_config: &inner.early_config,
            compressor_config: &convert_from_pressio_options(options.iter())?,
            metric_results: &convert_from_pressio_options(metric_results.iter())?,
            name: match name {
                "" => Option::None,
                name => Some(name),
            },
        }
        .serialize(serializer);
        std::mem::drop(compressor);
        std::mem::drop(inner);
        result
    }
}

impl<'de> Deserialize<'de> for PressioCompressor {
    #[expect(clippy::too_many_lines)] // FIXME
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        fn convert_to_pressio_options<E: serde::de::Error>(
            config: &BTreeMap<String, PressioOption>,
            template: Option<&libpressio::PressioOptions>,
            documentation: &libpressio::PressioOptions,
        ) -> Result<libpressio::PressioOptions, E> {
            let mut options =
                libpressio::PressioOptions::new().map_err(serde::de::Error::custom)?;

            let mut entries = vec![(vec![], config)];

            while let Some((path, entry)) = entries.pop() {
                for (key, value) in entry {
                    let option = match value {
                        PressioOption::None(None) => Option::None,
                        PressioOption::Bool(x) => Some(libpressio::PressioOption::bool(Some(*x))),
                        PressioOption::U8(x) => Some(libpressio::PressioOption::uint8(Some(*x))),
                        PressioOption::I8(x) => Some(libpressio::PressioOption::int8(Some(*x))),
                        PressioOption::U16(x) => Some(libpressio::PressioOption::uint16(Some(*x))),
                        PressioOption::I16(x) => Some(libpressio::PressioOption::int16(Some(*x))),
                        PressioOption::U32(x) => Some(libpressio::PressioOption::uint32(Some(*x))),
                        PressioOption::I32(x) => Some(libpressio::PressioOption::int32(Some(*x))),
                        PressioOption::U64(x) => Some(libpressio::PressioOption::uint64(Some(*x))),
                        PressioOption::I64(x) => Some(libpressio::PressioOption::int64(Some(*x))),
                        PressioOption::F32(x) => Some(libpressio::PressioOption::float32(Some(*x))),
                        PressioOption::F64(x) => Some(libpressio::PressioOption::float64(Some(*x))),
                        PressioOption::String(x) => {
                            Some(libpressio::PressioOption::string(Some(x.clone())))
                        }
                        PressioOption::VecString(x) => {
                            Some(libpressio::PressioOption::vec_string(Some(x.clone())))
                        }
                        PressioOption::DataBool(NdArray(x)) => {
                            Some(libpressio::PressioOption::data(Some(
                                libpressio::PressioData::new_copied(x),
                            )))
                        }
                        PressioOption::DataU8(NdArray(x)) => Some(libpressio::PressioOption::data(
                            Some(libpressio::PressioData::new_copied(x)),
                        )),
                        PressioOption::DataU16(NdArray(x)) => {
                            Some(libpressio::PressioOption::data(Some(
                                libpressio::PressioData::new_copied(x),
                            )))
                        }
                        PressioOption::DataU32(NdArray(x)) => {
                            Some(libpressio::PressioOption::data(Some(
                                libpressio::PressioData::new_copied(x),
                            )))
                        }
                        PressioOption::DataU64(NdArray(x)) => {
                            Some(libpressio::PressioOption::data(Some(
                                libpressio::PressioData::new_copied(x),
                            )))
                        }
                        PressioOption::DataI8(NdArray(x)) => Some(libpressio::PressioOption::data(
                            Some(libpressio::PressioData::new_copied(x)),
                        )),
                        PressioOption::DataI16(NdArray(x)) => {
                            Some(libpressio::PressioOption::data(Some(
                                libpressio::PressioData::new_copied(x),
                            )))
                        }
                        PressioOption::DataI32(NdArray(x)) => {
                            Some(libpressio::PressioOption::data(Some(
                                libpressio::PressioData::new_copied(x),
                            )))
                        }
                        PressioOption::DataI64(NdArray(x)) => {
                            Some(libpressio::PressioOption::data(Some(
                                libpressio::PressioData::new_copied(x),
                            )))
                        }
                        PressioOption::DataF32(NdArray(x)) => {
                            Some(libpressio::PressioOption::data(Some(
                                libpressio::PressioData::new_copied(x),
                            )))
                        }
                        PressioOption::DataF64(NdArray(x)) => {
                            Some(libpressio::PressioOption::data(Some(
                                libpressio::PressioData::new_copied(x),
                            )))
                        }
                        PressioOption::Nested(entry) => {
                            let mut nested_path = path.clone();
                            nested_path.push(key.clone());
                            entries.push((nested_path, entry));
                            continue;
                        }
                    };

                    let name = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("/{path}:{key}", path = path.join("/"))
                    };

                    if let Some(template) = template {
                        let Some(option_template) =
                            template.get(&name).map_err(serde::de::Error::custom)?
                        else {
                            let supported_options = template
                                .iter()
                                .filter_map(|(key, _value)| key)
                                .map(|x| format!("`{x}`"))
                                .collect::<Vec<_>>()
                                .join(", ");

                            return Err(serde::de::Error::custom(format!(
                                "unknown compressor configuration option: `{name}`, use one of {supported_options}"
                            )));
                        };

                        options
                            .set(&name, option_template.copy_type_only())
                            .map_err(serde::de::Error::custom)?;

                        if let Some(option) = option {
                            options
                                .set_with_cast(
                                    &name,
                                    option,
                                    libpressio::PressioConversionSafety::Special,
                                )
                                .map_err(|err| {
                                    let docs = match documentation.get(&name) {
                                        Ok(Some(libpressio::PressioOption::string(Some(docs)))) => {
                                            Some(docs)
                                        }
                                        _ => Option::None,
                                    };

                                    if let Some(docs) = docs {
                                        serde::de::Error::custom(format_args!("{err} ({docs})"))
                                    } else {
                                        serde::de::Error::custom(err)
                                    }
                                })?;
                        }
                    } else if let Some(option) = option {
                        options
                            .set(name, option)
                            .map_err(serde::de::Error::custom)?;
                    }
                }
            }

            Ok(options)
        }

        // TODO: better error handling
        let format = PressioCompressorOwnedFormat::deserialize(deserializer)?;
        std::mem::drop(format.metric_results);

        let mut pressio = libpressio::Pressio::new().map_err(serde::de::Error::custom)?;
        let mut compressor = pressio
            .get_compressor(format.compressor_id.as_str())
            .map_err(|err| {
                let supported_compressors = libpressio::supported_compressors().map_or_else(
                    |_| String::from("<unknown>"),
                    |x| {
                        x.iter()
                            .map(|x| format!("`{x}`"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    },
                );

                serde::de::Error::custom(format_args!(
                    "{err}, choose one of: {supported_compressors}"
                ))
            })?;

        if let Some(name) = &format.name {
            compressor
                .set_name(name)
                .map_err(serde::de::Error::custom)?;
        }

        let documentation = compressor
            .get_documentation()
            .map_err(serde::de::Error::custom)?;

        let early_options =
            convert_to_pressio_options(&format.early_config, Option::None, &documentation)?;
        compressor
            .set_options(&early_options)
            .map_err(serde::de::Error::custom)?;
        let options_template = compressor.get_options().map_err(serde::de::Error::custom)?;

        let options = convert_to_pressio_options(
            &format.compressor_config,
            Some(&options_template),
            &documentation,
        )?;
        compressor
            .set_options(&options)
            .map_err(serde::de::Error::custom)?;

        Ok(Self {
            inner: RwLock::new(Arc::new(PressioCompressorInner {
                compressor: Mutex::new(compressor),
                compressor_id: format.compressor_id,
                early_config: format.early_config,
            })),
        })
    }
}

impl JsonSchema for PressioCompressor {
    fn schema_name() -> Cow<'static, str> {
        PressioCompressorOwnedFormat::schema_name()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        PressioCompressorOwnedFormat::json_schema(generator)
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename = "PressioCompressor")]
struct PressioCompressorOwnedFormat {
    /// The id of the compressor
    compressor_id: String,
    /// Configuration for the structure of the compressor
    #[serde(default)]
    early_config: BTreeMap<String, PressioOption>,
    /// Configuration for the compressor
    #[serde(default)]
    compressor_config: BTreeMap<String, PressioOption>,
    /// Results of the compressor metrics (output-only)
    #[serde(default)]
    metric_results: BTreeMap<String, PressioOption>,
    /// Optional name for the compressor when used in hierarchical mode
    #[serde(default)]
    name: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename = "PressioCompressor")]
struct PressioCompressorBorrowedFormat<'a> {
    /// The id of the compressor
    compressor_id: &'a str,
    /// Configuration for the structure of the compressor
    early_config: &'a BTreeMap<String, PressioOption>,
    /// Configuration for the compressor
    compressor_config: &'a BTreeMap<String, PressioOption>,
    /// Results of the compressor metrics (output-only)
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    metric_results: &'a BTreeMap<String, PressioOption>,
    /// Optional name for the compressor when used in hierarchical mode
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<&'a str>,
}

#[expect(missing_docs)]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
/// Pressio option value
pub enum PressioOption {
    None(None),
    Bool(bool),
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    String(String),
    VecString(Vec<String>),
    DataBool(NdArray<bool>),
    DataU8(NdArray<u8>),
    DataU16(NdArray<u16>),
    DataU32(NdArray<u32>),
    DataU64(NdArray<u64>),
    DataI8(NdArray<i8>),
    DataI16(NdArray<i16>),
    DataI32(NdArray<i32>),
    DataI64(NdArray<i64>),
    DataF32(NdArray<f32>),
    DataF64(NdArray<f64>),
    Nested(BTreeMap<String, Self>),
}

#[derive(Clone)]
/// Pressio n-dimensional data array
pub struct NdArray<T>(Array<T, IxDyn>);

impl<T: std::fmt::Debug> std::fmt::Debug for NdArray<T> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.0.fmt(fmt)
    }
}

impl<T: Serialize> Serialize for NdArray<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serde_ndim::serialize(&self.0, serializer)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for NdArray<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        serde_ndim::deserialize(deserializer).map(Self)
    }
}

impl<T: JsonSchema> JsonSchema for NdArray<T> {
    fn inline_schema() -> bool {
        false
    }

    fn schema_name() -> Cow<'static, str> {
        Cow::Owned(format!("{}NdArray", std::any::type_name::<T>()))
    }

    fn schema_id() -> Cow<'static, str> {
        Cow::Owned(format!(
            "{}::NdArray<{}>",
            module_path!(),
            std::any::type_name::<T>()
        ))
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        let item = generator.subschema_for::<T>();
        let nested = generator.subschema_for::<Self>();

        json_schema!({
            "anyOf": [
                {
                    "type": "array",
                    "items": item,
                },
                {
                    "type": "array",
                    "items": nested,
                }
            ]
        })
    }
}

#[derive(Copy, Clone, Debug)]
/// Equivalent of `Option<!>::None`
pub struct None;

impl Serialize for None {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_none()
    }
}

impl<'de> Deserialize<'de> for None {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        enum Never {}

        impl<'de> Deserialize<'de> for Never {
            fn deserialize<D: Deserializer<'de>>(_deserializer: D) -> Result<Self, D::Error> {
                Err(serde::de::Error::custom("never"))
            }
        }

        match Option::<Never>::deserialize(deserializer) {
            Ok(Option::Some(x)) => match x {},
            Ok(Option::None) => Ok(Self),
            Err(err) => Err(err),
        }
    }
}

impl JsonSchema for None {
    fn schema_name() -> Cow<'static, str> {
        Cow::Borrowed("null")
    }

    fn inline_schema() -> bool {
        true
    }

    fn json_schema(_generator: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "type": "null"
        })
    }
}

impl Codec for PressioCodec {
    type Error = PressioCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        fn encode_typed<T: libpressio::PressioElement>(
            compressor: &mut libpressio::PressioCompressor,
            data: CowArray<T, IxDyn>,
        ) -> Result<AnyArray, PressioCodecError> {
            let compressed_data = libpressio::PressioData::new_with_shared(data, |data| {
                let compressed_data =
                    libpressio::PressioData::new_empty(libpressio::PressioDtype::Byte, []);

                compressor.compress(data, compressed_data).map_err(|err| {
                    PressioCodecError::PressioEncodeFailed {
                        source: PressioCodingError(err),
                    }
                })
            })?;

            let Some(compressed_data) = compressed_data.clone_into_array() else {
                if compressed_data.has_data() {
                    return Err(PressioCodecError::EncodeToUnknownDtype);
                }

                return Err(PressioCodecError::EncodeToArrayWithoutData);
            };

            match compressed_data {
                libpressio::PressioArray::Bool(_) => Err(PressioCodecError::EncodeToBoolArray),
                libpressio::PressioArray::U8(a) | libpressio::PressioArray::Byte(a) => {
                    Ok(AnyArray::U8(a))
                }
                libpressio::PressioArray::U16(a) => Ok(AnyArray::U16(a)),
                libpressio::PressioArray::U32(a) => Ok(AnyArray::U32(a)),
                libpressio::PressioArray::U64(a) => Ok(AnyArray::U64(a)),
                libpressio::PressioArray::I8(a) => Ok(AnyArray::I8(a)),
                libpressio::PressioArray::I16(a) => Ok(AnyArray::I16(a)),
                libpressio::PressioArray::I32(a) => Ok(AnyArray::I32(a)),
                libpressio::PressioArray::I64(a) => Ok(AnyArray::I64(a)),
                libpressio::PressioArray::F32(a) => Ok(AnyArray::F32(a)),
                libpressio::PressioArray::F64(a) => Ok(AnyArray::F64(a)),
            }
        }

        let Ok(mut inner) = self.compressor.inner.write() else {
            return Err(PressioCodecError::PressioPoisonedLock);
        };

        let Ok(compressor) = Arc::make_mut(&mut inner).compressor.get_mut() else {
            return Err(PressioCodecError::PressioPoisonedLock);
        };

        match data {
            AnyCowArray::U8(data) => encode_typed(compressor, data),
            AnyCowArray::U16(data) => encode_typed(compressor, data),
            AnyCowArray::U32(data) => encode_typed(compressor, data),
            AnyCowArray::U64(data) => encode_typed(compressor, data),
            AnyCowArray::I8(data) => encode_typed(compressor, data),
            AnyCowArray::I16(data) => encode_typed(compressor, data),
            AnyCowArray::I32(data) => encode_typed(compressor, data),
            AnyCowArray::I64(data) => encode_typed(compressor, data),
            AnyCowArray::F32(data) => encode_typed(compressor, data),
            AnyCowArray::F64(data) => encode_typed(compressor, data),
            data => Err(PressioCodecError::UnsupportedDtype(data.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        fn decode_typed<T: libpressio::PressioElement>(
            compressor: &mut libpressio::PressioCompressor,
            encoded: CowArray<T, IxDyn>,
        ) -> Result<AnyArray, PressioCodecError> {
            let decompressed_data = libpressio::PressioData::new_with_shared(encoded, |encoded| {
                let decompressed_data =
                    libpressio::PressioData::new_empty(libpressio::PressioDtype::Byte, []);

                compressor
                    .decompress(encoded, decompressed_data)
                    .map_err(|err| PressioCodecError::PressioDecodeFailed {
                        source: PressioCodingError(err),
                    })
            })?;

            let Some(decompressed_data) = decompressed_data.clone_into_array() else {
                if decompressed_data.has_data() {
                    return Err(PressioCodecError::DecodeToUnknownDtype);
                }

                return Err(PressioCodecError::DecodeToArrayWithoutData);
            };

            match decompressed_data {
                libpressio::PressioArray::Bool(_) => Err(PressioCodecError::DecodeToBoolArray),
                libpressio::PressioArray::U8(a) | libpressio::PressioArray::Byte(a) => {
                    Ok(AnyArray::U8(a))
                }
                libpressio::PressioArray::U16(a) => Ok(AnyArray::U16(a)),
                libpressio::PressioArray::U32(a) => Ok(AnyArray::U32(a)),
                libpressio::PressioArray::U64(a) => Ok(AnyArray::U64(a)),
                libpressio::PressioArray::I8(a) => Ok(AnyArray::I8(a)),
                libpressio::PressioArray::I16(a) => Ok(AnyArray::I16(a)),
                libpressio::PressioArray::I32(a) => Ok(AnyArray::I32(a)),
                libpressio::PressioArray::I64(a) => Ok(AnyArray::I64(a)),
                libpressio::PressioArray::F32(a) => Ok(AnyArray::F32(a)),
                libpressio::PressioArray::F64(a) => Ok(AnyArray::F64(a)),
            }
        }

        let Ok(mut inner) = self.compressor.inner.write() else {
            return Err(PressioCodecError::PressioPoisonedLock);
        };

        let Ok(compressor) = Arc::make_mut(&mut inner).compressor.get_mut() else {
            return Err(PressioCodecError::PressioPoisonedLock);
        };

        match encoded {
            AnyCowArray::U8(encoded) => decode_typed(compressor, encoded),
            AnyCowArray::U16(encoded) => decode_typed(compressor, encoded),
            AnyCowArray::U32(encoded) => decode_typed(compressor, encoded),
            AnyCowArray::U64(encoded) => decode_typed(compressor, encoded),
            AnyCowArray::I8(encoded) => decode_typed(compressor, encoded),
            AnyCowArray::I16(encoded) => decode_typed(compressor, encoded),
            AnyCowArray::I32(encoded) => decode_typed(compressor, encoded),
            AnyCowArray::I64(encoded) => decode_typed(compressor, encoded),
            AnyCowArray::F32(encoded) => decode_typed(compressor, encoded),
            AnyCowArray::F64(encoded) => decode_typed(compressor, encoded),
            encoded => Err(PressioCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    #[expect(clippy::too_many_lines)] // FIXME
    fn decode_into(
        &self,
        encoded: AnyArrayView,
        decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        fn decompress_typed<T: libpressio::PressioElement>(
            compressor: &mut libpressio::PressioCompressor,
            encoded: ArrayView<T, IxDyn>,
            decoded_dtype: libpressio::PressioDtype,
            decoded_shape: &[usize],
        ) -> Result<libpressio::PressioData, PressioCodecError> {
            libpressio::PressioData::new_with_shared(encoded, |encoded| {
                let decompressed_data =
                    libpressio::PressioData::new_empty(decoded_dtype, decoded_shape);

                compressor
                    .decompress(encoded, decompressed_data)
                    .map_err(|err| PressioCodecError::PressioDecodeFailed {
                        source: PressioCodingError(err),
                    })
            })
        }

        fn decode_into_typed<T: libpressio::PressioElement>(
            decompressed_data: &libpressio::PressioData,
            mut decoded: ArrayViewMut<T, IxDyn>,
        ) -> Result<(), PressioCodecError> {
            if !decompressed_data.has_data() {
                return Err(PressioCodecError::DecodeToArrayWithoutData);
            }

            let dtype = match <T as libpressio::PressioElement>::DTYPE {
                libpressio::PressioDtype::Bool => {
                    return Err(PressioCodecError::DecodeToBoolArray);
                }
                libpressio::PressioDtype::Byte | libpressio::PressioDtype::U8 => AnyArrayDType::U8,
                libpressio::PressioDtype::U16 => AnyArrayDType::U16,
                libpressio::PressioDtype::U32 => AnyArrayDType::U32,
                libpressio::PressioDtype::U64 => AnyArrayDType::U64,
                libpressio::PressioDtype::I8 => AnyArrayDType::I8,
                libpressio::PressioDtype::I16 => AnyArrayDType::I16,
                libpressio::PressioDtype::I32 => AnyArrayDType::I32,
                libpressio::PressioDtype::I64 => AnyArrayDType::I64,
                libpressio::PressioDtype::F32 => AnyArrayDType::F32,
                libpressio::PressioDtype::F64 => AnyArrayDType::F64,
            };
            let decompressed_dtype = match decompressed_data.dtype() {
                Option::None => return Err(PressioCodecError::DecodeToUnknownDtype),
                Some(libpressio::PressioDtype::Bool) => {
                    return Err(PressioCodecError::DecodeToBoolArray);
                }
                Some(libpressio::PressioDtype::Byte | libpressio::PressioDtype::U8) => {
                    AnyArrayDType::U8
                }
                Some(libpressio::PressioDtype::U16) => AnyArrayDType::U16,
                Some(libpressio::PressioDtype::U32) => AnyArrayDType::U32,
                Some(libpressio::PressioDtype::U64) => AnyArrayDType::U64,
                Some(libpressio::PressioDtype::I8) => AnyArrayDType::I8,
                Some(libpressio::PressioDtype::I16) => AnyArrayDType::I16,
                Some(libpressio::PressioDtype::I32) => AnyArrayDType::I32,
                Some(libpressio::PressioDtype::I64) => AnyArrayDType::I64,
                Some(libpressio::PressioDtype::F32) => AnyArrayDType::F32,
                Some(libpressio::PressioDtype::F64) => AnyArrayDType::F64,
            };

            if dtype != decompressed_dtype {
                return Err(PressioCodecError::MismatchedDecodeIntoArray {
                    source: AnyArrayAssignError::DTypeMismatch {
                        src: decompressed_dtype,
                        dst: dtype,
                    },
                });
            }

            if decompressed_data
                .with_shared::<T, IxDyn, ()>(decoded.dim(), |decompressed| {
                    decoded.assign(&decompressed);
                })
                .is_none()
            {
                return Err(PressioCodecError::MismatchedDecodeIntoArray {
                    source: AnyArrayAssignError::ShapeMismatch {
                        src: decompressed_data.shape(),
                        dst: decoded.shape().to_vec(),
                    },
                });
            }

            Ok(())
        }

        let Ok(mut inner) = self.compressor.inner.write() else {
            return Err(PressioCodecError::PressioPoisonedLock);
        };

        let Ok(compressor) = Arc::make_mut(&mut inner).compressor.get_mut() else {
            return Err(PressioCodecError::PressioPoisonedLock);
        };

        let decoded_dtype = match decoded.dtype() {
            AnyArrayDType::U8 => libpressio::PressioDtype::U8,
            AnyArrayDType::U16 => libpressio::PressioDtype::U16,
            AnyArrayDType::U32 => libpressio::PressioDtype::U32,
            AnyArrayDType::U64 => libpressio::PressioDtype::U64,
            AnyArrayDType::I8 => libpressio::PressioDtype::I8,
            AnyArrayDType::I16 => libpressio::PressioDtype::I16,
            AnyArrayDType::I32 => libpressio::PressioDtype::I32,
            AnyArrayDType::I64 => libpressio::PressioDtype::I64,
            AnyArrayDType::F32 => libpressio::PressioDtype::F32,
            AnyArrayDType::F64 => libpressio::PressioDtype::F64,
            decoded_dtype => return Err(PressioCodecError::UnsupportedDtype(decoded_dtype)),
        };
        let decoded_shape = decoded.shape();

        let decompressed_data = match encoded {
            AnyArrayView::U8(encoded) => {
                decompress_typed(compressor, encoded, decoded_dtype, decoded_shape)
            }
            AnyArrayView::U16(encoded) => {
                decompress_typed(compressor, encoded, decoded_dtype, decoded_shape)
            }
            AnyArrayView::U32(encoded) => {
                decompress_typed(compressor, encoded, decoded_dtype, decoded_shape)
            }
            AnyArrayView::U64(encoded) => {
                decompress_typed(compressor, encoded, decoded_dtype, decoded_shape)
            }
            AnyArrayView::I8(encoded) => {
                decompress_typed(compressor, encoded, decoded_dtype, decoded_shape)
            }
            AnyArrayView::I16(encoded) => {
                decompress_typed(compressor, encoded, decoded_dtype, decoded_shape)
            }
            AnyArrayView::I32(encoded) => {
                decompress_typed(compressor, encoded, decoded_dtype, decoded_shape)
            }
            AnyArrayView::I64(encoded) => {
                decompress_typed(compressor, encoded, decoded_dtype, decoded_shape)
            }
            AnyArrayView::F32(encoded) => {
                decompress_typed(compressor, encoded, decoded_dtype, decoded_shape)
            }
            AnyArrayView::F64(encoded) => {
                decompress_typed(compressor, encoded, decoded_dtype, decoded_shape)
            }
            encoded => return Err(PressioCodecError::UnsupportedDtype(encoded.dtype())),
        }?;

        match decoded {
            AnyArrayViewMut::U8(decoded) => decode_into_typed(&decompressed_data, decoded),
            AnyArrayViewMut::U16(decoded) => decode_into_typed(&decompressed_data, decoded),
            AnyArrayViewMut::U32(decoded) => decode_into_typed(&decompressed_data, decoded),
            AnyArrayViewMut::U64(decoded) => decode_into_typed(&decompressed_data, decoded),
            AnyArrayViewMut::I8(decoded) => decode_into_typed(&decompressed_data, decoded),
            AnyArrayViewMut::I16(decoded) => decode_into_typed(&decompressed_data, decoded),
            AnyArrayViewMut::I32(decoded) => decode_into_typed(&decompressed_data, decoded),
            AnyArrayViewMut::I64(decoded) => decode_into_typed(&decompressed_data, decoded),
            AnyArrayViewMut::F32(decoded) => decode_into_typed(&decompressed_data, decoded),
            AnyArrayViewMut::F64(decoded) => decode_into_typed(&decompressed_data, decoded),
            decoded => Err(PressioCodecError::UnsupportedDtype(decoded.dtype())),
        }
    }
}

impl StaticCodec for PressioCodec {
    const CODEC_ID: &'static str = "pressio.rs";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<'_, Self> {
        StaticCodecConfig::from(self)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`PressioCodec`].
pub enum PressioCodecError {
    /// [`PressioCodec`] does not support the dtype
    #[error("Pressio does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`PressioCodec`] lock was poisoned
    #[error("Pressio lock was poisoned")]
    PressioPoisonedLock,
    /// [`PressioCodec`] failed to encode the data
    #[error("Pressio failed to encode the data")]
    PressioEncodeFailed {
        /// Opaque source error
        source: PressioCodingError,
    },
    /// [`PressioCodec`] encoded to an unknown unsupported dtype
    #[error("Pressio encoded to an unknown unsupported dtype")]
    EncodeToUnknownDtype,
    /// [`PressioCodec`] encoded to an array without data
    #[error("Pressio encoded to an array without data")]
    EncodeToArrayWithoutData,
    /// [`PressioCodec`] encoded to a bool array, which is unsupported
    #[error("Pressio encoded to a bool array, which is unsupported")]
    EncodeToBoolArray,
    /// [`PressioCodec`] failed to decode the data
    #[error("Pressio failed to decode the data")]
    PressioDecodeFailed {
        /// Opaque source error
        source: PressioCodingError,
    },
    /// [`PressioCodec`] decoded to an unknown unsupported dtype
    #[error("Pressio decoded to an unknown unsupported dtype")]
    DecodeToUnknownDtype,
    /// [`PressioCodec`] decoded to an array without data
    #[error("Pressio decoded to an array without data")]
    DecodeToArrayWithoutData,
    /// [`PressioCodec`] decoded to a bool array, which is unsupported
    #[error("Pressio decoded to a bool array, which is unsupported")]
    DecodeToBoolArray,
    /// [`PressioCodec`] cannot decode into the provided array
    #[error("Pressio cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding with libpressio fails
pub struct PressioCodingError(libpressio::PressioError);

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    use ndarray::Array1;
    use serde_json::json;

    #[test]
    fn linear_quantizer() {
        let pressio = PressioCodec::deserialize(json!({
            "compressor_id": "linear_quantizer",
            "early_config": {
                "pressio:metric": "composite",
            },
            "compressor_config": {
                "pressio:abs": 10.0,
                "pressio:metric": "composite",
                "composite:plugins": ["printer", "size", "time"],
            }
        }))
        .unwrap();

        let data = ndarray::linspace(0.0, 100.0, 50)
            .collect::<Array1<f64>>()
            .into_dyn();

        let encoded = pressio
            .encode(AnyCowArray::F64(CowArray::from(&data)))
            .unwrap();

        let decoded = pressio.decode(encoded.cow());
        assert!(matches!(
            decoded,
            Err(PressioCodecError::DecodeToArrayWithoutData)
        ));

        let mut decoded = ndarray::Array::zeros(data.dim());
        pressio
            .decode_into(encoded.view(), AnyArrayViewMut::F64(decoded.view_mut()))
            .unwrap();

        for (i, o) in data.iter().zip(decoded.iter()) {
            assert!(((*i) - (*o)).abs() <= 10.0);
        }

        let config = serde_json::to_string(&pressio.get_config()).unwrap();
        assert!(config.contains("\"size:compressed_size\":400"));
    }
}
