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

use std::{borrow::Cow, collections::BTreeMap, sync::Mutex};

use ndarray::{ArrayView, ArrayViewMut, CowArray, IxDyn};
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
    format: PressioCompressorFormat,
    compressor: Mutex<libpressio::PressioCompressor>,
}

impl Clone for PressioCompressor {
    #[expect(clippy::unwrap_used)]
    fn clone(&self) -> Self {
        let mut pressio = libpressio::Pressio::new().unwrap();
        let mut compressor = pressio
            .get_compressor(self.format.compressor_id.as_str())
            .unwrap();
        let options = self.compressor.lock().unwrap().get_options().unwrap();
        compressor.set_options(&options).unwrap();

        Self {
            format: self.format.clone(),
            compressor: Mutex::new(compressor),
        }
    }
}

impl Serialize for PressioCompressor {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.format.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PressioCompressor {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        fn convert_to_pressio_options(
            config: &BTreeMap<String, PressioOption>,
        ) -> Result<libpressio::PressioOptions, libpressio::PressioError> {
            let mut options = libpressio::PressioOptions::new()?;

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
                        format!("{path}:{key}", path = path.join("/"))
                    };

                    if let Some(option) = option {
                        options = options.set(name, option)?;
                    }
                }
            }

            Ok(options)
        }

        // TODO: better error handling
        let format = PressioCompressorFormat::deserialize(deserializer)?;

        let mut pressio = libpressio::Pressio::new()
            .map_err(|err| serde::de::Error::custom(err.message.as_str()))?;
        let mut compressor = pressio
            .get_compressor(format.compressor_id.as_str())
            .map_err(|err| {
                let supported_compressors = pressio.supported_compressors().map_or_else(
                    |_| String::from("<unknown>"),
                    |x| {
                        x.iter()
                            .map(|x| format!("`{x}`"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    },
                );

                serde::de::Error::custom(format_args!(
                    "{}, choose one of: {}",
                    err.message, supported_compressors
                ))
            })?;

        if let Some(name) = &format.name {
            compressor
                .set_name(name)
                .map_err(|err| serde::de::Error::custom(err.message.as_str()))?;
        }

        let early_options = convert_to_pressio_options(&format.early_config)
            .map_err(|err| serde::de::Error::custom(err.message.as_str()))?;
        compressor
            .set_options(&early_options)
            .map_err(|err| serde::de::Error::custom(err.message.as_str()))?;

        let _options_template = compressor
            .get_options()
            .map_err(|err| serde::de::Error::custom(err.message))?;

        if !format.compressor_config.is_empty() {
            // TODO
            return Err(serde::de::Error::custom(
                "compressor_config is not yet supported",
            ));
        }

        Ok(Self {
            format,
            compressor: Mutex::new(compressor),
        })
    }
}

impl JsonSchema for PressioCompressor {
    fn schema_name() -> Cow<'static, str> {
        PressioCompressorFormat::schema_name()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        PressioCompressorFormat::json_schema(generator)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename = "PressioCompressor")]
struct PressioCompressorFormat {
    /// The id of the compressor
    compressor_id: String,
    /// Configuration for the structure of the compressor
    #[serde(default)]
    early_config: BTreeMap<String, PressioOption>,
    /// Configuration for the compressor
    #[serde(default)]
    compressor_config: BTreeMap<String, PressioOption>,
    /// Optional name for the compressor when used in hierarchical mode
    #[serde(default)]
    name: Option<String>,
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
    Nested(BTreeMap<String, PressioOption>),
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

            eprintln!(
                "compressed: {} {} {} {:?}",
                compressed_data.has_data(),
                compressed_data.len(),
                compressed_data.ndim(),
                compressed_data.dtype()
            );

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

        let Ok(mut compressor) = self.compressor.compressor.lock() else {
            return Err(PressioCodecError::PressioPoisonedMutex);
        };

        match data {
            AnyCowArray::U8(data) => encode_typed(&mut compressor, data),
            AnyCowArray::U16(data) => encode_typed(&mut compressor, data),
            AnyCowArray::U32(data) => encode_typed(&mut compressor, data),
            AnyCowArray::U64(data) => encode_typed(&mut compressor, data),
            AnyCowArray::I8(data) => encode_typed(&mut compressor, data),
            AnyCowArray::I16(data) => encode_typed(&mut compressor, data),
            AnyCowArray::I32(data) => encode_typed(&mut compressor, data),
            AnyCowArray::I64(data) => encode_typed(&mut compressor, data),
            AnyCowArray::F32(data) => encode_typed(&mut compressor, data),
            AnyCowArray::F64(data) => encode_typed(&mut compressor, data),
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
                    .compress(encoded, decompressed_data)
                    .map_err(|err| PressioCodecError::PressioDecodeFailed {
                        source: PressioCodingError(err),
                    })
            })?;

            eprintln!(
                "decompressed: {} {} {} {:?}",
                decompressed_data.has_data(),
                decompressed_data.len(),
                decompressed_data.ndim(),
                decompressed_data.dtype()
            );

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

        let Ok(mut compressor) = self.compressor.compressor.lock() else {
            return Err(PressioCodecError::PressioPoisonedMutex);
        };

        match encoded {
            AnyCowArray::U8(encoded) => decode_typed(&mut compressor, encoded),
            AnyCowArray::U16(encoded) => decode_typed(&mut compressor, encoded),
            AnyCowArray::U32(encoded) => decode_typed(&mut compressor, encoded),
            AnyCowArray::U64(encoded) => decode_typed(&mut compressor, encoded),
            AnyCowArray::I8(encoded) => decode_typed(&mut compressor, encoded),
            AnyCowArray::I16(encoded) => decode_typed(&mut compressor, encoded),
            AnyCowArray::I32(encoded) => decode_typed(&mut compressor, encoded),
            AnyCowArray::I64(encoded) => decode_typed(&mut compressor, encoded),
            AnyCowArray::F32(encoded) => decode_typed(&mut compressor, encoded),
            AnyCowArray::F64(encoded) => decode_typed(&mut compressor, encoded),
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
                    .compress(encoded, decompressed_data)
                    .map_err(|err| PressioCodecError::PressioDecodeFailed {
                        source: PressioCodingError(err),
                    })
            })
        }

        fn decode_into_typed<T: libpressio::PressioElement>(
            decompressed_data: &libpressio::PressioData,
            mut decoded: ArrayViewMut<T, IxDyn>,
        ) -> Result<(), PressioCodecError> {
            eprintln!(
                "decompressed into: {} {} {} {:?}",
                decompressed_data.has_data(),
                decompressed_data.len(),
                decompressed_data.ndim(),
                decompressed_data.dtype()
            );

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

        let Ok(mut compressor) = self.compressor.compressor.lock() else {
            return Err(PressioCodecError::PressioPoisonedMutex);
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
                decompress_typed(&mut compressor, encoded, decoded_dtype, decoded_shape)
            }
            AnyArrayView::U16(encoded) => {
                decompress_typed(&mut compressor, encoded, decoded_dtype, decoded_shape)
            }
            AnyArrayView::U32(encoded) => {
                decompress_typed(&mut compressor, encoded, decoded_dtype, decoded_shape)
            }
            AnyArrayView::U64(encoded) => {
                decompress_typed(&mut compressor, encoded, decoded_dtype, decoded_shape)
            }
            AnyArrayView::I8(encoded) => {
                decompress_typed(&mut compressor, encoded, decoded_dtype, decoded_shape)
            }
            AnyArrayView::I16(encoded) => {
                decompress_typed(&mut compressor, encoded, decoded_dtype, decoded_shape)
            }
            AnyArrayView::I32(encoded) => {
                decompress_typed(&mut compressor, encoded, decoded_dtype, decoded_shape)
            }
            AnyArrayView::I64(encoded) => {
                decompress_typed(&mut compressor, encoded, decoded_dtype, decoded_shape)
            }
            AnyArrayView::F32(encoded) => {
                decompress_typed(&mut compressor, encoded, decoded_dtype, decoded_shape)
            }
            AnyArrayView::F64(encoded) => {
                decompress_typed(&mut compressor, encoded, decoded_dtype, decoded_shape)
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
    PressioPoisonedMutex,
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
