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

use std::{borrow::Cow, collections::BTreeMap, sync::LazyLock};

use ndarray::{CowArray, IxDyn};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig, StaticCodecVersion,
};
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

static PRESSIO: LazyLock<Pressio> = LazyLock::new(Pressio::new);

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
    compressor: libpressio::PressioCompressor,
}

// FIXME: UNSOUND
#[expect(unsafe_code, clippy::non_send_fields_in_send_ty)]
unsafe impl Send for PressioCompressor {}
#[expect(unsafe_code)]
unsafe impl Sync for PressioCompressor {}

impl Clone for PressioCompressor {
    #[expect(clippy::unwrap_used)]
    fn clone(&self) -> Self {
        let pressio = PRESSIO.get_or_unwrap();
        let compressor = pressio
            .get_compressor(self.format.compressor.as_str())
            .unwrap();
        let options = self.compressor.get_options().unwrap();
        compressor.set_options(&options).unwrap();

        Self {
            format: self.format.clone(),
            compressor,
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
        let pressio = PRESSIO
            .get()
            .map_err(|err| serde::de::Error::custom(err.message.as_str()))?;
        // TODO: better error handling
        let format = PressioCompressorFormat::deserialize(deserializer)?;
        let compressor = pressio
            .get_compressor(format.compressor.as_str())
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
        let mut options = compressor
            .get_options()
            .map_err(|err| serde::de::Error::custom(err.message))?;

        for (key, value) in &format.options {
            options = options
                .set(
                    key,
                    match value {
                        PressioOption::Bool(x) => libpressio::PressioOption::bool(Some(*x)),
                        PressioOption::U8(x) => libpressio::PressioOption::uint8(Some(*x)),
                        PressioOption::I8(x) => libpressio::PressioOption::int8(Some(*x)),
                        PressioOption::U16(x) => libpressio::PressioOption::uint16(Some(*x)),
                        PressioOption::I16(x) => libpressio::PressioOption::int16(Some(*x)),
                        PressioOption::U32(x) => libpressio::PressioOption::uint32(Some(*x)),
                        PressioOption::I32(x) => libpressio::PressioOption::int32(Some(*x)),
                        PressioOption::U64(x) => libpressio::PressioOption::uint64(Some(*x)),
                        PressioOption::I64(x) => libpressio::PressioOption::int64(Some(*x)),
                        PressioOption::F32(x) => libpressio::PressioOption::float32(Some(*x)),
                        PressioOption::F64(x) => libpressio::PressioOption::float64(Some(*x)),
                        PressioOption::String(x) => {
                            libpressio::PressioOption::string(Some(x.clone()))
                        }
                        PressioOption::VecString(x) => {
                            libpressio::PressioOption::vec_string(Some(x.clone()))
                        }
                    },
                )
                .map_err(|err| serde::de::Error::custom(err.message))?;
        }

        let mut format = format;
        if let Ok(format_options) = options.get_options() {
            format.options = format_options
                .into_iter()
                .filter_map(|(k, v)| match v {
                    libpressio::PressioOption::bool(Some(x)) => Some((k, PressioOption::Bool(x))),
                    libpressio::PressioOption::int8(Some(x)) => Some((k, PressioOption::I8(x))),
                    libpressio::PressioOption::int16(Some(x)) => Some((k, PressioOption::I16(x))),
                    libpressio::PressioOption::int32(Some(x)) => Some((k, PressioOption::I32(x))),
                    libpressio::PressioOption::int64(Some(x)) => Some((k, PressioOption::I64(x))),
                    libpressio::PressioOption::uint8(Some(x)) => Some((k, PressioOption::U8(x))),
                    libpressio::PressioOption::uint16(Some(x)) => Some((k, PressioOption::U16(x))),
                    libpressio::PressioOption::uint32(Some(x)) => Some((k, PressioOption::U32(x))),
                    libpressio::PressioOption::uint64(Some(x)) => Some((k, PressioOption::U64(x))),
                    libpressio::PressioOption::float32(Some(x)) => Some((k, PressioOption::F32(x))),
                    libpressio::PressioOption::float64(Some(x)) => Some((k, PressioOption::F64(x))),
                    libpressio::PressioOption::string(Some(x)) => {
                        Some((k, PressioOption::String(x)))
                    }
                    // FIXME: seems to return strings as a single joined string
                    libpressio::PressioOption::vec_string(Some(x)) => {
                        Some((k, PressioOption::VecString(x)))
                    }
                    _ => None,
                })
                .collect();
        }
        let format = format;

        Ok(Self { format, compressor })
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
    compressor: String,
    // TODO: flatten
    #[serde(default)]
    options: BTreeMap<String, PressioOption>,
}

#[expect(missing_docs)]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
/// Pressio option value
pub enum PressioOption {
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
}

struct Pressio {
    pressio: Result<libpressio::Pressio, libpressio::PressioError>,
}

impl Pressio {
    fn new() -> Self {
        Self {
            pressio: libpressio::Pressio::new(),
        }
    }

    const fn get(&self) -> Result<&libpressio::Pressio, &libpressio::PressioError> {
        self.pressio.as_ref()
    }

    #[expect(clippy::unwrap_used)]
    fn get_or_unwrap(&self) -> &libpressio::Pressio {
        self.pressio.as_ref().unwrap()
    }
}

// FIXME: UNSOUND
#[expect(unsafe_code, clippy::non_send_fields_in_send_ty)]
unsafe impl Send for Pressio {}
#[expect(unsafe_code)]
unsafe impl Sync for Pressio {}

impl Codec for PressioCodec {
    type Error = PressioCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        fn encode_typed<T: libpressio::PressioElement>(
            compressor: &libpressio::PressioCompressor,
            data: CowArray<T, IxDyn>,
        ) -> Result<AnyArray, PressioCodecError> {
            let data = match data.try_into_owned_nocopy() {
                Ok(data) => libpressio::PressioData::new(data),
                Err(data) => libpressio::PressioData::new_copied(data.view()),
            };

            let compressed_data =
                libpressio::PressioData::new_empty(libpressio::PressioDtype::Byte, []);

            let compressed_data = compressor.compress(&data, compressed_data).map_err(|err| {
                PressioCodecError::PressioEncodeFailed {
                    source: PressioCodingError(err),
                }
            })?;

            let Some(compressed_data) = compressed_data.clone_into_array() else {
                return Err(PressioCodecError::EncodeToUnknownDtype);
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

        match data {
            AnyCowArray::U8(data) => encode_typed(&self.compressor.compressor, data),
            AnyCowArray::U16(data) => encode_typed(&self.compressor.compressor, data),
            AnyCowArray::U32(data) => encode_typed(&self.compressor.compressor, data),
            AnyCowArray::U64(data) => encode_typed(&self.compressor.compressor, data),
            AnyCowArray::I8(data) => encode_typed(&self.compressor.compressor, data),
            AnyCowArray::I16(data) => encode_typed(&self.compressor.compressor, data),
            AnyCowArray::I32(data) => encode_typed(&self.compressor.compressor, data),
            AnyCowArray::I64(data) => encode_typed(&self.compressor.compressor, data),
            AnyCowArray::F32(data) => encode_typed(&self.compressor.compressor, data),
            AnyCowArray::F64(data) => encode_typed(&self.compressor.compressor, data),
            data => Err(PressioCodecError::UnsupportedDtype(data.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        fn decode_typed<T: libpressio::PressioElement>(
            compressor: &libpressio::PressioCompressor,
            encoded: CowArray<T, IxDyn>,
        ) -> Result<AnyArray, PressioCodecError> {
            let encoded = match encoded.try_into_owned_nocopy() {
                Ok(encoded) => libpressio::PressioData::new(encoded),
                Err(encoded) => libpressio::PressioData::new_copied(encoded.view()),
            };

            let decompressed_data =
                libpressio::PressioData::new_empty(libpressio::PressioDtype::Byte, []);

            let decompressed_data =
                compressor
                    .compress(&encoded, decompressed_data)
                    .map_err(|err| PressioCodecError::PressioDecodeFailed {
                        source: PressioCodingError(err),
                    })?;

            let Some(decompressed_data) = decompressed_data.clone_into_array() else {
                return Err(PressioCodecError::DecodeToUnknownDtype);
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

        match encoded {
            AnyCowArray::U8(encoded) => decode_typed(&self.compressor.compressor, encoded),
            AnyCowArray::U16(encoded) => decode_typed(&self.compressor.compressor, encoded),
            AnyCowArray::U32(encoded) => decode_typed(&self.compressor.compressor, encoded),
            AnyCowArray::U64(encoded) => decode_typed(&self.compressor.compressor, encoded),
            AnyCowArray::I8(encoded) => decode_typed(&self.compressor.compressor, encoded),
            AnyCowArray::I16(encoded) => decode_typed(&self.compressor.compressor, encoded),
            AnyCowArray::I32(encoded) => decode_typed(&self.compressor.compressor, encoded),
            AnyCowArray::I64(encoded) => decode_typed(&self.compressor.compressor, encoded),
            AnyCowArray::F32(encoded) => decode_typed(&self.compressor.compressor, encoded),
            AnyCowArray::F64(encoded) => decode_typed(&self.compressor.compressor, encoded),
            encoded => Err(PressioCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        mut decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        // TODO: optimize
        let decoded_in = self.decode(encoded.cow())?;

        Ok(decoded.assign(&decoded_in)?)
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
    /// [`PressioCodec`] failed to encode the data
    #[error("Pressio failed to encode the data")]
    PressioEncodeFailed {
        /// Opaque source error
        source: PressioCodingError,
    },
    /// [`PressioCodec`] encoded to an unknown unsupported dtype
    #[error("Pressio encoded to an unknown unsupported dtype")]
    EncodeToUnknownDtype,
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
