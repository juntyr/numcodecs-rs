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

use numcodecs::{
    AnyArray, AnyArrayView, AnyArrayViewMut, AnyCowArray, Codec, StaticCodec, StaticCodecConfig,
    StaticCodecVersion,
};
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

static PRESSIO: LazyLock<Pressio> = LazyLock::new(Pressio::new);

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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
        let compressor = pressio.get_compressor(self.format.id.as_str()).unwrap();
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
            .get_compressor(format.id.as_str())
            .map_err(|err| serde::de::Error::custom(err.message))?;
        let mut options = compressor
            .get_options()
            .map_err(|err| serde::de::Error::custom(err.message))?;

        for (key, value) in &format.options {
            options = options
                .set(
                    key,
                    match value {
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

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename = "PressioCompressor")]
struct PressioCompressorFormat {
    id: String,
    #[serde(flatten)]
    options: BTreeMap<String, PressioOption>,
}

#[expect(missing_docs)]
#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
/// Pressio option value
pub enum PressioOption {
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

    fn encode(&self, _data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        Err(PressioCodecError::Unimplemented)
    }

    fn decode(&self, _encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        Err(PressioCodecError::Unimplemented)
    }

    fn decode_into(
        &self,
        _encoded: AnyArrayView,
        _decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        Err(PressioCodecError::Unimplemented)
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
    /// [`PressioCodec`] does not yet implement this functionality
    #[error("Pressio does not yet implement this functionality")]
    Unimplemented,
}

// FIXME: don't stub
#[expect(unsafe_code)]
#[unsafe(no_mangle)]
const extern "C" fn pressio_register_all() {}
