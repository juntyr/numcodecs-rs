//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.88.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-onion
//! [crates.io]: https://crates.io/crates/numcodecs-onion
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-onion
//! [docs.rs]: https://docs.rs/numcodecs-onion/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_onion
//!
//! Onion identity meta-codec implementation for the [`numcodecs`] API.

use numcodecs::{
    AnyArray, AnyArrayView, AnyArrayViewMut, AnyCowArray, Codec, DynCodec, ErasedDynCodec,
    ErasedError, StaticCodec, StaticCodecConfig, StaticCodecVersion,
};
use numcodecs_registry::GlobalRegistry;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
/// Onion identity meta-codec which wraps an existing codec and passes the
/// inputs and outputs unchanged to and from this codec during encoding and
/// decoding.
pub struct OnionCodec {
    /// The configuration of the wrapped codec.
    #[serde(serialize_with = "DynCodec::get_config")]
    #[serde(deserialize_with = "GlobalRegistry::codec_from_config")]
    #[schemars(schema_with = "ErasedDynCodec::codec_config_schema")]
    pub codec: ErasedDynCodec,
    /// The codec's encoding format version. Do not provide this parameter explicitly.
    #[serde(default, rename = "_version")]
    pub version: StaticCodecVersion<1, 0, 0>,
}

impl Codec for OnionCodec {
    type Error = OnionCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        self.codec
            .encode(data)
            .map_err(|err| OnionCodecError { error: err })
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        self.codec
            .decode(encoded)
            .map_err(|err| OnionCodecError { error: err })
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        self.codec
            .decode_into(encoded, decoded)
            .map_err(|err| OnionCodecError { error: err })
    }
}

impl StaticCodec for OnionCodec {
    const CODEC_ID: &'static str = "onion.rs";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<'_, Self> {
        StaticCodecConfig::from(self)
    }
}

#[derive(Debug, Error)]
/// Error that may occur when applying the [`OnionCodec`].
#[error(transparent)]
pub struct OnionCodecError {
    error: ErasedError,
}
