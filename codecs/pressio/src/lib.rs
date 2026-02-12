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

use ::libpressio as _;
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayView, AnyArrayViewMut, AnyCowArray, Codec, StaticCodec,
    StaticCodecConfig, StaticCodecVersion,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
/// Pressio codec which applies the identity function, i.e. passes through the
/// input unchanged during encoding and decoding.
pub struct PressioCodec {
    /// The codec's encoding format version. Do not provide this parameter explicitly.
    #[serde(default, rename = "_version")]
    pub version: StaticCodecVersion<1, 0, 0>,
}

impl Codec for PressioCodec {
    type Error = PressioCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        Ok(data.into_owned())
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        Ok(encoded.into_owned())
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        mut decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        Ok(decoded.assign(&encoded)?)
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
    /// [`PressioCodec`] cannot decode into the provided array
    #[error("Pressio cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}
