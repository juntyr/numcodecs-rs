//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.76.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-identity
//! [crates.io]: https://crates.io/crates/numcodecs-identity
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-identity
//! [docs.rs]: https://docs.rs/numcodecs-identity/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_identity
//!
//! Identity codec implementation for the [`numcodecs`] API.

use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayView, AnyArrayViewMut, AnyCowArray, Codec, StaticCodec,
    StaticCodecConfig,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
/// Identity codec which applies the identity function, i.e. passes through the
/// input unchanged during encoding and decoding.
pub struct IdentityCodec {
    // empty
}

impl Codec for IdentityCodec {
    type Error = IdentityCodecError;

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

impl StaticCodec for IdentityCodec {
    const CODEC_ID: &'static str = "identity";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<Self> {
        StaticCodecConfig::from(self)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`IdentityCodec`].
pub enum IdentityCodecError {
    /// [`IdentityCodec`] cannot decode into the provided array
    #[error("Identity cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}
