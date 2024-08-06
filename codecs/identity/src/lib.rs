//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.64.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-identity
//! [crates.io]: https://crates.io/crates/numcodecs-identity
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-identity
//! [docs.rs]: https://docs.rs/numcodecs-identity/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs-identity
//!
//! Bit rounding codec implementation for the [`numcodecs`] API.

use numcodecs::{
    AnyArray, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray, Codec, StaticCodec,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

#[derive(Clone, Serialize, Deserialize)]
/// Identity codec which applies the identity function, i.e. passes through the
/// input unchanged during encoding and decoding.
pub struct IdentityCodec;

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
        #[allow(clippy::unit_arg)]
        match (&encoded, &mut decoded) {
            (AnyArrayView::U8(encoded), AnyArrayViewMut::U8(decoded)) => {
                Ok(decoded.assign(encoded))
            }
            (AnyArrayView::U16(encoded), AnyArrayViewMut::U16(decoded)) => {
                Ok(decoded.assign(encoded))
            }
            (AnyArrayView::U32(encoded), AnyArrayViewMut::U32(decoded)) => {
                Ok(decoded.assign(encoded))
            }
            (AnyArrayView::U64(encoded), AnyArrayViewMut::U64(decoded)) => {
                Ok(decoded.assign(encoded))
            }
            (AnyArrayView::I8(encoded), AnyArrayViewMut::I8(decoded)) => {
                Ok(decoded.assign(encoded))
            }
            (AnyArrayView::I16(encoded), AnyArrayViewMut::I16(decoded)) => {
                Ok(decoded.assign(encoded))
            }
            (AnyArrayView::I32(encoded), AnyArrayViewMut::I32(decoded)) => {
                Ok(decoded.assign(encoded))
            }
            (AnyArrayView::I64(encoded), AnyArrayViewMut::I64(decoded)) => {
                Ok(decoded.assign(encoded))
            }
            (AnyArrayView::F32(encoded), AnyArrayViewMut::F32(decoded)) => {
                Ok(decoded.assign(encoded))
            }
            (AnyArrayView::F64(encoded), AnyArrayViewMut::F64(decoded)) => {
                Ok(decoded.assign(encoded))
            }
            (encoded, decoded) => Err(IdentityCodecError::MismatchedDecodeIntoDtype {
                decoded: encoded.dtype(),
                provided: decoded.dtype(),
            }),
        }
    }

    fn get_config<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.serialize(serializer)
    }
}

impl StaticCodec for IdentityCodec {
    const CODEC_ID: &'static str = "identity";

    fn from_config<'de, D: Deserializer<'de>>(config: D) -> Result<Self, D::Error> {
        Self::deserialize(config)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`IdentityCodec`].
pub enum IdentityCodecError {
    /// [`IdentityCodec`] cannot decode the `decoded` dtype into the `provided`
    /// array
    #[error("Identity cannot decode the dtype {decoded} into the provided {provided} array")]
    MismatchedDecodeIntoDtype {
        /// Dtype of the `decoded` data
        decoded: AnyArrayDType,
        /// Dtype of the `provided` array into which the data is to be decoded
        provided: AnyArrayDType,
    },
}
