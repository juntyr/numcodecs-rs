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

use ndarray::{ArrayViewD, ArrayViewMutD};
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
        fn shape_checked_assign<T: Copy>(
            encoded: &ArrayViewD<T>,
            decoded: &mut ArrayViewMutD<T>,
        ) -> Result<(), IdentityCodecError> {
            #[allow(clippy::unit_arg)]
            if encoded.shape() == decoded.shape() {
                Ok(decoded.assign(encoded))
            } else {
                Err(IdentityCodecError::MismatchedDecodeIntoShape {
                    decoded: encoded.shape().to_vec(),
                    provided: decoded.shape().to_vec(),
                })
            }
        }

        match (&encoded, &mut decoded) {
            (AnyArrayView::U8(encoded), AnyArrayViewMut::U8(decoded)) => {
                shape_checked_assign(encoded, decoded)
            }
            (AnyArrayView::U16(encoded), AnyArrayViewMut::U16(decoded)) => {
                shape_checked_assign(encoded, decoded)
            }
            (AnyArrayView::U32(encoded), AnyArrayViewMut::U32(decoded)) => {
                shape_checked_assign(encoded, decoded)
            }
            (AnyArrayView::U64(encoded), AnyArrayViewMut::U64(decoded)) => {
                shape_checked_assign(encoded, decoded)
            }
            (AnyArrayView::I8(encoded), AnyArrayViewMut::I8(decoded)) => {
                shape_checked_assign(encoded, decoded)
            }
            (AnyArrayView::I16(encoded), AnyArrayViewMut::I16(decoded)) => {
                shape_checked_assign(encoded, decoded)
            }
            (AnyArrayView::I32(encoded), AnyArrayViewMut::I32(decoded)) => {
                shape_checked_assign(encoded, decoded)
            }
            (AnyArrayView::I64(encoded), AnyArrayViewMut::I64(decoded)) => {
                shape_checked_assign(encoded, decoded)
            }
            (AnyArrayView::F32(encoded), AnyArrayViewMut::F32(decoded)) => {
                shape_checked_assign(encoded, decoded)
            }
            (AnyArrayView::F64(encoded), AnyArrayViewMut::F64(decoded)) => {
                shape_checked_assign(encoded, decoded)
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
    /// [`IdentityCodec`] cannot decode the decoded array into the provided
    /// array of a different shape
    #[error("Identity cannot decode the decoded array of shape {decoded:?} into the provided array of shape {provided:?}")]
    MismatchedDecodeIntoShape {
        /// Shape of the `decoded` data
        decoded: Vec<usize>,
        /// Shape of the `provided` array into which the data is to be decoded
        provided: Vec<usize>,
    },
}
