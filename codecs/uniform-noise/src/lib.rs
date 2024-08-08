//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.64.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-uniform-noise
//! [crates.io]: https://crates.io/crates/numcodecs-uniform-noise
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-uniform-noise
//! [docs.rs]: https://docs.rs/numcodecs-uniform-noise/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs-uniform-noise
//!
//! Uniform noise codec implementation for the [`numcodecs`] API.

use std::hash::{Hash, Hasher};

use ndarray::{Array, ArrayViewD, ArrayViewMutD, CowArray, Dimension};
use numcodecs::{
    AnyArray, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray, Codec, StaticCodec,
};
use rand::{
    distributions::{Distribution, Open01},
    SeedableRng,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;
use wyhash::{WyHash, WyRng};

#[derive(Clone, Serialize, Deserialize)]
/// Codec that adds `seed`ed uniform noise of the given `scale` and with
/// [`add_uniform_noise`] during encoding and passes through the input unchanged
/// during decoding.
pub struct UniformNoiseCodec {
    /// Scale of the uniform noise, which is sampled from
    /// `U(-scale/2, +scale/2)`
    pub scale: f64,
    /// Seed for the random noise generator
    pub seed: u64,
}

impl Codec for UniformNoiseCodec {
    type Error = UniformNoiseCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            #[allow(clippy::cast_possible_truncation)]
            AnyCowArray::F32(data) => Ok(AnyArray::F32(add_uniform_noise(
                data,
                self.scale as f32,
                self.seed,
            ))),
            AnyCowArray::F64(data) => Ok(AnyArray::F64(add_uniform_noise(
                data, self.scale, self.seed,
            ))),
            encoded => Err(UniformNoiseCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match encoded {
            AnyCowArray::F32(encoded) => Ok(AnyArray::F32(encoded.into_owned())),
            AnyCowArray::F64(encoded) => Ok(AnyArray::F64(encoded.into_owned())),
            encoded => Err(UniformNoiseCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        mut decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        fn shape_checked_assign<T: Copy>(
            encoded: &ArrayViewD<T>,
            decoded: &mut ArrayViewMutD<T>,
        ) -> Result<(), UniformNoiseCodecError> {
            #[allow(clippy::unit_arg)]
            if encoded.shape() == decoded.shape() {
                Ok(decoded.assign(encoded))
            } else {
                Err(UniformNoiseCodecError::MismatchedDecodeIntoShape {
                    decoded: encoded.shape().to_vec(),
                    provided: decoded.shape().to_vec(),
                })
            }
        }

        match (&encoded, &mut decoded) {
            (AnyArrayView::F32(encoded), AnyArrayViewMut::F32(decoded)) => {
                shape_checked_assign(encoded, decoded)
            }
            (AnyArrayView::F64(encoded), AnyArrayViewMut::F64(decoded)) => {
                shape_checked_assign(encoded, decoded)
            }
            (AnyArrayView::F32(_), decoded) => {
                Err(UniformNoiseCodecError::MismatchedDecodeIntoDtype {
                    decoded: AnyArrayDType::F32,
                    provided: decoded.dtype(),
                })
            }
            (AnyArrayView::F64(_), decoded) => {
                Err(UniformNoiseCodecError::MismatchedDecodeIntoDtype {
                    decoded: AnyArrayDType::F64,
                    provided: decoded.dtype(),
                })
            }
            (encoded, _decoded) => Err(UniformNoiseCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn get_config<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.serialize(serializer)
    }
}

impl StaticCodec for UniformNoiseCodec {
    const CODEC_ID: &'static str = "uniform-noise";

    fn from_config<'de, D: Deserializer<'de>>(config: D) -> Result<Self, D::Error> {
        Self::deserialize(config)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`UniformNoiseCodec`].
pub enum UniformNoiseCodecError {
    /// [`UniformNoiseCodec`] does not support the dtype
    #[error("UniformNoise does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`UniformNoiseCodec`] cannot decode the `decoded` dtype into the `provided`
    /// array
    #[error("UniformNoise cannot decode the dtype {decoded} into the provided {provided} array")]
    MismatchedDecodeIntoDtype {
        /// Dtype of the `decoded` data
        decoded: AnyArrayDType,
        /// Dtype of the `provided` array into which the data is to be decoded
        provided: AnyArrayDType,
    },
    /// [`UniformNoiseCodec`] cannot decode the decoded array into the provided
    /// array of a different shape
    #[error("UniformNoise cannot decode the decoded array of shape {decoded:?} into the provided array of shape {provided:?}")]
    MismatchedDecodeIntoShape {
        /// Shape of the `decoded` data
        decoded: Vec<usize>,
        /// Shape of the `provided` array into which the data is to be decoded
        provided: Vec<usize>,
    },
}

/// Adds `U(-scale/2, scale/2)` uniform random noise to the input `data`.
///
/// This function first hashes the input and its shape to then seed a pseudo-
/// random number generator that generates the uniform noise. Therefore,
/// passing in the same input with the same `seed` will produce the same noise
/// and thus the same output.
#[must_use]
pub fn add_uniform_noise<T: Float, D: Dimension>(
    data: CowArray<T, D>,
    scale: T,
    seed: u64,
) -> Array<T, D>
where
    Open01: Distribution<T>,
{
    let mut hasher = WyHash::with_seed(seed);
    // hashing the shape provides a prefix for the flattened data
    data.shape().hash(&mut hasher);
    // the data must be visited in a defined order
    data.iter().copied().for_each(|x| x.hash_bits(&mut hasher));
    let seed = hasher.finish();

    let mut rng: WyRng = WyRng::seed_from_u64(seed);

    let mut encoded = data.into_owned();

    // the data must be visited in a defined order
    for x in &mut encoded {
        // x = U(0,1)*scale + (scale*-0.5 + x)
        // --- is equivalent to ---
        // x += U(-scale/2, +scale/2)
        *x = Open01
            .sample(&mut rng)
            .mul_add(scale, scale.mul_add(T::NEG_HALF, *x));
    }

    encoded
}

/// Floating point types
pub trait Float: Copy {
    /// -0.5
    const NEG_HALF: Self;

    #[must_use]
    /// Compute (self * a) + b
    fn mul_add(self, a: Self, b: Self) -> Self;

    /// Hash the binary representation of the floating point value
    fn hash_bits<H: Hasher>(self, hasher: &mut H);
}

impl Float for f32 {
    const NEG_HALF: Self = -0.5;

    fn mul_add(self, a: Self, b: Self) -> Self {
        Self::mul_add(self, a, b)
    }

    fn hash_bits<H: Hasher>(self, hasher: &mut H) {
        hasher.write_u32(self.to_bits());
    }
}

impl Float for f64 {
    const NEG_HALF: Self = -0.5;

    fn mul_add(self, a: Self, b: Self) -> Self {
        Self::mul_add(self, a, b)
    }

    fn hash_bits<H: Hasher>(self, hasher: &mut H) {
        hasher.write_u64(self.to_bits());
    }
}
