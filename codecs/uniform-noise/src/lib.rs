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
//! Bit rounding codec implementation for the [`numcodecs`] API.

use std::hash::{Hash, Hasher};

use ndarray::{Array, CowArray, Dimension};
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
/// [`add_uniform_noise`].
pub struct UniformNoiseCodec {
    /// Scale of the uniform noise, which is sampled from
    /// `U(-scale/2, +scale/2)`
    pub scale: f64,
    /// Seed for the random noise generator
    pub seed: u64,
}

impl Codec for UniformNoiseCodec {
    type Error = UniformNoiseError;

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
            encoded => Err(UniformNoiseError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match encoded {
            AnyCowArray::F32(encoded) => Ok(AnyArray::F32(encoded.into_owned())),
            AnyCowArray::F64(encoded) => Ok(AnyArray::F64(encoded.into_owned())),
            encoded => Err(UniformNoiseError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        mut decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        #[allow(clippy::unit_arg)]
        match (&encoded, &mut decoded) {
            (AnyArrayView::F32(encoded), AnyArrayViewMut::F32(decoded)) => {
                Ok(decoded.assign(encoded))
            }
            (AnyArrayView::F64(encoded), AnyArrayViewMut::F64(decoded)) => {
                Ok(decoded.assign(encoded))
            }
            (AnyArrayView::F32(_), decoded) => Err(UniformNoiseError::MismatchedDecodeIntoDtype {
                decoded: AnyArrayDType::F32,
                provided: decoded.dtype(),
            }),
            (AnyArrayView::F64(_), decoded) => Err(UniformNoiseError::MismatchedDecodeIntoDtype {
                decoded: AnyArrayDType::F64,
                provided: decoded.dtype(),
            }),
            (encoded, _decoded) => Err(UniformNoiseError::UnsupportedDtype(encoded.dtype())),
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
pub enum UniformNoiseError {
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
}

/// Uniform noise codec which adds `U(-scale/2, scale/2)` uniform random noise
/// to the input on encoding and passes through the input unchanged during
/// decoding.
///
/// This codec first hashes the input and its shape to then seed a pseudo-random
/// number generator that generates the uniform noise. Therefore, encoding the
/// same data with the same seed will produce the same noise and thus the same
/// encoded data.
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
