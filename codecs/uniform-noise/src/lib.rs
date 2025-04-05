//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.82.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-uniform-noise
//! [crates.io]: https://crates.io/crates/numcodecs-uniform-noise
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-uniform-noise
//! [docs.rs]: https://docs.rs/numcodecs-uniform-noise/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_uniform_noise
//!
//! Uniform noise codec implementation for the [`numcodecs`] API.

use std::hash::{Hash, Hasher};

use ndarray::{Array, ArrayBase, Data, Dimension};
use num_traits::Float;
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig, StaticCodecVersion,
};
use rand::{
    distributions::{Distribution, Open01},
    SeedableRng,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use wyhash::{WyHash, WyRng};

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
/// Codec that adds `seed`ed `$\text{U}(-0.5 \cdot scale, 0.5 \cdot scale)$`
/// uniform noise of the given `scale` during encoding and passes through the
/// input unchanged during decoding.
///
/// This codec first hashes the input array data and shape to then seed a
/// pseudo-random number generator that generates the uniform noise. Therefore,
/// passing in the same input with the same `seed` will produce the same noise
/// and thus the same encoded output.
pub struct UniformNoiseCodec {
    /// Scale of the uniform noise, which is sampled from
    /// `$\text{U}(-0.5 \cdot scale, 0.5 \cdot scale)$`
    pub scale: f64,
    /// Seed for the random noise generator
    pub seed: u64,
    /// The codec's encoding format version. Do not provide this parameter explicitly.
    #[serde(default, rename = "_version")]
    pub version: StaticCodecVersion<1, 0, 0>,
}

impl Codec for UniformNoiseCodec {
    type Error = UniformNoiseCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            #[expect(clippy::cast_possible_truncation)]
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
        if !matches!(encoded.dtype(), AnyArrayDType::F32 | AnyArrayDType::F64) {
            return Err(UniformNoiseCodecError::UnsupportedDtype(encoded.dtype()));
        }

        Ok(decoded.assign(&encoded)?)
    }
}

impl StaticCodec for UniformNoiseCodec {
    const CODEC_ID: &'static str = "uniform-noise.rs";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<Self> {
        StaticCodecConfig::from(self)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`UniformNoiseCodec`].
pub enum UniformNoiseCodecError {
    /// [`UniformNoiseCodec`] does not support the dtype
    #[error("UniformNoise does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`UniformNoiseCodec`] cannot decode into the provided array
    #[error("UniformNoise cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

/// Adds `$\text{U}(-0.5 \cdot scale, 0.5 \cdot scale)$` uniform random noise
/// to the input `data`.
///
/// This function first hashes the input and its shape to then seed a pseudo-
/// random number generator that generates the uniform noise. Therefore,
/// passing in the same input with the same `seed` will produce the same noise
/// and thus the same output.
#[must_use]
pub fn add_uniform_noise<T: FloatExt, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
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
pub trait FloatExt: Float {
    /// -0.5
    const NEG_HALF: Self;

    /// Hash the binary representation of the floating point value
    fn hash_bits<H: Hasher>(self, hasher: &mut H);
}

impl FloatExt for f32 {
    const NEG_HALF: Self = -0.5;

    fn hash_bits<H: Hasher>(self, hasher: &mut H) {
        hasher.write_u32(self.to_bits());
    }
}

impl FloatExt for f64 {
    const NEG_HALF: Self = -0.5;

    fn hash_bits<H: Hasher>(self, hasher: &mut H) {
        hasher.write_u64(self.to_bits());
    }
}
