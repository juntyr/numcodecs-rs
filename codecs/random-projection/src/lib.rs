//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.76.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-random-projection
//! [crates.io]: https://crates.io/crates/numcodecs-random-projection
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-random-projection
//! [docs.rs]: https://docs.rs/numcodecs-random-projection/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_random_projection
//!
//! Random Projection codec implementation for the [`numcodecs`] API.

use ndarray::{s, Array, ArrayBase, ArrayViewMut, Data, Dimension, Ix2, ShapeError};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RandomProjectionCodec {
    pub seed: u64,
    pub epsilon: f64, // TODO: (0, 1)
    #[serde(flatten)]
    pub projection: RandomProjection,
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[serde(tag = "projection", rename_all = "kebab-case")]
pub enum RandomProjection {
    Gaussian,
    Sparse {
        #[serde(skip_serializing_if = "Option::is_none")]
        density: Option<f64>, // TODO: (0, 1]
    },
}

impl Codec for RandomProjectionCodec {
    type Error = RandomProjectionCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            AnyCowArray::F32(data) => Ok(AnyArray::F32(
                project_with_projection(data, self.seed, self.epsilon, &self.projection)?
                    .into_dyn(),
            )),
            AnyCowArray::F64(data) => Ok(AnyArray::F64(
                project_with_projection(data, self.seed, self.epsilon, &self.projection)?
                    .into_dyn(),
            )),
            encoded => Err(RandomProjectionCodecError::UnsupportedDtype(
                encoded.dtype(),
            )),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match encoded {
            AnyCowArray::F32(encoded) => Ok(AnyArray::F32(
                reconstruct_with_projection(encoded, self.seed, &self.projection)?.into_dyn(),
            )),
            AnyCowArray::F64(encoded) => Ok(AnyArray::F64(
                reconstruct_with_projection(encoded, self.seed, &self.projection)?.into_dyn(),
            )),
            encoded => Err(RandomProjectionCodecError::UnsupportedDtype(
                encoded.dtype(),
            )),
        }
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        match (encoded, decoded) {
            (AnyArrayView::F32(encoded), AnyArrayViewMut::F32(decoded)) => {
                reconstruct_into_with_projection(encoded, decoded, self.seed, &self.projection)
            }
            (AnyArrayView::F64(encoded), AnyArrayViewMut::F64(decoded)) => {
                reconstruct_into_with_projection(encoded, decoded, self.seed, &self.projection)
            }
            (encoded @ (AnyArrayView::F32(_) | AnyArrayView::F64(_)), decoded) => {
                Err(RandomProjectionCodecError::MismatchedDecodeIntoArray {
                    source: AnyArrayAssignError::DTypeMismatch {
                        src: encoded.dtype(),
                        dst: decoded.dtype(),
                    },
                })
            }
            (encoded, _decoded) => Err(RandomProjectionCodecError::UnsupportedDtype(
                encoded.dtype(),
            )),
        }
    }
}

impl StaticCodec for RandomProjectionCodec {
    const CODEC_ID: &'static str = "random-projection";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<Self> {
        StaticCodecConfig::from(self)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`RandomProjectionCodec`].
pub enum RandomProjectionCodecError {
    /// [`RandomProjectionCodec`] does not support the dtype
    #[error("RandomProjection does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`RandomProjectionCodec`] does not support the dtype
    #[error("RandomProjection only supports matrix / 2d-shaped arrays")]
    NonMatrixData {
        /// The source of the error
        #[from]
        source: ShapeError,
    },
    /// [`RandomProjectionCodec`] does not support non-finite (infinite or NaN) floating
    /// point data
    #[error("RandomProjection does not support non-finite (infinite or NaN) floating point data")]
    NonFiniteData,
    /// [`RandomProjectionCodec`] cannot decode into the provided array
    #[error("RandomProjection cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

pub fn project_with_projection<T: Float, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
    seed: u64,
    epsilon: f64,
    projection: &RandomProjection,
) -> Result<Array<T, Ix2>, RandomProjectionCodecError> {
    let data = data
        .into_dimensionality()
        .map_err(|err| RandomProjectionCodecError::NonMatrixData { source: err })?;

    let (n, d) = data.dim();

    let k = johnson_lindenstrauss_min_k(n, epsilon);

    let mut projected = Array::<T, Ix2>::from_elem((n, k + 1), T::ZERO);

    for p in projected.slice_mut(s!(.., k)) {
        *p = T::from_usize(d);
    }

    match projection {
        RandomProjection::Gaussian => project_into(
            data,
            projected.slice_mut(s!(.., ..k)),
            |x, y| gaussian_project(x, y, seed),
            gaussian_normaliser(k),
        ),
        RandomProjection::Sparse { density } => {
            let density = density
                .map(T::from_f64)
                .unwrap_or(T::from_usize(d).sqrt().recip());
            project_into(
                data,
                projected.slice_mut(s!(.., ..k)),
                |x, y| sparse_project(x, y, density, seed),
                sparse_normaliser(k, density),
            )
        }
    }?;

    Ok(projected)
}

pub fn project_into<T: Float, S: Data<Elem = T>>(
    data: ArrayBase<S, Ix2>,
    mut projected: ArrayViewMut<T, Ix2>,
    projection: impl Fn(usize, usize) -> T,
    normalizer: T,
) -> Result<(), RandomProjectionCodecError> {
    let (n, d) = data.dim();

    if projected.dim().0 != n {
        panic!("must have the same number of samples");
    }

    for ((i, j), p) in projected.indexed_iter_mut() {
        let mut acc = T::ZERO;

        for l in 0..d {
            acc += data[(i, l)] * projection(l, j);
        }

        *p = acc * normalizer;
    }

    if !projected.iter().copied().all(T::is_finite) {
        return Err(RandomProjectionCodecError::NonFiniteData);
    }

    Ok(())
}

pub fn reconstruct_with_projection<T: Float, S: Data<Elem = T>, D: Dimension>(
    projected: ArrayBase<S, D>,
    seed: u64,
    projection: &RandomProjection,
) -> Result<Array<T, Ix2>, RandomProjectionCodecError> {
    let projected = projected
        .into_dimensionality()
        .map_err(|err| RandomProjectionCodecError::NonMatrixData { source: err })?;

    if projected.is_empty() {
        return Ok(projected.to_owned());
    }

    let (_n, k): (usize, usize) = projected.dim();
    let Some(k) = k.checked_sub(1) else {
        panic!("projected array must have non-zero dimensionality");
    };

    let ds = projected.slice(s!(.., k));
    let Ok(Some(d)) = ds.fold(Ok(None), |acc, d| match acc {
        Ok(None) => Ok(Some(d.into_usize())),
        Ok(Some(d2)) if d2 == d.into_usize() => Ok(Some(d2)),
        _ => Err(()),
    }) else {
        panic!("projected array must have consistent dimensionality metadata");
    };

    let projected = projected.slice_move(s!(.., ..k));

    match projection {
        RandomProjection::Gaussian => reconstruct(
            projected,
            d,
            |x, y| gaussian_project(x, y, seed),
            gaussian_normaliser(k),
        ),
        RandomProjection::Sparse { density } => {
            let density = density
                .map(T::from_f64)
                .unwrap_or(T::from_usize(d).sqrt().recip());
            reconstruct(
                projected,
                d,
                |x, y| sparse_project(x, y, density, seed),
                sparse_normaliser(k, density),
            )
        }
    }
}

pub fn reconstruct<T: Float, S: Data<Elem = T>>(
    projected: ArrayBase<S, Ix2>,
    d: usize,
    projection: impl Fn(usize, usize) -> T,
    normalizer: T,
) -> Result<Array<T, Ix2>, RandomProjectionCodecError> {
    if projected.is_empty() {
        return Ok(projected.to_owned());
    }

    let (n, k) = projected.dim();

    let reconstructed = Array::<T, Ix2>::from_shape_fn((n, d), |(i, j)| {
        let mut acc = T::ZERO;

        for l in 0..k {
            // projection[j,l] = transpose(projection)[l,j]
            acc += projected[(i, l)] * projection(j, l);
        }

        acc * normalizer
    });

    if !reconstructed.iter().copied().all(T::is_finite) {
        return Err(RandomProjectionCodecError::NonFiniteData);
    }

    Ok(reconstructed)
}

pub fn reconstruct_into_with_projection<T: Float, S: Data<Elem = T>, D: Dimension>(
    projected: ArrayBase<S, D>,
    reconstructed: ArrayViewMut<T, D>,
    seed: u64,
    projection: &RandomProjection,
) -> Result<(), RandomProjectionCodecError> {
    let projected: ArrayBase<S, Ix2> = projected
        .into_dimensionality()
        .map_err(|err| RandomProjectionCodecError::NonMatrixData { source: err })?;
    let reconstructed: ArrayViewMut<T, Ix2> = reconstructed
        .into_dimensionality()
        .map_err(|err| RandomProjectionCodecError::NonMatrixData { source: err })?;

    let (n, _k) = projected.dim();

    if reconstructed.dim().0 != n {
        panic!("must have the same number of samples");
    }

    let (_n, k): (usize, usize) = projected.dim();
    let Some(k) = k.checked_sub(1) else {
        panic!("projected array must have non-zero dimensionality");
    };

    let ds = projected.slice(s!(.., k));
    let Ok(Some(d)) = ds.fold(Ok(None), |acc, d| match acc {
        Ok(None) => Ok(Some(d.into_usize())),
        Ok(Some(d2)) if d2 == d.into_usize() => Ok(Some(d2)),
        _ => Err(()),
    }) else {
        panic!("projected array must have consistent dimensionality metadata");
    };

    if reconstructed.dim().1 != d {
        panic!("must have the same number of features");
    }

    let projected = projected.slice_move(s!(.., ..k));

    match projection {
        RandomProjection::Gaussian => reconstruct_into(
            projected,
            reconstructed,
            |x, y| gaussian_project(x, y, seed),
            gaussian_normaliser(k),
        ),
        RandomProjection::Sparse { density } => {
            let density = density
                .map(T::from_f64)
                .unwrap_or(T::from_usize(d).sqrt().recip());
            reconstruct_into(
                projected,
                reconstructed,
                |x, y| sparse_project(x, y, density, seed),
                sparse_normaliser(k, density),
            )
        }
    }
}

pub fn reconstruct_into<T: Float, S: Data<Elem = T>>(
    projected: ArrayBase<S, Ix2>,
    mut reconstructed: ArrayViewMut<T, Ix2>,
    projection: impl Fn(usize, usize) -> T,
    normalizer: T,
) -> Result<(), RandomProjectionCodecError> {
    let (n, k) = projected.dim();

    if reconstructed.dim().0 != n {
        panic!("must have the same number of samples");
    }

    for ((i, j), r) in reconstructed.indexed_iter_mut() {
        let mut acc = T::ZERO;

        for l in 0..k {
            // projection[j,l] = transpose(projection)[l,j]
            acc += projected[(i, l)] * projection(j, l);
        }

        *r = acc * normalizer;
    }

    if !reconstructed.iter().copied().all(T::is_finite) {
        return Err(RandomProjectionCodecError::NonFiniteData);
    }

    Ok(())
}

// https://github.com/scikit-learn/scikit-learn/blob/3b39d7cb957ab781744b346c1848be9db3f4e221/sklearn/random_projection.py#L56-L142
pub fn johnson_lindenstrauss_min_k(n_samples: usize, epsilon: f64) -> usize {
    let denominator = (epsilon * epsilon * 0.5) - (epsilon * epsilon * epsilon / 3.0);
    let min_k = (n_samples as f64).ln() * 4.0 / denominator;
    min_k as usize
}

fn gaussian_project<T: Float>(x: usize, y: usize, seed: u64) -> T {
    let (u0, u1) = T::u0_u1(hash_matrix_index(x, y, seed));

    let r = (-T::TWO * u0.ln()).sqrt();
    let theta = -T::TAU * u1;

    r * theta.sin()
}

fn gaussian_normaliser<T: Float>(k: usize) -> T {
    T::from_usize(k).sqrt().recip()
}

fn sparse_project<T: Float>(x: usize, y: usize, density: T, seed: u64) -> T {
    let (u0, _u1) = T::u0_u1(hash_matrix_index(x, y, seed));

    if u0 <= (density * T::HALF) {
        -T::ONE
    } else if u0 <= density {
        T::ONE
    } else {
        T::ZERO
    }
}

fn sparse_normaliser<T: Float>(k: usize, density: T) -> T {
    (T::from_usize(k) * density).recip().sqrt()
}

fn hash_matrix_index(x: usize, y: usize, seed: u64) -> u64 {
    seahash_diffuse(seahash_diffuse(x as u64) ^ seed ^ (y as u64))
}

#[inline]
const fn seahash_diffuse(mut x: u64) -> u64 {
    // SeaHash diffusion function
    // https://docs.rs/seahash/4.1.0/src/seahash/helper.rs.html#75-92

    // These are derived from the PCG RNG's round. Thanks to @Veedrac for proposing
    // this. The basic idea is that we use dynamic shifts, which are determined
    // by the input itself. The shift is chosen by the higher bits, which means
    // that changing those flips the lower bits, which scatters upwards because
    // of the multiplication.

    x = x.wrapping_mul(0x6eed_0e9d_a4d9_4a4f);

    let a = x >> 32;
    let b = x >> 60;

    x ^= a >> b;

    x = x.wrapping_mul(0x6eed_0e9d_a4d9_4a4f);

    x
}

/// Floating point types.
pub trait Float:
    Copy
    + PartialOrd
    + std::ops::Neg<Output = Self>
    + std::ops::AddAssign
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
{
    const ZERO: Self;
    const HALF: Self;
    const ONE: Self;
    const TWO: Self;
    const TAU: Self;

    /// Returns `true` if this number is neither infinite nor NaN.
    fn is_finite(self) -> bool;

    fn sqrt(self) -> Self;

    fn recip(self) -> Self;

    fn sin(self) -> Self;

    fn ln(self) -> Self;

    fn from_f64(x: f64) -> Self;

    fn from_usize(n: usize) -> Self;

    fn into_usize(self) -> usize;

    fn u0_u1(hash: u64) -> (Self, Self);
}

impl Float for f32 {
    const ZERO: Self = 0.0;
    const HALF: Self = 0.5;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const TAU: Self = std::f32::consts::TAU;

    fn is_finite(self) -> bool {
        self.is_finite()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn recip(self) -> Self {
        self.recip()
    }

    fn sin(self) -> Self {
        self.sin()
    }

    fn ln(self) -> Self {
        self.ln()
    }

    fn from_f64(x: f64) -> Self {
        x as f32
    }

    fn from_usize(n: usize) -> Self {
        n as Self
    }

    fn into_usize(self) -> usize {
        self as usize
    }

    fn u0_u1(hash: u64) -> (Self, Self) {
        let (low, high) = (
            (hash & u64::from(u32::MAX)) as u32,
            ((hash >> 32) & u64::from(u32::MAX)) as u32,
        );

        // http://prng.di.unimi.it -> Generating uniform doubles in the unit interval (0.0, 1.0]
        let u0 = (((low >> 8) + 1) as f32) * f32::from_bits(0x3380_0000_u32); // 0x1.0p-24

        // http://prng.di.unimi.it -> Generating uniform doubles in the unit interval [0.0, 1.0)
        let u1 = ((high >> 8) as f32) * f32::from_bits(0x3380_0000_u32); // 0x1.0p-24

        (u0, u1)
    }
}

impl Float for f64 {
    const ZERO: Self = 0.0;
    const HALF: Self = 0.5;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const TAU: Self = std::f64::consts::TAU;

    fn is_finite(self) -> bool {
        self.is_finite()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn recip(self) -> Self {
        self.recip()
    }

    fn sin(self) -> Self {
        self.sin()
    }

    fn ln(self) -> Self {
        self.ln()
    }

    fn from_f64(x: f64) -> Self {
        x
    }

    fn from_usize(n: usize) -> Self {
        n as Self
    }

    fn into_usize(self) -> usize {
        self as usize
    }

    fn u0_u1(hash: u64) -> (Self, Self) {
        // http://prng.di.unimi.it -> Generating uniform doubles in the unit interval (0.0, 1.0]
        let u0 = (((hash >> 11) + 1) as f64) * f64::from_bits(0x3CA0_0000_0000_0000_u64); // 0x1.0p-53

        let hash = seahash_diffuse(hash + 1);
        // http://prng.di.unimi.it -> Generating uniform doubles in the unit interval [0.0, 1.0)
        let u1 = ((hash >> 11) as f64) * f64::from_bits(0x3CA0_0000_0000_0000_u64); // 0x1.0p-53

        (u0, u1)
    }
}

#[cfg(test)]
mod tests {
    use ndarray_rand::rand_distr::{Distribution, Normal};
    use ndarray_rand::RandomExt;

    use super::*;

    #[test]
    fn gaussian_f32() {
        test_error_decline::<f32>(
            (10, 10),
            Normal::new(42.0, 24.0).unwrap(),
            42,
            RandomProjection::Gaussian,
        );
    }

    #[test]
    fn gaussian_f64() {
        test_error_decline::<f64>(
            (10, 10),
            Normal::new(42.0, 24.0).unwrap(),
            42,
            RandomProjection::Gaussian,
        );
    }

    #[test]
    fn achlioptas_sparse_f32() {
        test_error_decline::<f32>(
            (100, 10),
            Normal::new(42.0, 24.0).unwrap(),
            42,
            RandomProjection::Sparse {
                density: Some(1.0 / 3.0),
            },
        );
    }

    #[test]
    fn achlioptas_sparse_f64() {
        test_error_decline::<f64>(
            (100, 10),
            Normal::new(42.0, 24.0).unwrap(),
            42,
            RandomProjection::Sparse {
                density: Some(1.0 / 3.0),
            },
        );
    }

    #[test]
    fn ping_li_sparse_f32() {
        test_error_decline::<f32>(
            (10, 100),
            Normal::new(42.0, 24.0).unwrap(),
            42,
            RandomProjection::Sparse { density: None },
        );
    }

    #[test]
    fn ping_li_sparse_f64() {
        test_error_decline::<f64>(
            (10, 100),
            Normal::new(42.0, 24.0).unwrap(),
            42,
            RandomProjection::Sparse { density: None },
        );
    }

    fn test_error_decline<T: Float + std::fmt::Display>(
        shape: (usize, usize),
        distribution: impl Distribution<T>,
        seed: u64,
        projection: RandomProjection,
    ) {
        let data = Array::<T, Ix2>::random(shape, distribution);

        let mut max_rmse = rmse(&data, &roundtrip(&data, 42, 1.0, &projection));

        for epsilon in [0.5, 0.1, 0.01] {
            let new_rmse = rmse(&data, &roundtrip(&data, seed, epsilon, &projection));
            assert!(
                new_rmse <= max_rmse,
                "{new_rmse} > {max_rmse} for {epsilon}"
            );
            max_rmse = new_rmse;
        }
    }

    fn roundtrip<T: Float>(
        data: &Array<T, Ix2>,
        seed: u64,
        epsilon: f64,
        projection: &RandomProjection,
    ) -> Array<T, Ix2> {
        let projected = project_with_projection(data.view(), seed, epsilon, projection)
            .expect("projecting must not fail");
        let reconstructed = reconstruct_with_projection(projected, seed, projection)
            .expect("reconstruction must not fail");
        reconstructed
    }

    fn rmse<T: Float>(data: &Array<T, Ix2>, reconstructed: &Array<T, Ix2>) -> T {
        let mut err = T::ZERO;

        for (&d, &r) in data.iter().zip(reconstructed) {
            err += (d - r) * (d - r);
        }

        let mse = err * T::from_usize(data.len()).recip();
        mse.sqrt()
    }
}
