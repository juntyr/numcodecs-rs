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

use std::{borrow::Cow, num::NonZeroUsize, ops::AddAssign};

use ndarray::{s, Array, ArrayBase, ArrayViewMut, Data, Dimension, Ix2, ShapeError, Zip};
use num_traits::{ConstOne, ConstZero, Float, FloatConst};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig,
};
use schemars::{json_schema, JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

/// Codec that uses random projections to reduce the dimensionality of high-
/// dimensional data to compress it.
///
/// A two-dimensional array of shape `N x D` is encoded as n array of shape
/// `N x K`, where `K` is either set explicitly or chosen using the the Johnson-
/// Lindenstrauss lemma. For `K` to be smaller than `D`, `D` must be quite
/// large. Therefore, this codec should only applied on large datasets as it
/// otherwise significantly inflates the data size instead of reducing it.
///
/// Choosing a lower distortion rate `epsilon` will improve the quality of the
/// lossy compression, i.e. reduce the compression error, at the cost of
/// increasing `K`.
///
/// This codec only supports finite floating point data.
#[derive(Clone, Serialize, Deserialize, JsonSchema)]
// FIXME: #[serde(deny_unknown_fields)]
pub struct RandomProjectionCodec {
    /// Seed for generating the random projection matrix
    pub seed: u64,
    /// Method with which the reduced dimensionality `K` is selected
    #[serde(flatten)]
    pub reduction: RandomProjectionReduction,
    /// Projection kind that is used to generate the random projection matrix
    #[serde(flatten)]
    pub projection: RandomProjectionKind,
}

/// Method with which the reduced dimensionality `K` is selected
#[derive(Clone, Serialize, Deserialize, JsonSchema)]
// FIXME: #[serde(deny_unknown_fields)]
#[serde(tag = "reduction", rename_all = "kebab-case")]
pub enum RandomProjectionReduction {
    /// The reduced dimensionality `K` is derived from `epsilon`, as defined by
    /// the Johnson-Lindenstrauss lemma.
    JohnsonLindenstrauss {
        /// Maximum distortion rate
        epsilon: OpenClosedUnit<f64>,
    },
    /// The reduced dimensionality `K`, to which the data is projected, is given
    /// explicitly.
    Explicit {
        /// Reduced dimensionality
        k: NonZeroUsize,
    },
}

/// Projection kind that is used to generate the random projection matrix
#[derive(Clone, Serialize, Deserialize, JsonSchema)]
// FIXME: #[serde(deny_unknown_fields)]
#[serde(tag = "projection", rename_all = "kebab-case")]
pub enum RandomProjectionKind {
    /// The random projection matrix is dense and its components are sampled
    /// from `N(0, 1/k)`
    Gaussian,
    /// The random projection matrix is sparse where only `density`% of entries
    /// are non-zero.
    ///
    /// The matrix's components are sampled from
    ///
    /// - `-sqrt(1 / (k * density))` with probability `density/2`
    /// - `0` with probability `1-density`
    /// - `+sqrt(1 / (k * density))` with probability `density/2`
    Sparse {
        /// The `density` of the sparse projection matrix.
        ///
        /// Setting `density` to `Some(1.0/3.0)` reproduces the settings by
        /// Achlioptas [^1]. If `density` is `None`, it is set to `1/sqrt(d)`,
        /// the minimum density as recommended by Li et al [^2].
        ///
        ///
        /// [^1]: Achlioptas, D. (2003). Database-friendly random projections:
        ///       Johnson-Lindenstrauss with binary coins. *Journal of Computer
        ///       and System Sciences*, 66(4), 671-687. Available from:
        ///       [doi:10.1016/S0022-0000(03)00025-4](https://doi.org/10.1016/S0022-0000(03)00025-4).
        ///
        /// [^2]: Li, P., Hastie, T. J., and Church, K. W. (2006). Very sparse
        ///       random projections. In *Proceedings of the 12th ACM SIGKDD
        ///       international conference on Knowledge discovery and data
        ///       mining (KDD '06)*. Association for Computing Machinery, New
        ///       York, NY, USA, 287–296. Available from:
        ///       [doi:10.1145/1150402.1150436](https://doi.org/10.1145/1150402.1150436).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        density: Option<OpenClosedUnit<f64>>,
    },
}

impl Codec for RandomProjectionCodec {
    type Error = RandomProjectionCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            AnyCowArray::F32(data) => Ok(AnyArray::F32(
                project_with_projection(data, self.seed, &self.reduction, &self.projection)?
                    .into_dyn(),
            )),
            AnyCowArray::F64(data) => Ok(AnyArray::F64(
                project_with_projection(data, self.seed, &self.reduction, &self.projection)?
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
    /// [`RandomProjectionCodec`] does not support non-finite (infinite or NaN)
    /// floating point data
    #[error("RandomProjection does not support non-finite (infinite or NaN) floating point data")]
    NonFiniteData,
    /// [`RandomProjectionCodec`] cannot encode or decode from an array with `N`
    /// samples to an array with a different number of samples
    #[error("RandomProjection cannot encode or decode from an array with {input} samples to an array with {output} samples")]
    NumberOfSamplesMismatch {
        /// Number of samples `N` in the input array
        input: usize,
        /// Number of samples `N` in the output array
        output: usize,
    },
    /// [`RandomProjectionCodec`] cannot decode from an array with zero
    /// dimensionality `K`
    #[error("RandomProjection cannot decode from an array with zero dimensionality `K`")]
    ProjectedArrayZeroComponents,
    /// [`RandomProjectionCodec`] cannot decode from an array with corrupted
    /// dimensionality metadata
    #[error("RandomProjection cannot decode from an array with corrupted dimensionality metadata")]
    CorruptedNumberOfComponents,
    /// [`RandomProjectionCodec`] cannot decode into an array with `D` features
    /// that differs from the `D` stored in the encoded metadata
    #[error("RandomProjection cannot decode into an array with {output} features that differs from the {metadata} features stored in the encoded metadata")]
    NumberOfFeaturesMismatch {
        /// Number of features `D` in the encoded array metadata
        metadata: usize,
        /// Number of features `D` in the decoded output array
        output: usize,
    },
    /// [`RandomProjectionCodec`] cannot decode into the provided array
    #[error("RandomProjection cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

/// Applies random projection to the input `data` with the given `seed`,
/// `reduction` method, and `projection` kind and returns the resulting
/// projected array.
///
/// # Errors
///
/// Errors with
/// - [`RandomProjectionCodecError::NonMatrixData`] if the input `data` is not
///   a two-dimensional matrix
/// - [`RandomProjectionCodecError::NonFiniteData`] if the input `data` or
///   projected output contains non-finite data
pub fn project_with_projection<T: FloatExt, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
    seed: u64,
    reduction: &RandomProjectionReduction,
    projection: &RandomProjectionKind,
) -> Result<Array<T, Ix2>, RandomProjectionCodecError> {
    let data = data
        .into_dimensionality()
        .map_err(|err| RandomProjectionCodecError::NonMatrixData { source: err })?;

    let (n, d) = data.dim();

    let k = match reduction {
        RandomProjectionReduction::JohnsonLindenstrauss { epsilon } => {
            johnson_lindenstrauss_min_k(n, *epsilon)
        }
        RandomProjectionReduction::Explicit { k } => k.get(),
    };

    let mut projected = Array::<T, Ix2>::from_elem((n, k + 1), T::ZERO);

    for p in projected.slice_mut(s!(.., k)) {
        *p = T::from_usize(d);
    }

    match projection {
        RandomProjectionKind::Gaussian => project_into(
            data,
            projected.slice_mut(s!(.., ..k)),
            |x, y| gaussian_project(x, y, seed),
            gaussian_normaliser(k),
        ),
        RandomProjectionKind::Sparse { density } => {
            let density = density_or_ping_li_minimum(*density, d);
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

#[allow(clippy::needless_pass_by_value)]
/// Applies random projection to the input `data` and outputs into the
/// `projected` array.
///
/// The random projection matrix is defined by the `projection` function
/// `(i, j) -> P[i, j]` and a globally applied `normalizer` factor.
///
/// # Errors
///
/// Errors with
/// - [`RandomProjectionCodecError::NumberOfSamplesMismatch`] if the input
///   `data`'s number of samples doesn't match the `projected` array's number
///   of samples
/// - [`RandomProjectionCodecError::NonFiniteData`] if the input `data` or
///   projected output contains non-finite data
pub fn project_into<T: FloatExt, S: Data<Elem = T>>(
    data: ArrayBase<S, Ix2>,
    mut projected: ArrayViewMut<T, Ix2>,
    projection: impl Fn(usize, usize) -> T,
    normalizer: T,
) -> Result<(), RandomProjectionCodecError> {
    let (n, d) = data.dim();
    let (n2, _k) = projected.dim();

    if n2 != n {
        return Err(RandomProjectionCodecError::NumberOfSamplesMismatch {
            input: n,
            output: n2,
        });
    }

    let mut skip_projection_column = Vec::with_capacity(d);

    for (j, projected_j) in projected.columns_mut().into_iter().enumerate() {
        // materialize one column of the projection matrix
        //   i.e. instead of A x B = C, compute A x [bs] = [cs].T
        // the column is stored in a sparse representation [(l, p)]
        //   where l is the index of the non-zero entry p
        skip_projection_column.clear();
        for l in 0..d {
            let p = projection(l, j);
            if !p.is_zero() {
                skip_projection_column.push((l, p));
            }
        }

        for (data_i, projected_ij) in data.rows().into_iter().zip(projected_j) {
            let mut acc = T::ZERO;

            #[allow(clippy::indexing_slicing)]
            // data_i is of shape (d) and all l's are in 0..d
            for &(l, p) in &skip_projection_column {
                acc += data_i[l] * p;
            }

            *projected_ij = acc * normalizer;
        }
    }

    if !Zip::from(projected).all(|x| x.is_finite()) {
        return Err(RandomProjectionCodecError::NonFiniteData);
    }

    Ok(())
}

/// Applies the (approximate) inverse of random projection to the `projected`
/// array to reconstruct the input data with the given `seed` and `projection`
/// kind and returns the resulting reconstructed array.
///
/// # Errors
///
/// Errors with
/// - [`RandomProjectionCodecError::NonMatrixData`] if the `projected` array is
///   not a two-dimensional matrix
/// - [`RandomProjectionCodecError::ProjectedArrayZeroComponents`] if the
///   `projected` array is of shape `(n, 0)`
/// - [`RandomProjectionCodecError::CorruptedNumberOfComponents`] if the
///   `projected` array's dimensionality metadata is corrupted
/// - [`RandomProjectionCodecError::NonFiniteData`] if the `projected` array or
///   the reconstructed output contains non-finite data
pub fn reconstruct_with_projection<T: FloatExt, S: Data<Elem = T>, D: Dimension>(
    projected: ArrayBase<S, D>,
    seed: u64,
    projection: &RandomProjectionKind,
) -> Result<Array<T, Ix2>, RandomProjectionCodecError> {
    let projected = projected
        .into_dimensionality()
        .map_err(|err| RandomProjectionCodecError::NonMatrixData { source: err })?;

    if projected.is_empty() {
        return Ok(projected.to_owned());
    }

    let (_n, k): (usize, usize) = projected.dim();
    let Some(k) = k.checked_sub(1) else {
        return Err(RandomProjectionCodecError::ProjectedArrayZeroComponents);
    };

    let ds = projected.slice(s!(.., k));
    let Ok(Some(d)) = ds.fold(Ok(None), |acc, d| match acc {
        Ok(None) => Ok(Some(d.into_usize())),
        Ok(Some(d2)) if d2 == d.into_usize() => Ok(Some(d2)),
        _ => Err(()),
    }) else {
        return Err(RandomProjectionCodecError::CorruptedNumberOfComponents);
    };

    let projected = projected.slice_move(s!(.., ..k));

    match projection {
        RandomProjectionKind::Gaussian => reconstruct(
            projected,
            d,
            |x, y| gaussian_project(x, y, seed),
            gaussian_normaliser(k),
        ),
        RandomProjectionKind::Sparse { density } => {
            let density = density_or_ping_li_minimum(*density, d);
            reconstruct(
                projected,
                d,
                |x, y| sparse_project(x, y, density, seed),
                sparse_normaliser(k, density),
            )
        }
    }
}

#[allow(clippy::needless_pass_by_value)]
/// Applies the (approximate) inverse of random projection to the `projected`
/// array to reconstruct the input data with dimensionality `d` and returns the
/// resulting reconstructed array.
///
/// The random projection matrix is defined by the `projection` function
/// `(i, j) -> P[i, j]` and a globally applied `normalizer` factor.
///
/// # Errors
///
/// Errors with
/// - [`RandomProjectionCodecError::NonFiniteData`] if the `projected` array or
///   the reconstructed output contains non-finite data
pub fn reconstruct<T: FloatExt, S: Data<Elem = T>>(
    projected: ArrayBase<S, Ix2>,
    d: usize,
    projection: impl Fn(usize, usize) -> T,
    normalizer: T,
) -> Result<Array<T, Ix2>, RandomProjectionCodecError> {
    if projected.is_empty() {
        return Ok(projected.to_owned());
    }

    let (n, k) = projected.dim();

    let mut reconstructed = Array::<T, Ix2>::from_elem((n, d), T::ZERO);

    let mut skip_projection_row = Vec::with_capacity(k);

    for (l, reconstructed_l) in reconstructed.columns_mut().into_iter().enumerate() {
        // materialize one row of the projection matrix transpose
        // i.e. instead of A x B = C, compute A x [bs] = [cs].T
        // the row is stored in a sparse representation [(j, p)]
        //   where j is the index of the non-zero entry p
        skip_projection_row.clear();
        for j in 0..k {
            let p = projection(l, j);
            if !p.is_zero() {
                skip_projection_row.push((j, p));
            }
        }

        for (projected_i, reconstructed_il) in projected.rows().into_iter().zip(reconstructed_l) {
            let mut acc = T::ZERO;

            #[allow(clippy::indexing_slicing)]
            // projected_i is of shape (k) and all j's are in 0..k
            for &(j, p) in &skip_projection_row {
                acc += projected_i[j] * p;
            }

            *reconstructed_il = acc * normalizer;
        }
    }

    if !Zip::from(&reconstructed).all(|x| x.is_finite()) {
        return Err(RandomProjectionCodecError::NonFiniteData);
    }

    Ok(reconstructed)
}

/// Applies the (approximate) inverse of random projection to the `projected`
/// array to reconstruct the input data with the given `seed` and `projection`
/// kind and outputs into the `reconstructed` array.
///
/// # Errors
///
/// Errors with
/// - [`RandomProjectionCodecError::NonMatrixData`] if the `projected` and
///   `reconstructed` arrays are not two-dimensional matrices
/// - [`RandomProjectionCodecError::NumberOfSamplesMismatch`] if the number of
///   samples `N` of the `projected` array don't match the number of samples of
///   the `reconstructed` array
/// - [`RandomProjectionCodecError::ProjectedArrayZeroComponents`] if the
///   `projected` array is of shape `(n, 0)`
/// - [`RandomProjectionCodecError::CorruptedNumberOfComponents`] if the
///   `projected` array's dimensionality metadata is corrupted
/// - [`RandomProjectionCodecError::NumberOfFeaturesMismatch`] if the
///   `reconstructed` array's dimensionality `D` does not match the `projected`
///   array's dimensionality metadata
/// - [`RandomProjectionCodecError::NonFiniteData`] if the `projected` array or
///   the reconstructed output contains non-finite data
pub fn reconstruct_into_with_projection<T: FloatExt, S: Data<Elem = T>, D: Dimension>(
    projected: ArrayBase<S, D>,
    reconstructed: ArrayViewMut<T, D>,
    seed: u64,
    projection: &RandomProjectionKind,
) -> Result<(), RandomProjectionCodecError> {
    let projected: ArrayBase<S, Ix2> = projected
        .into_dimensionality()
        .map_err(|err| RandomProjectionCodecError::NonMatrixData { source: err })?;
    let reconstructed: ArrayViewMut<T, Ix2> = reconstructed
        .into_dimensionality()
        .map_err(|err| RandomProjectionCodecError::NonMatrixData { source: err })?;

    let (n, k) = projected.dim();
    let (n2, d2) = reconstructed.dim();

    if n2 != n {
        return Err(RandomProjectionCodecError::NumberOfSamplesMismatch {
            input: n,
            output: n2,
        });
    }

    let Some(k) = k.checked_sub(1) else {
        return Err(RandomProjectionCodecError::ProjectedArrayZeroComponents);
    };

    let ds = projected.slice(s!(.., k));
    let Ok(Some(d)) = ds.fold(Ok(None), |acc, d| match acc {
        Ok(None) => Ok(Some(d.into_usize())),
        Ok(Some(d2)) if d2 == d.into_usize() => Ok(Some(d2)),
        _ => Err(()),
    }) else {
        return Err(RandomProjectionCodecError::CorruptedNumberOfComponents);
    };

    if d2 != d {
        return Err(RandomProjectionCodecError::NumberOfFeaturesMismatch {
            metadata: d,
            output: d2,
        });
    }

    let projected = projected.slice_move(s!(.., ..k));

    match projection {
        RandomProjectionKind::Gaussian => reconstruct_into(
            projected,
            reconstructed,
            |x, y| gaussian_project(x, y, seed),
            gaussian_normaliser(k),
        ),
        RandomProjectionKind::Sparse { density } => {
            let density = density_or_ping_li_minimum(*density, d);
            reconstruct_into(
                projected,
                reconstructed,
                |x, y| sparse_project(x, y, density, seed),
                sparse_normaliser(k, density),
            )
        }
    }
}

#[allow(clippy::needless_pass_by_value)]
/// Applies the (approximate) inverse of random projection to the `projected`
/// array to reconstruct the input data outputs into the `reconstructed` array.
///
/// The random projection matrix is defined by the `projection` function
/// `(i, j) -> P[i, j]` and a globally applied `normalizer` factor.
///
/// # Errors
///
/// Errors with
/// - [`RandomProjectionCodecError::NumberOfSamplesMismatch`] if the number of
///   samples `N` of the `projected` array don't match the number of samples of
///   the `reconstructed` array
/// - [`RandomProjectionCodecError::NonFiniteData`] if the `projected` array or
///   the reconstructed output contains non-finite data
pub fn reconstruct_into<T: FloatExt, S: Data<Elem = T>>(
    projected: ArrayBase<S, Ix2>,
    mut reconstructed: ArrayViewMut<T, Ix2>,
    projection: impl Fn(usize, usize) -> T,
    normalizer: T,
) -> Result<(), RandomProjectionCodecError> {
    let (n, k) = projected.dim();
    let (n2, _d) = reconstructed.dim();

    if n2 != n {
        return Err(RandomProjectionCodecError::NumberOfSamplesMismatch {
            input: n,
            output: n2,
        });
    }

    let mut skip_projection_row = Vec::with_capacity(k);

    for (l, reconstructed_l) in reconstructed.columns_mut().into_iter().enumerate() {
        // materialize one row of the projection matrix transpose
        // i.e. instead of A x B = C, compute A x [bs] = [cs].T
        // the row is stored in a sparse representation [(j, p)]
        //   where j is the index of the non-zero entry p
        skip_projection_row.clear();
        for j in 0..k {
            let p = projection(l, j);
            if !p.is_zero() {
                skip_projection_row.push((j, p));
            }
        }

        for (projected_i, reconstructed_il) in projected.rows().into_iter().zip(reconstructed_l) {
            let mut acc = T::ZERO;

            #[allow(clippy::indexing_slicing)]
            // projected_i is of shape (k) and all j's are in 0..k
            for &(j, p) in &skip_projection_row {
                acc += projected_i[j] * p;
            }

            *reconstructed_il = acc * normalizer;
        }
    }

    if !Zip::from(reconstructed).all(|x| x.is_finite()) {
        return Err(RandomProjectionCodecError::NonFiniteData);
    }

    Ok(())
}

/// Find a 'safe' number of components `K` to randomly project to.
///
/// The minimum number of components to guarantee the eps-embedding is
/// given by `K >= 4 log(N) / (eps^2 / 2 - eps^3 / 3)`.
///
/// The implementation is adapted from [`sklearn`]'s.
///
/// [`sklearn`]: https://github.com/scikit-learn/scikit-learn/blob/3b39d7cb957ab781744b346c1848be9db3f4e221/sklearn/random_projection.py#L56-L142
#[must_use]
pub fn johnson_lindenstrauss_min_k(
    n_samples: usize,
    OpenClosedUnit(epsilon): OpenClosedUnit<f64>,
) -> usize {
    #[allow(clippy::suboptimal_flops)]
    let denominator = (epsilon * epsilon * 0.5) - (epsilon * epsilon * epsilon / 3.0);
    #[allow(clippy::cast_precision_loss)]
    let min_k = (n_samples as f64).ln() * 4.0 / denominator;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let min_k = min_k as usize;
    min_k
}

/// Extract the provided `density` if it is `Some(_)`, or compute the minimum
/// required density `1/sqrt(d)` as recommended by Li et al [^3].
///
/// [^3]: Li, P., Hastie, T. J., and Church, K. W. (2006). Very sparse
///       random projections. In *Proceedings of the 12th ACM SIGKDD
///       international conference on Knowledge discovery and data
///       mining (KDD '06)*. Association for Computing Machinery, New
///       York, NY, USA, 287–296. Available from:
///       [doi:10.1145/1150402.1150436](https://doi.org/10.1145/1150402.1150436).
#[must_use]
pub fn density_or_ping_li_minimum<T: FloatExt>(
    density: Option<OpenClosedUnit<f64>>,
    d: usize,
) -> T {
    match density {
        Some(OpenClosedUnit(density)) => T::from_f64(density),
        None => T::from_usize(d).sqrt().recip(),
    }
}

fn gaussian_project<T: FloatExt>(x: usize, y: usize, seed: u64) -> T {
    let (ClosedOpenUnit(u0), OpenClosedUnit(u1)) = T::u01x2(hash_matrix_index(x, y, seed));

    let theta = -T::TAU() * u0;
    let r = (-T::TWO * u1.ln()).sqrt();

    r * theta.sin()
}

fn gaussian_normaliser<T: FloatExt>(k: usize) -> T {
    T::from_usize(k).sqrt().recip()
}

fn sparse_project<T: FloatExt>(x: usize, y: usize, density: T, seed: u64) -> T {
    let (ClosedOpenUnit(u0), _u1) = T::u01x2(hash_matrix_index(x, y, seed));

    if u0 < (density * T::HALF) {
        -T::ONE
    } else if u0 < density {
        T::ONE
    } else {
        T::ZERO
    }
}

fn sparse_normaliser<T: FloatExt>(k: usize, density: T) -> T {
    (T::from_usize(k) * density).recip().sqrt()
}

const fn hash_matrix_index(x: usize, y: usize, seed: u64) -> u64 {
    seahash_diffuse(seahash_diffuse(x as u64) ^ seed ^ (y as u64))
}

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

#[allow(clippy::derive_partial_eq_without_eq)] // floats are not Eq
#[derive(Copy, Clone, PartialEq, PartialOrd, Hash)]
/// Floating point number in [0.0, 1.0)
pub struct ClosedOpenUnit<T: FloatExt>(T);

#[allow(clippy::derive_partial_eq_without_eq)] // floats are not Eq
#[derive(Copy, Clone, PartialEq, PartialOrd, Hash)]
/// Floating point number in (0.0, 1.0]
pub struct OpenClosedUnit<T: FloatExt>(T);

impl Serialize for OpenClosedUnit<f64> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(self.0)
    }
}

impl<'de> Deserialize<'de> for OpenClosedUnit<f64> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let x = f64::deserialize(deserializer)?;

        if x > 0.0 && x <= 1.0 {
            Ok(Self(x))
        } else {
            Err(serde::de::Error::invalid_value(
                serde::de::Unexpected::Float(x),
                &"a value in (0.0, 1.0]",
            ))
        }
    }
}

impl JsonSchema for OpenClosedUnit<f64> {
    fn schema_name() -> Cow<'static, str> {
        Cow::Borrowed("OpenClosedUnitF64")
    }

    fn schema_id() -> Cow<'static, str> {
        Cow::Borrowed(concat!(module_path!(), "::", "OpenClosedUnit<f64>"))
    }

    fn json_schema(_gen: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "type": "number",
            "exclusiveMinimum": 0.0,
            "maximum": 1.0,
        })
    }
}

/// Floating point types.
pub trait FloatExt: Float + ConstZero + ConstOne + FloatConst + AddAssign {
    /// `0.5`
    const HALF: Self;
    /// `2.0`
    const TWO: Self;

    /// Converts from a [`f64`].
    #[must_use]
    fn from_f64(x: f64) -> Self;

    /// Converts from a [`usize`].
    #[must_use]
    fn from_usize(n: usize) -> Self;

    /// Converts into a [`usize`].
    #[must_use]
    fn into_usize(self) -> usize;

    /// Generates two uniform random numbers from a random `hash` value.
    ///
    /// The first is sampled from `[0.0, 1.0)`, the second from `(0.0, 1.0]`.
    #[must_use]
    fn u01x2(hash: u64) -> (ClosedOpenUnit<Self>, OpenClosedUnit<Self>);
}

impl FloatExt for f32 {
    const HALF: Self = 0.5;
    const TWO: Self = 2.0;

    #[allow(clippy::cast_possible_truncation)]
    fn from_f64(x: f64) -> Self {
        x as Self
    }

    #[allow(clippy::cast_precision_loss)]
    fn from_usize(n: usize) -> Self {
        n as Self
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn into_usize(self) -> usize {
        self as usize
    }

    fn u01x2(hash: u64) -> (ClosedOpenUnit<Self>, OpenClosedUnit<Self>) {
        #[allow(clippy::cast_possible_truncation)] // split u64 into (u32, u32)
        let (low, high) = (
            (hash & u64::from(u32::MAX)) as u32,
            ((hash >> 32) & u64::from(u32::MAX)) as u32,
        );

        // http://prng.di.unimi.it -> Generating uniform doubles in the unit interval [0.0, 1.0)
        // the hash is shifted to only cover the mantissa
        #[allow(clippy::cast_precision_loss)]
        let u0 = ClosedOpenUnit(((high >> 8) as Self) * Self::from_bits(0x3380_0000_u32)); // 0x1.0p-24

        // http://prng.di.unimi.it -> Generating uniform doubles in the unit interval (0.0, 1.0]
        // the hash is shifted to only cover the mantissa
        #[allow(clippy::cast_precision_loss)]
        let u1 = OpenClosedUnit((((low >> 8) + 1) as Self) * Self::from_bits(0x3380_0000_u32)); // 0x1.0p-24

        (u0, u1)
    }
}

impl FloatExt for f64 {
    const HALF: Self = 0.5;
    const TWO: Self = 2.0;

    fn from_f64(x: f64) -> Self {
        x
    }

    #[allow(clippy::cast_precision_loss)]
    fn from_usize(n: usize) -> Self {
        n as Self
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn into_usize(self) -> usize {
        self as usize
    }

    fn u01x2(hash: u64) -> (ClosedOpenUnit<Self>, OpenClosedUnit<Self>) {
        // http://prng.di.unimi.it -> Generating uniform doubles in the unit interval [0.0, 1.0)
        // the hash is shifted to only cover the mantissa
        #[allow(clippy::cast_precision_loss)]
        let u0 =
            ClosedOpenUnit(((hash >> 11) as Self) * Self::from_bits(0x3CA0_0000_0000_0000_u64)); // 0x1.0p-53

        let hash = seahash_diffuse(hash + 1);

        // http://prng.di.unimi.it -> Generating uniform doubles in the unit interval (0.0, 1.0]
        // the hash is shifted to only cover the mantissa
        #[allow(clippy::cast_precision_loss)]
        let u1 = OpenClosedUnit(
            (((hash >> 11) + 1) as Self) * Self::from_bits(0x3CA0_0000_0000_0000_u64),
        ); // 0x1.0p-53

        (u0, u1)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use ndarray_rand::rand_distr::{Distribution, Normal};
    use ndarray_rand::RandomExt;

    use super::*;

    #[test]
    fn gaussian_f32() {
        test_error_decline::<f32>(
            (100, 100),
            Normal::new(42.0, 24.0).unwrap(),
            42,
            RandomProjectionKind::Gaussian,
        );
    }

    #[test]
    fn gaussian_f64() {
        test_error_decline::<f64>(
            (100, 100),
            Normal::new(42.0, 24.0).unwrap(),
            42,
            RandomProjectionKind::Gaussian,
        );
    }

    #[test]
    fn achlioptas_sparse_f32() {
        test_error_decline::<f32>(
            (100, 100),
            Normal::new(42.0, 24.0).unwrap(),
            42,
            RandomProjectionKind::Sparse {
                density: Some(OpenClosedUnit(1.0 / 3.0)),
            },
        );
    }

    #[test]
    fn achlioptas_sparse_f64() {
        test_error_decline::<f64>(
            (100, 100),
            Normal::new(42.0, 24.0).unwrap(),
            42,
            RandomProjectionKind::Sparse {
                density: Some(OpenClosedUnit(1.0 / 3.0)),
            },
        );
    }

    #[test]
    fn ping_li_sparse_f32() {
        test_error_decline::<f32>(
            (100, 100),
            Normal::new(42.0, 24.0).unwrap(),
            42,
            RandomProjectionKind::Sparse { density: None },
        );
    }

    #[test]
    fn ping_li_sparse_f64() {
        test_error_decline::<f64>(
            (100, 100),
            Normal::new(42.0, 24.0).unwrap(),
            42,
            RandomProjectionKind::Sparse { density: None },
        );
    }

    #[allow(clippy::needless_pass_by_value)]
    fn test_error_decline<T: FloatExt + std::fmt::Display>(
        shape: (usize, usize),
        distribution: impl Distribution<T>,
        seed: u64,
        projection: RandomProjectionKind,
    ) {
        let data = Array::<T, Ix2>::random(shape, distribution);

        let mut max_rmse = rmse(
            &data,
            &roundtrip(&data, 42, OpenClosedUnit(1.0), &projection),
        );

        for epsilon in [
            OpenClosedUnit(0.5),
            OpenClosedUnit(0.1),
            OpenClosedUnit(0.01),
        ] {
            let new_rmse = rmse(&data, &roundtrip(&data, seed, epsilon, &projection));
            assert!(
                new_rmse <= max_rmse,
                "{new_rmse} > {max_rmse} for {epsilon}",
                epsilon = epsilon.0
            );
            max_rmse = new_rmse;
        }
    }

    fn roundtrip<T: FloatExt>(
        data: &Array<T, Ix2>,
        seed: u64,
        epsilon: OpenClosedUnit<f64>,
        projection: &RandomProjectionKind,
    ) -> Array<T, Ix2> {
        let projected = project_with_projection(
            data.view(),
            seed,
            &RandomProjectionReduction::JohnsonLindenstrauss { epsilon },
            projection,
        )
        .expect("projecting must not fail");
        let reconstructed = reconstruct_with_projection(projected, seed, projection)
            .expect("reconstruction must not fail");
        #[allow(clippy::let_and_return)]
        reconstructed
    }

    fn rmse<T: FloatExt>(data: &Array<T, Ix2>, reconstructed: &Array<T, Ix2>) -> T {
        let mut err = T::ZERO;

        for (&d, &r) in data.iter().zip(reconstructed) {
            err += (d - r) * (d - r);
        }

        let mse = err * T::from_usize(data.len()).recip();
        mse.sqrt()
    }
}
