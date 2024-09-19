//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.76.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-pca
//! [crates.io]: https://crates.io/crates/numcodecs-pca
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-pca
//! [docs.rs]: https://docs.rs/numcodecs-pca/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_pca
//!
//! PCA codec implementation for the [`numcodecs`] API.

use std::{
    borrow::Cow,
    num::NonZeroUsize,
    ops::{MulAssign, SubAssign},
};

use nalgebra::{
    linalg::{QR, SVD},
    ComplexField, Scalar,
};
use ndarray::{Array, ArrayBase, Axis, Data, Dimension, Ix1, Ix2, ShapeError};
use ndarray_rand::{
    rand::SeedableRng,
    rand_distr::{Distribution, StandardNormal},
    RandomExt,
};
use nshare::{ToNalgebra, ToNdarray2};
use num_traits::{Float, FromPrimitive, Zero};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig,
};
use schemars::{json_schema, JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;
use wyhash::WyRng;

/// Codec that uses principal component analysis to reduce the dimensionality
/// of high-dimensional data to compress it.
///
/// A two-dimensional array of shape `N x D` is encoded as n array of shape
/// `N x K`, where `K` is the number of principal components to keep.
///
/// For high-dimensionality arrays, randomized PCA should be used by providing
/// a `seed` for a random number generator.
///
/// This codec only supports finite floating point data.
#[derive(Clone, Serialize, Deserialize, JsonSchema)]
// FIXME: #[serde(deny_unknown_fields)]
pub struct PCACodec {
    /// The number of principal components to keep for the dimensionality of the
    /// encoded data
    pub k: NonZeroUsize,
    /// Optional seed for using randomized PCA instead of full PCA
    #[serde(default)]
    pub seed: Option<u64>,
    /// Tolerance for the SVD solver when checking for convergence, 0.0 by default
    #[serde(default = "NonNegative::zero")]
    pub svd_tolerance: NonNegative<f64>,
    /// Optional maximum number of iterations before the SVD solver fails.
    ///
    /// By default, the solver tries indefinitely.
    #[serde(default)]
    pub svd_max_iterations: Option<NonZeroUsize>,
}

impl Codec for PCACodec {
    type Error = PCACodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            AnyCowArray::F32(data) => Ok(AnyArray::F32(
                project_with_projection(
                    data,
                    self.k,
                    self.seed,
                    self.svd_tolerance,
                    self.svd_max_iterations,
                )?
                .into_dyn(),
            )),
            AnyCowArray::F64(data) => Ok(AnyArray::F64(
                project_with_projection(
                    data,
                    self.k,
                    self.seed,
                    self.svd_tolerance,
                    self.svd_max_iterations,
                )?
                .into_dyn(),
            )),
            encoded => Err(PCACodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match encoded {
            AnyCowArray::F32(encoded) => Ok(AnyArray::F32(unimplemented!())),
            AnyCowArray::F64(encoded) => Ok(AnyArray::F64(unimplemented!())),
            encoded => Err(PCACodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        match (encoded, decoded) {
            (AnyArrayView::F32(encoded), AnyArrayViewMut::F32(decoded)) => {
                unimplemented!()
            }
            (AnyArrayView::F64(encoded), AnyArrayViewMut::F64(decoded)) => {
                unimplemented!()
            }
            (encoded @ (AnyArrayView::F32(_) | AnyArrayView::F64(_)), decoded) => {
                Err(PCACodecError::MismatchedDecodeIntoArray {
                    source: AnyArrayAssignError::DTypeMismatch {
                        src: encoded.dtype(),
                        dst: decoded.dtype(),
                    },
                })
            }
            (encoded, _decoded) => Err(PCACodecError::UnsupportedDtype(encoded.dtype())),
        }
    }
}

impl StaticCodec for PCACodec {
    const CODEC_ID: &'static str = "pca";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<Self> {
        StaticCodecConfig::from(self)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`PCACodec`].
pub enum PCACodecError {
    /// [`PCACodec`] does not support the dtype
    #[error("PCA does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`PCACodec`] does not support the dtype
    #[error("PCA only supports matrix / 2d-shaped arrays")]
    NonMatrixData {
        /// The source of the error
        #[from]
        source: ShapeError,
    },
    /// [`PCACodec`] does not support non-finite (infinite or NaN)
    /// floating point data
    #[error("PCA does not support non-finite (infinite or NaN) floating point data")]
    NonFiniteData,
    /// [`PCACodec`] cannot encode or decode from an array with `N`
    /// samples to an array with a different number of samples
    #[error("PCA cannot encode or decode from an array with {input} samples to an array with {output} samples")]
    NumberOfSamplesMismatch {
        /// Number of samples `N` in the input array
        input: usize,
        /// Number of samples `N` in the output array
        output: usize,
    },
    /// [`PCACodec`] cannot decode from an array with zero
    /// dimensionality `K`
    #[error("PCA cannot decode from an array with zero dimensionality `K`")]
    ProjectedArrayZeroComponents,
    /// [`PCACodec`] cannot decode from an array with corrupted
    /// dimensionality metadata
    #[error("PCA cannot decode from an array with corrupted dimensionality metadata")]
    CorruptedNumberOfComponents,
    /// [`PCACodec`] cannot decode into an array with `D` features
    /// that differs from the `D` stored in the encoded metadata
    #[error("PCA cannot decode into an array with {output} features that differs from the {metadata} features stored in the encoded metadata")]
    NumberOfFeaturesMismatch {
        /// Number of features `D` in the encoded array metadata
        metadata: usize,
        /// Number of features `D` in the decoded output array
        output: usize,
    },
    /// [`PCACodec`] cannot decode into the provided array
    #[error("PCA cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

pub fn project_with_projection<T: FloatExt, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
    k: NonZeroUsize,
    seed: Option<u64>,
    svd_tolerance: NonNegative<f64>,
    svd_max_iterations: Option<NonZeroUsize>,
) -> Result<Array<T, Ix2>, PCACodecError>
where
    StandardNormal: Distribution<T>,
{
    let data: ArrayBase<S, Ix2> = data
        .into_dimensionality()
        .map_err(|err| PCACodecError::NonMatrixData { source: err })?;

    let (n, d) = data.dim();

    let Some(mean): Option<Array<T, Ix1>> = data.mean_axis(Axis(0)) else {
        return Ok(Array::zeros((n, k.get())));
    };

    let mut centred_data = data.into_owned();
    centred_data.sub_assign(&mean);

    let centred_data_matrix = centred_data.view_mut().into_nalgebra();

    // adapted from https://github.com/ekg/rsvd/blob/a44fd1584144f8f60c2a0b872edcb47b8b64d769/src/lib.rs#L39-L78
    // published under the MIT License by Erik Garrison
    let (projection, projected) = if let Some(seed) = seed {
        let mut rng = WyRng::seed_from_u64(seed);

        // number of oversamples
        const P: usize = 10;
        let l = k.get() + P;

        // generate the Gaussian random test matrix
        let omega = Array::random_using((d, l), StandardNormal, &mut rng);
        let omega = omega.view().into_nalgebra();

        // form the sample matrix Y
        let y = (&centred_data_matrix) * omega;

        // orthogonalize Y
        let (q, _r) = QR::new(y).unpack();

        // project the input data to the lower dimension
        let b = q.transpose() * (&centred_data_matrix);

        // compute the SVD of the small matrix B
        let Some(SVD { v_t: Some(v_t), .. }) = SVD::try_new(
            b,
            false,
            true,
            <T as FloatExt>::from_f64(svd_tolerance.0),
            svd_max_iterations.map_or(0, NonZeroUsize::get),
        ) else {
            panic!("SVD failed");
        };

        // truncated PCs
        let projection = v_t;
        let projected = centred_data_matrix * (&projection);

        (projection.into_ndarray2(), projected.into_ndarray2())
    } else {
        let Some(SVD { v_t: Some(v_t), .. }) = SVD::try_new(
            centred_data_matrix.clone_owned(),
            false,
            true,
            <T as FloatExt>::from_f64(svd_tolerance.0),
            svd_max_iterations.map_or(0, NonZeroUsize::get),
        ) else {
            panic!("full SVD failed");
        };

        // full SVD, needs to be truncated
        let projection = v_t.slice((0, 0), (k.get(), d)).into_owned();
        let projected = centred_data_matrix * (&projection);

        (projection.into_ndarray2(), projected.into_ndarray2())
    };

    todo!();
}

/// Floating point types.
pub trait FloatExt:
    Float + FromPrimitive + SubAssign + Scalar + MulAssign + ComplexField<RealField = Self>
{
    /// Converts from a [`f64`].
    #[must_use]
    fn from_f64(x: f64) -> Self;
}

impl FloatExt for f32 {
    #[allow(clippy::cast_possible_truncation)]
    fn from_f64(x: f64) -> Self {
        x as Self
    }
}

impl FloatExt for f64 {
    fn from_f64(x: f64) -> Self {
        x
    }
}

#[allow(clippy::derive_partial_eq_without_eq)] // floats are not Eq
#[derive(Copy, Clone, PartialEq, PartialOrd, Hash)]
/// Non-negative floating point number
pub struct NonNegative<T: Float>(T);

impl<T: Zero + Float> NonNegative<T> {
    #[must_use]
    pub fn zero() -> Self {
        Self(T::zero())
    }
}

impl Serialize for NonNegative<f64> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(self.0)
    }
}

impl<'de> Deserialize<'de> for NonNegative<f64> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let x = f64::deserialize(deserializer)?;

        if x >= 0.0 {
            Ok(Self(x))
        } else {
            Err(serde::de::Error::invalid_value(
                serde::de::Unexpected::Float(x),
                &"a non-negative value",
            ))
        }
    }
}

impl JsonSchema for NonNegative<f64> {
    fn schema_name() -> Cow<'static, str> {
        Cow::Borrowed("NonNegativeF64")
    }

    fn schema_id() -> Cow<'static, str> {
        Cow::Borrowed(concat!(module_path!(), "::", "NonNegative<f64>"))
    }

    fn json_schema(_gen: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "type": "number",
            "minimum": 0.0
        })
    }
}
