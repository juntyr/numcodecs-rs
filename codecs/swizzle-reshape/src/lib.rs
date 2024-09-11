//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.76.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-swizzle-reshape
//! [crates.io]: https://crates.io/crates/numcodecs-swizzle-reshape
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-swizzle-reshape
//! [docs.rs]: https://docs.rs/numcodecs-swizzle-reshape/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_swizzle_reshape
//!
//! Array axis swizzle and reshape codec implementation for the [`numcodecs`]
//! API.

use std::{
    borrow::Cow,
    collections::VecDeque,
    fmt::{self, Debug},
    iter::{Chain, Once},
    slice::Iter,
};

use ndarray::{Array, ArrayBase, ArrayView, ArrayViewMut, Data, IxDyn};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig,
};
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
/// Codec to swizzle/swap the axes of an array and reshape it.
///
/// This codec does not store metadata about the original shape of the array.
/// Since axes that have been combined during encoding cannot be split without
/// further information, decoding may fail if an output array is not provided.
///
/// Swizzling axes is always supported since no additional information about the
/// array's shape is required to reconstruct it.
pub struct SwizzleReshapeCodec {
    /// The permutation of the axes that is applied on encoding
    pub axes: Vec<Axis>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
#[serde(deny_unknown_fields)]
/// An axis, potentially from a merged combination of multiple input axes
pub enum Axis {
    /// A single axis
    Single(usize),
    /// A merged combination of multiple input axes
    Merged(Multiple<usize>),
}

impl Codec for SwizzleReshapeCodec {
    type Error = SwizzleReshapeCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            AnyCowArray::U8(data) => Ok(AnyArray::U8(swizzle_reshape(data, &self.axes)?)),
            AnyCowArray::U16(data) => Ok(AnyArray::U16(swizzle_reshape(data, &self.axes)?)),
            AnyCowArray::U32(data) => Ok(AnyArray::U32(swizzle_reshape(data, &self.axes)?)),
            AnyCowArray::U64(data) => Ok(AnyArray::U64(swizzle_reshape(data, &self.axes)?)),
            AnyCowArray::I8(data) => Ok(AnyArray::I8(swizzle_reshape(data, &self.axes)?)),
            AnyCowArray::I16(data) => Ok(AnyArray::I16(swizzle_reshape(data, &self.axes)?)),
            AnyCowArray::I32(data) => Ok(AnyArray::I32(swizzle_reshape(data, &self.axes)?)),
            AnyCowArray::I64(data) => Ok(AnyArray::I64(swizzle_reshape(data, &self.axes)?)),
            AnyCowArray::F32(data) => Ok(AnyArray::F32(swizzle_reshape(data, &self.axes)?)),
            AnyCowArray::F64(data) => Ok(AnyArray::F64(swizzle_reshape(data, &self.axes)?)),
            data => Err(SwizzleReshapeCodecError::UnsupportedDtype(data.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match encoded {
            AnyCowArray::U8(encoded) => {
                Ok(AnyArray::U8(undo_swizzle_reshape(encoded, &self.axes)?))
            }
            AnyCowArray::U16(encoded) => {
                Ok(AnyArray::U16(undo_swizzle_reshape(encoded, &self.axes)?))
            }
            AnyCowArray::U32(encoded) => {
                Ok(AnyArray::U32(undo_swizzle_reshape(encoded, &self.axes)?))
            }
            AnyCowArray::U64(encoded) => {
                Ok(AnyArray::U64(undo_swizzle_reshape(encoded, &self.axes)?))
            }
            AnyCowArray::I8(encoded) => {
                Ok(AnyArray::I8(undo_swizzle_reshape(encoded, &self.axes)?))
            }
            AnyCowArray::I16(encoded) => {
                Ok(AnyArray::I16(undo_swizzle_reshape(encoded, &self.axes)?))
            }
            AnyCowArray::I32(encoded) => {
                Ok(AnyArray::I32(undo_swizzle_reshape(encoded, &self.axes)?))
            }
            AnyCowArray::I64(encoded) => {
                Ok(AnyArray::I64(undo_swizzle_reshape(encoded, &self.axes)?))
            }
            AnyCowArray::F32(encoded) => {
                Ok(AnyArray::F32(undo_swizzle_reshape(encoded, &self.axes)?))
            }
            AnyCowArray::F64(encoded) => {
                Ok(AnyArray::F64(undo_swizzle_reshape(encoded, &self.axes)?))
            }
            encoded => Err(SwizzleReshapeCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        match (encoded, decoded) {
            (AnyArrayView::U8(encoded), AnyArrayViewMut::U8(decoded)) => {
                undo_swizzle_reshape_into(encoded, decoded, &self.axes)
            }
            (AnyArrayView::U16(encoded), AnyArrayViewMut::U16(decoded)) => {
                undo_swizzle_reshape_into(encoded, decoded, &self.axes)
            }
            (AnyArrayView::U32(encoded), AnyArrayViewMut::U32(decoded)) => {
                undo_swizzle_reshape_into(encoded, decoded, &self.axes)
            }
            (AnyArrayView::U64(encoded), AnyArrayViewMut::U64(decoded)) => {
                undo_swizzle_reshape_into(encoded, decoded, &self.axes)
            }
            (AnyArrayView::I8(encoded), AnyArrayViewMut::I8(decoded)) => {
                undo_swizzle_reshape_into(encoded, decoded, &self.axes)
            }
            (AnyArrayView::I16(encoded), AnyArrayViewMut::I16(decoded)) => {
                undo_swizzle_reshape_into(encoded, decoded, &self.axes)
            }
            (AnyArrayView::I32(encoded), AnyArrayViewMut::I32(decoded)) => {
                undo_swizzle_reshape_into(encoded, decoded, &self.axes)
            }
            (AnyArrayView::I64(encoded), AnyArrayViewMut::I64(decoded)) => {
                undo_swizzle_reshape_into(encoded, decoded, &self.axes)
            }
            (AnyArrayView::F32(encoded), AnyArrayViewMut::F32(decoded)) => {
                undo_swizzle_reshape_into(encoded, decoded, &self.axes)
            }
            (AnyArrayView::F64(encoded), AnyArrayViewMut::F64(decoded)) => {
                undo_swizzle_reshape_into(encoded, decoded, &self.axes)
            }
            (encoded, decoded) if encoded.dtype() != decoded.dtype() => {
                Err(SwizzleReshapeCodecError::MismatchedDecodeIntoArray {
                    source: AnyArrayAssignError::DTypeMismatch {
                        src: encoded.dtype(),
                        dst: decoded.dtype(),
                    },
                })
            }
            (encoded, _decoded) => Err(SwizzleReshapeCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }
}

impl StaticCodec for SwizzleReshapeCodec {
    const CODEC_ID: &'static str = "swizzle-reshape";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<Self> {
        StaticCodecConfig::from(self)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`SwizzleReshapeCodec`].
pub enum SwizzleReshapeCodecError {
    /// [`SwizzleReshapeCodec`] does not support the dtype
    #[error("SwizzleReshape does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`SwizzleReshapeCodec`] cannot decode from an array with merged axes
    /// without receiving an output array to decode into
    #[error("SwizzleReshape cannot decode from an array with merged axes without receiving an output array to decode into")]
    CannotDecodeMergedAxes,
    /// [`SwizzleReshapeCodec`] cannot encode or decode with an invalid axis
    /// `index` for an array with `ndim` dimensions
    #[error("SwizzleReshape cannot encode or decode with an invalid axis {index} for an array with {ndim} dimensions")]
    InvalidAxisIndex {
        /// The out-of-bounds axis index
        index: usize,
        /// The number of dimensions of the array
        ndim: usize,
    },
    /// [`SwizzleReshapeCodec`] can only encode or decode with an axis
    /// permutation `axes` that contains every axis of an array with `ndim`
    /// dimensions index exactly once
    #[error("SwizzleReshape can only encode or decode with an axis permutation {axes:?} that contains every axis of an array with {ndim} dimensions index exactly once")]
    InvalidAxisPermutation {
        /// The invalid permutation of axes
        axes: Vec<Axis>,
        /// The number of dimensions of the array
        ndim: usize,
    },
    /// [`SwizzleReshapeCodec`] cannot decode into the provided array
    #[error("SwizzleReshape cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

#[allow(clippy::missing_panics_doc)]
/// Swizzle and reshape the input `data` array with the new `axes`.
///
/// # Errors
///
/// Errors with
/// - [`SwizzleReshapeCodecError::InvalidAxisIndex`] if any axis is out of
///   bounds
/// - [`SwizzleReshapeCodecError::InvalidAxisPermutation`] if the `axes`
///   permutation does not contain every axis index exactly once
pub fn swizzle_reshape<T: Copy, S: Data<Elem = T>>(
    data: ArrayBase<S, IxDyn>,
    axes: &[Axis],
) -> Result<Array<T, IxDyn>, SwizzleReshapeCodecError> {
    let (permutation, new_shape) = validate_into_axes_shape(&data, axes)?;

    let swizzled = data.permuted_axes(permutation);
    // TODO: use into_shape_clone for ndarray >=0.16
    #[allow(clippy::expect_used)] // only panics on an implementation bug
    let reshaped = swizzled
        .to_shape(new_shape)
        .expect("new encoding shape should have the correct number of elements")
        .into_owned();

    Ok(reshaped)
}

/// Reverts the swizzle and reshape of the `encoded` array with the `axes` and
/// returns the original array.
///
/// Since the shape of the original array is not known, only permutations of
/// axes are supported.
///
/// # Errors
///
/// Errors with
/// - [`SwizzleReshapeCodecError::CannotDecodeMergedAxes`] if any axes were
///   merged and thus cannot be split without further information
/// - [`SwizzleReshapeCodecError::InvalidAxisIndex`] if any axis is out of
///   bounds
/// - [`SwizzleReshapeCodecError::InvalidAxisPermutation`] if the `axes`
///   permutation does not contain every axis index exactly once
pub fn undo_swizzle_reshape<T: Copy, S: Data<Elem = T>>(
    encoded: ArrayBase<S, IxDyn>,
    axes: &[Axis],
) -> Result<Array<T, IxDyn>, SwizzleReshapeCodecError> {
    if !axes.iter().all(|axis| matches!(axis, Axis::Single(_))) {
        return Err(SwizzleReshapeCodecError::CannotDecodeMergedAxes);
    }

    let (permutation, _shape) = validate_into_axes_shape(&encoded, axes)?;

    #[allow(clippy::from_iter_instead_of_collect)]
    let mut inverse_permutation = Vec::from_iter(0..permutation.len());
    #[allow(clippy::indexing_slicing)] // all indices have been validated
    inverse_permutation.sort_by_key(|i| permutation[*i]);

    // since no axes were merged, no reshape is needed
    let unshaped = encoded;
    let unswizzled = unshaped.permuted_axes(inverse_permutation);

    Ok(unswizzled.into_owned())
}

#[allow(clippy::missing_panics_doc)]
#[allow(clippy::needless_pass_by_value)]
/// Reverts the swizzle and reshape of the `encoded` array with the `axes` and
/// outputs it into the `decoded` array.
///
/// # Errors
///
/// Errors with
/// - [`SwizzleReshapeCodecError::InvalidAxisIndex`] if any axis is out of
///   bounds
/// - [`SwizzleReshapeCodecError::InvalidAxisPermutation`] if the `axes`
///   permutation does not contain every axis index exactly once
/// - [`SwizzleReshapeCodecError::MismatchedDecodeIntoArray`] if the `encoded`
///   array's shape does not match the shape that swizzling and reshaping an
///   array of the `decoded` array's shape would have produced
pub fn undo_swizzle_reshape_into<T: Copy>(
    encoded: ArrayView<T, IxDyn>,
    mut decoded: ArrayViewMut<T, IxDyn>,
    axes: &[Axis],
) -> Result<(), SwizzleReshapeCodecError> {
    let (permutation, new_shape) = validate_into_axes_shape(&decoded, axes)?;

    if encoded.shape() != new_shape {
        return Err(SwizzleReshapeCodecError::MismatchedDecodeIntoArray {
            source: AnyArrayAssignError::ShapeMismatch {
                src: encoded.shape().to_vec(),
                dst: new_shape,
            },
        });
    }

    let mut permuted_shape = decoded.shape().to_vec();
    #[allow(clippy::indexing_slicing)] // all indices have been validated
    permuted_shape.sort_by_key(|i| permutation[*i]);

    #[allow(clippy::from_iter_instead_of_collect)]
    let mut inverse_permutation = Vec::from_iter(0..permutation.len());
    #[allow(clippy::indexing_slicing)] // all indices have been validated
    inverse_permutation.sort_by_key(|i| permutation[*i]);

    #[allow(clippy::expect_used)] // only panics on an implementation bug
    let unshaped = encoded
        .to_shape(permuted_shape)
        .expect("new decoding shape should have the correct number of elements");
    let unswizzled = unshaped.permuted_axes(inverse_permutation);

    decoded.assign(&unswizzled);

    Ok(())
}

fn validate_into_axes_shape<T, S: Data<Elem = T>>(
    array: &ArrayBase<S, IxDyn>,
    axes: &[Axis],
) -> Result<(Vec<usize>, Vec<usize>), SwizzleReshapeCodecError> {
    let mut new_axes = vec![0_usize; array.ndim()];
    let mut new_shape = vec![0_usize; axes.len()];
    let mut axis_counts = vec![0_usize; array.ndim()];

    for axis in axes {
        match axis {
            Axis::Single(index) => {
                if let Some(c) = axis_counts.get_mut(*index) {
                    *c += 1;
                    new_axes.push(*index);
                    new_shape.push(array.len_of(ndarray::Axis(*index)));
                } else {
                    return Err(SwizzleReshapeCodecError::InvalidAxisIndex {
                        index: *index,
                        ndim: array.ndim(),
                    });
                }
            }
            Axis::Merged(axes) => {
                let mut new_len = 1;
                for index in axes {
                    if let Some(c) = axis_counts.get_mut(*index) {
                        *c += 1;
                        new_axes.push(*index);
                        new_len *= array.len_of(ndarray::Axis(*index));
                    } else {
                        return Err(SwizzleReshapeCodecError::InvalidAxisIndex {
                            index: *index,
                            ndim: array.ndim(),
                        });
                    }
                }
                new_shape.push(new_len);
            }
        }
    }

    if !axis_counts.into_iter().all(|c| c == 1) {
        return Err(SwizzleReshapeCodecError::InvalidAxisPermutation {
            axes: axes.to_vec(),
            ndim: array.ndim(),
        });
    }

    Ok((new_axes, new_shape))
}

#[derive(Clone)]
/// A collection of multiple (≥2) elements
pub struct Multiple<T> {
    /// The first element
    pub first: T,
    /// The second element
    pub second: T,
    /// The remaining tail of elements
    pub tail: Vec<T>,
}

impl<T> Multiple<T> {
    /// Iterate over the elements in the collection
    pub fn iter(&self) -> <&Self as IntoIterator>::IntoIter {
        self.into_iter()
    }
}

impl<T: fmt::Debug> fmt::Debug for Multiple<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut seq = fmt.debug_list();
        seq.entry(&self.first);
        seq.entry(&self.second);
        seq.entries(&self.tail);
        seq.finish()
    }
}

impl<'a, T> IntoIterator for &'a Multiple<T> {
    type Item = &'a T;

    type IntoIter = Chain<Chain<Once<&'a T>, Once<&'a T>>, Iter<'a, T>>;

    fn into_iter(self) -> Self::IntoIter {
        std::iter::once(&self.first)
            .chain(std::iter::once(&self.second))
            .chain(&self.tail)
    }
}

impl<T: Serialize> Serialize for Multiple<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Multiple<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut vec = VecDeque::<T>::deserialize(deserializer)?;

        let len = vec.len();

        let (Some(first), Some(second)) = (vec.pop_front(), vec.pop_front()) else {
            return Err(serde::de::Error::invalid_length(
                len,
                &"a list of at least two elements",
            ));
        };

        Ok(Self {
            first,
            second,
            tail: vec.into(),
        })
    }
}

impl JsonSchema for Multiple<usize> {
    fn schema_name() -> Cow<'static, str> {
        Cow::Borrowed("MultipleUsize")
    }

    fn schema_id() -> Cow<'static, str> {
        Cow::Borrowed(concat!(module_path!(), "::", "Multiple<usize>"))
    }

    fn json_schema(gen: &mut SchemaGenerator) -> Schema {
        let mut schema = Vec::<usize>::json_schema(gen);
        schema
            .ensure_object()
            .insert(String::from("minItems"), 2.into());
        schema
    }
}