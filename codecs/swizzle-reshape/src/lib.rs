//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.81.0-blue
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
    fmt::{self, Debug},
};

use ndarray::{Array, ArrayBase, ArrayView, ArrayViewMut, Data, IxDyn};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig,
};
use schemars::{json_schema, JsonSchema, Schema, SchemaGenerator};
use serde::{
    de::{MapAccess, Visitor},
    ser::SerializeMap,
    Deserialize, Deserializer, Serialize, Serializer,
};
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
    /// The permutation of the axes that is applied on encoding.
    ///
    /// The permutation is given as a list of axis groups, where each group
    /// corresponds to one encoded output axis that may consist of several
    /// decoded input axes. For instance, `[[0], [1, 2]]` flattens a three-
    /// dimensional array into a two-dimensional one by combining the second and
    /// third axes.
    ///
    /// The permutation also allows specifying a special catch-all remaining
    /// axes marker:
    /// - `[[0], {}]` moves the second axis to be the first and appends all
    ///   other axes afterwards, i.e. the encoded array has the same number
    ///   of axes as the input array
    /// - `[[0], [{}]]` in contrast collapses all other axes into one, i.e.
    ///   the encoded array is two-dimensional
    pub axes: Vec<AxisGroup>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
#[serde(deny_unknown_fields)]
/// An axis group, potentially from a merged combination of multiple input axes
pub enum AxisGroup {
    /// A merged combination of zero, one, or multiple input axes
    Group(Vec<Axis>),
    /// All remaining axes, each in a separate single-axis group
    AllRest(Rest),
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
#[serde(deny_unknown_fields)]
/// An axis or all remaining axes
pub enum Axis {
    /// A single axis, as determined by its index
    Index(usize),
    /// All remaining axes, combined into one
    MergedRest(Rest),
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
        axes: Vec<AxisGroup>,
        /// The number of dimensions of the array
        ndim: usize,
    },
    /// [`SwizzleReshapeCodec`] cannot encode or decode with an axis permutation
    /// that contains multiple rest-axes markers
    #[error("SwizzleReshape cannot encode or decode with an axis permutation that contains multiple rest-axes markers")]
    MultipleRestAxes,
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
/// - [`SwizzleReshapeCodecError::MultipleRestAxes`] if the `axes` permutation
///   contains more than one [`Rest`]-axes marker
pub fn swizzle_reshape<T: Copy, S: Data<Elem = T>>(
    data: ArrayBase<S, IxDyn>,
    axes: &[AxisGroup],
) -> Result<Array<T, IxDyn>, SwizzleReshapeCodecError> {
    let (permutation, new_shape) = validate_into_axes_shape(&data, axes)?;

    let swizzled = data.permuted_axes(permutation);
    #[allow(clippy::expect_used)] // only panics on an implementation bug
    let reshaped = swizzled
        .into_owned()
        .into_shape_clone(new_shape)
        .expect("new encoding shape should have the correct number of elements");

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
/// - [`SwizzleReshapeCodecError::MultipleRestAxes`] if the `axes` permutation
///   contains more than one [`Rest`]-axes marker
pub fn undo_swizzle_reshape<T: Copy, S: Data<Elem = T>>(
    encoded: ArrayBase<S, IxDyn>,
    axes: &[AxisGroup],
) -> Result<Array<T, IxDyn>, SwizzleReshapeCodecError> {
    if !axes.iter().all(|axis| match axis {
        AxisGroup::Group(axes) => matches!(axes.as_slice(), [Axis::Index(_)]),
        AxisGroup::AllRest(Rest) => true,
    }) {
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
/// - [`SwizzleReshapeCodecError::MultipleRestAxes`] if the `axes` permutation
///   contains more than one [`Rest`]-axes marker
/// - [`SwizzleReshapeCodecError::MismatchedDecodeIntoArray`] if the `encoded`
///   array's shape does not match the shape that swizzling and reshaping an
///   array of the `decoded` array's shape would have produced
pub fn undo_swizzle_reshape_into<T: Copy>(
    encoded: ArrayView<T, IxDyn>,
    mut decoded: ArrayViewMut<T, IxDyn>,
    axes: &[AxisGroup],
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

    let mut permuted_shape_indices = decoded.shape().iter().enumerate().collect::<Vec<_>>();
    #[allow(clippy::indexing_slicing)] // all indices have been validated
    permuted_shape_indices.sort_by_key(|(i, _s)| permutation[*i]);

    let (inverse_permutation, permuted_shape): (Vec<_>, Vec<_>) =
        permuted_shape_indices.into_iter().unzip();

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
    axes: &[AxisGroup],
) -> Result<(Vec<usize>, Vec<usize>), SwizzleReshapeCodecError> {
    // counts of each axis index, used to check for missing or duplicate axes,
    //  and for knowing which axes are caught by the rest catch-all
    let mut axis_index_counts = vec![0_usize; array.ndim()];

    let mut has_rest = false;

    // validate that all axis indices are in bounds and that there is at most
    //  one catch-all remaining axes marker
    for group in axes {
        match group {
            AxisGroup::Group(axes) => {
                for axis in axes {
                    match axis {
                        Axis::Index(index) => {
                            if let Some(c) = axis_index_counts.get_mut(*index) {
                                *c += 1;
                            } else {
                                return Err(SwizzleReshapeCodecError::InvalidAxisIndex {
                                    index: *index,
                                    ndim: array.ndim(),
                                });
                            }
                        }
                        Axis::MergedRest(Rest) => {
                            if std::mem::replace(&mut has_rest, true) {
                                return Err(SwizzleReshapeCodecError::MultipleRestAxes);
                            }
                        }
                    }
                }
            }
            AxisGroup::AllRest(Rest) => {
                if std::mem::replace(&mut has_rest, true) {
                    return Err(SwizzleReshapeCodecError::MultipleRestAxes);
                }
            }
        }
    }

    // check that each axis is mentioned
    // - exactly once if no catch-all is used
    // - at most once if a catch-all is used
    if !axis_index_counts
        .iter()
        .all(|c| if has_rest { *c <= 1 } else { *c == 1 })
    {
        return Err(SwizzleReshapeCodecError::InvalidAxisPermutation {
            axes: axes.to_vec(),
            ndim: array.ndim(),
        });
    }

    // the permutation to apply to the input axes
    let mut axis_permutation = Vec::with_capacity(array.len());
    // the shape of the already permuted and grouped output array
    let mut grouped_shape = Vec::with_capacity(axes.len());

    for axis in axes {
        match axis {
            // a group merged all of its axes
            // an empty group adds an additional axis of size 1
            AxisGroup::Group(axes) => {
                let mut new_len = 1;
                for axis in axes {
                    match axis {
                        Axis::Index(index) => {
                            axis_permutation.push(*index);
                            new_len *= array.len_of(ndarray::Axis(*index));
                        }
                        Axis::MergedRest(Rest) => {
                            for (index, count) in axis_index_counts.iter().enumerate() {
                                if *count == 0 {
                                    axis_permutation.push(index);
                                    new_len *= array.len_of(ndarray::Axis(index));
                                }
                            }
                        }
                    }
                }
                grouped_shape.push(new_len);
            }
            AxisGroup::AllRest(Rest) => {
                for (index, count) in axis_index_counts.iter().enumerate() {
                    if *count == 0 {
                        axis_permutation.push(index);
                        grouped_shape.push(array.len_of(ndarray::Axis(index)));
                    }
                }
            }
        }
    }

    Ok((axis_permutation, grouped_shape))
}

#[derive(Copy, Clone, Debug)]
/// Marker to signify all remaining (not explicitly named) axes
pub struct Rest;

impl Serialize for Rest {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_map(Some(0))?.end()
    }
}

impl<'de> Deserialize<'de> for Rest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct RestVisitor;

        impl<'de> Visitor<'de> for RestVisitor {
            type Value = Rest;

            fn expecting(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.write_str("an empty map")
            }

            fn visit_map<A: MapAccess<'de>>(self, _map: A) -> Result<Self::Value, A::Error> {
                Ok(Rest)
            }
        }

        deserializer.deserialize_map(RestVisitor)
    }
}

impl JsonSchema for Rest {
    fn schema_name() -> Cow<'static, str> {
        Cow::Borrowed("Rest")
    }

    fn schema_id() -> Cow<'static, str> {
        Cow::Borrowed(concat!(module_path!(), "::", "Rest"))
    }

    fn json_schema(_gen: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "type": "object",
            "properties": {},
            "additionalProperties": false,
        })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn identity() {
        roundtrip(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            &[AxisGroup::AllRest(Rest)],
            &[2, 2, 2],
        );

        roundtrip(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            &[
                AxisGroup::Group(vec![Axis::Index(0)]),
                AxisGroup::Group(vec![Axis::Index(1)]),
                AxisGroup::AllRest(Rest),
            ],
            &[2, 2, 2],
        );
        roundtrip(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            &[
                AxisGroup::Group(vec![Axis::Index(0)]),
                AxisGroup::AllRest(Rest),
                AxisGroup::Group(vec![Axis::Index(2)]),
            ],
            &[2, 2, 2],
        );
        roundtrip(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            &[
                AxisGroup::AllRest(Rest),
                AxisGroup::Group(vec![Axis::Index(1)]),
                AxisGroup::Group(vec![Axis::Index(2)]),
            ],
            &[2, 2, 2],
        );

        roundtrip(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            &[
                AxisGroup::Group(vec![Axis::Index(0)]),
                AxisGroup::Group(vec![Axis::Index(1)]),
                AxisGroup::Group(vec![Axis::MergedRest(Rest)]),
            ],
            &[2, 2, 2],
        );
        roundtrip(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            &[
                AxisGroup::Group(vec![Axis::Index(0)]),
                AxisGroup::Group(vec![Axis::MergedRest(Rest)]),
                AxisGroup::Group(vec![Axis::Index(2)]),
            ],
            &[2, 2, 2],
        );
        roundtrip(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            &[
                AxisGroup::Group(vec![Axis::MergedRest(Rest)]),
                AxisGroup::Group(vec![Axis::Index(1)]),
                AxisGroup::Group(vec![Axis::Index(2)]),
            ],
            &[2, 2, 2],
        );
    }

    #[test]
    fn swizzle() {
        roundtrip(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            &[
                AxisGroup::Group(vec![Axis::Index(0)]),
                AxisGroup::Group(vec![Axis::Index(1)]),
                AxisGroup::AllRest(Rest),
            ],
            &[2, 2, 2],
        );
        roundtrip(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            &[
                AxisGroup::Group(vec![Axis::Index(2)]),
                AxisGroup::AllRest(Rest),
                AxisGroup::Group(vec![Axis::Index(1)]),
            ],
            &[2, 2, 2],
        );
        roundtrip(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            &[
                AxisGroup::AllRest(Rest),
                AxisGroup::Group(vec![Axis::Index(0)]),
                AxisGroup::Group(vec![Axis::Index(1)]),
            ],
            &[2, 2, 2],
        );

        roundtrip(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            &[
                AxisGroup::Group(vec![Axis::Index(0)]),
                AxisGroup::Group(vec![Axis::Index(1)]),
                AxisGroup::Group(vec![Axis::MergedRest(Rest)]),
            ],
            &[2, 2, 2],
        );
        roundtrip(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            &[
                AxisGroup::Group(vec![Axis::Index(2)]),
                AxisGroup::Group(vec![Axis::MergedRest(Rest)]),
                AxisGroup::Group(vec![Axis::Index(1)]),
            ],
            &[2, 2, 2],
        );
        roundtrip(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            &[
                AxisGroup::Group(vec![Axis::MergedRest(Rest)]),
                AxisGroup::Group(vec![Axis::Index(0)]),
                AxisGroup::Group(vec![Axis::Index(1)]),
            ],
            &[2, 2, 2],
        );
    }

    #[test]
    fn collapse() {
        roundtrip(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            &[AxisGroup::Group(vec![Axis::MergedRest(Rest)])],
            &[8],
        );

        roundtrip(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            &[AxisGroup::Group(vec![
                Axis::Index(0),
                Axis::Index(1),
                Axis::Index(2),
            ])],
            &[8],
        );
        roundtrip(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            &[AxisGroup::Group(vec![
                Axis::Index(2),
                Axis::Index(1),
                Axis::Index(0),
            ])],
            &[8],
        );
        roundtrip(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            &[AxisGroup::Group(vec![
                Axis::Index(1),
                Axis::MergedRest(Rest),
            ])],
            &[8],
        );

        roundtrip(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            &[
                AxisGroup::Group(vec![Axis::Index(0), Axis::Index(1)]),
                AxisGroup::AllRest(Rest),
            ],
            &[4, 2],
        );
        roundtrip(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            &[
                AxisGroup::Group(vec![Axis::Index(2)]),
                AxisGroup::Group(vec![Axis::Index(1), Axis::Index(0)]),
            ],
            &[2, 4],
        );
        roundtrip(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            &[
                AxisGroup::Group(vec![Axis::Index(1), Axis::MergedRest(Rest)]),
                AxisGroup::Group(vec![Axis::Index(0), Axis::Index(2)]),
            ],
            &[2, 4],
        );

        roundtrip(
            array![[1, 2], [3, 4], [5, 6], [7, 8]].into_dyn(),
            &[AxisGroup::Group(vec![Axis::MergedRest(Rest)])],
            &[8],
        );
    }

    #[test]
    fn extend() {
        roundtrip(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            &[
                AxisGroup::Group(vec![]),
                AxisGroup::Group(vec![Axis::Index(0)]),
                AxisGroup::Group(vec![]),
                AxisGroup::AllRest(Rest),
                AxisGroup::Group(vec![]),
                AxisGroup::Group(vec![Axis::Index(2)]),
                AxisGroup::Group(vec![]),
            ],
            &[1, 2, 1, 2, 1, 2, 1],
        );
    }

    #[allow(clippy::needless_pass_by_value)]
    fn roundtrip(data: Array<i32, IxDyn>, axes: &[AxisGroup], swizzle_shape: &[usize]) {
        let swizzled = swizzle_reshape(data.view(), axes).expect("swizzle should not fail");

        assert_eq!(swizzled.shape(), swizzle_shape);

        let mut unswizzled = Array::zeros(data.shape());
        undo_swizzle_reshape_into(swizzled.view(), unswizzled.view_mut(), axes)
            .expect("unswizzle into should not fail");

        assert_eq!(data, unswizzled);

        if axes.iter().any(|a| matches!(a, AxisGroup::Group(a) if a.len() != 1 || a.iter().any(|a| matches!(a, Axis::MergedRest(Rest))))) {
            undo_swizzle_reshape(swizzled.view(), axes).expect_err("unswizzle should fail");
        } else {
            let unswizzled = undo_swizzle_reshape(swizzled.view(), axes).expect("unswizzle should not fail");
            assert_eq!(data, unswizzled);
        }
    }
}
