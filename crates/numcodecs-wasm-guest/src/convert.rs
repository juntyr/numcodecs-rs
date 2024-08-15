use std::error::Error;

use ndarray::{Array, ArrayD, ShapeError};
use numcodecs::{AnyArray, AnyArrayDType};
use thiserror::Error;

use crate::wit;

pub fn from_wit_any_array(array: wit::AnyArray) -> Result<AnyArray, AnyArrayConversionError> {
    let shape = u32_as_usize_vec(array.shape);

    let array = match array.data {
        wit::AnyArrayData::U8(data) => AnyArray::U8(Array::from_shape_vec(shape, data)?),
        wit::AnyArrayData::U16(data) => AnyArray::U16(Array::from_shape_vec(shape, data)?),
        wit::AnyArrayData::U32(data) => AnyArray::U32(Array::from_shape_vec(shape, data)?),
        wit::AnyArrayData::U64(data) => AnyArray::U64(Array::from_shape_vec(shape, data)?),
        wit::AnyArrayData::I8(data) => AnyArray::I8(Array::from_shape_vec(shape, data)?),
        wit::AnyArrayData::I16(data) => AnyArray::I16(Array::from_shape_vec(shape, data)?),
        wit::AnyArrayData::I32(data) => AnyArray::I32(Array::from_shape_vec(shape, data)?),
        wit::AnyArrayData::I64(data) => AnyArray::I64(Array::from_shape_vec(shape, data)?),
        wit::AnyArrayData::F32(data) => AnyArray::F32(Array::from_shape_vec(shape, data)?),
        wit::AnyArrayData::F64(data) => AnyArray::F64(Array::from_shape_vec(shape, data)?),
    };

    Ok(array)
}

pub fn into_wit_any_array(array: AnyArray) -> Result<wit::AnyArray, AnyArrayConversionError> {
    fn array_into_standard_layout_vec<T>(array: ArrayD<T>) -> Vec<T> {
        if array.is_standard_layout() {
            array.into_raw_vec()
        } else {
            array.into_iter().collect()
        }
    }

    let shape = usize_as_u32_slice(array.shape());

    let data = match array {
        AnyArray::U8(array) => wit::AnyArrayData::U8(array_into_standard_layout_vec(array)),
        AnyArray::U16(array) => wit::AnyArrayData::U16(array_into_standard_layout_vec(array)),
        AnyArray::U32(array) => wit::AnyArrayData::U32(array_into_standard_layout_vec(array)),
        AnyArray::U64(array) => wit::AnyArrayData::U64(array_into_standard_layout_vec(array)),
        AnyArray::I8(array) => wit::AnyArrayData::I8(array_into_standard_layout_vec(array)),
        AnyArray::I16(array) => wit::AnyArrayData::I16(array_into_standard_layout_vec(array)),
        AnyArray::I32(array) => wit::AnyArrayData::I32(array_into_standard_layout_vec(array)),
        AnyArray::I64(array) => wit::AnyArrayData::I64(array_into_standard_layout_vec(array)),
        AnyArray::F32(array) => wit::AnyArrayData::F32(array_into_standard_layout_vec(array)),
        AnyArray::F64(array) => wit::AnyArrayData::F64(array_into_standard_layout_vec(array)),
        array => {
            return Err(AnyArrayConversionError::UnsupportedDtype {
                dtype: array.dtype(),
            })
        }
    };

    Ok(wit::AnyArray { data, shape })
}

#[derive(Debug, Error)]
pub enum AnyArrayConversionError {
    #[error("numcodecs-wasm-guest received an array of an invalid shape")]
    ShapeMismatch {
        #[from]
        source: ShapeError,
    },
    #[error("numcodecs-wasm-guest does not support transferring arrays of dtype {dtype}")]
    UnsupportedDtype { dtype: AnyArrayDType },
}

#[must_use]
pub fn into_wit_error<T: Error>(err: T) -> wit::Error {
    let mut source: Option<&dyn Error> = err.source();

    let mut error = wit::Error {
        message: format!("{err}"),
        chain: if source.is_some() {
            Vec::with_capacity(4)
        } else {
            Vec::new()
        },
    };

    while let Some(err) = source.take() {
        chain.push(format!("{err}"));
        source = err.source();
    }

    error
}

#[allow(clippy::cast_possible_truncation)]
#[must_use]
fn usize_as_u32_slice(slice: &[usize]) -> Vec<u32> {
    slice.iter().map(|x| *x as u32).collect()
}

#[must_use]
fn u32_as_usize_vec(vec: Vec<u32>) -> Vec<usize> {
    vec.into_iter().map(|x| x as usize).collect()
}
