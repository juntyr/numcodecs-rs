use std::error::Error;

use ndarray::{Array, ArrayD, ShapeError};
use numcodecs::{AnyArray, AnyArrayDType};
use thiserror::Error;

use crate::wit;

pub fn from_wit_any_array(
    array: wit::types::AnyArray,
) -> Result<AnyArray, AnyArrayConversionError> {
    let shape = u32_as_usize_vec(array.shape);

    let array = match array.data {
        wit::types::AnyArrayData::U8(data) => AnyArray::U8(Array::from_shape_vec(shape, data)?),
        wit::types::AnyArrayData::U16(data) => AnyArray::U16(Array::from_shape_vec(shape, data)?),
        wit::types::AnyArrayData::U32(data) => AnyArray::U32(Array::from_shape_vec(shape, data)?),
        wit::types::AnyArrayData::U64(data) => AnyArray::U64(Array::from_shape_vec(shape, data)?),
        wit::types::AnyArrayData::I8(data) => AnyArray::I8(Array::from_shape_vec(shape, data)?),
        wit::types::AnyArrayData::I16(data) => AnyArray::I16(Array::from_shape_vec(shape, data)?),
        wit::types::AnyArrayData::I32(data) => AnyArray::I32(Array::from_shape_vec(shape, data)?),
        wit::types::AnyArrayData::I64(data) => AnyArray::I64(Array::from_shape_vec(shape, data)?),
        wit::types::AnyArrayData::F32(data) => AnyArray::F32(Array::from_shape_vec(shape, data)?),
        wit::types::AnyArrayData::F64(data) => AnyArray::F64(Array::from_shape_vec(shape, data)?),
    };

    Ok(array)
}

pub fn zeros_from_wit_any_array_prototype(prototype: wit::types::AnyArrayPrototype) -> AnyArray {
    let shape = u32_as_usize_vec(prototype.shape);

    match prototype.dtype {
        wit::types::AnyArrayDtype::U8 => AnyArray::U8(Array::zeros(shape)),
        wit::types::AnyArrayDtype::U16 => AnyArray::U16(Array::zeros(shape)),
        wit::types::AnyArrayDtype::U32 => AnyArray::U32(Array::zeros(shape)),
        wit::types::AnyArrayDtype::U64 => AnyArray::U64(Array::zeros(shape)),
        wit::types::AnyArrayDtype::I8 => AnyArray::I8(Array::zeros(shape)),
        wit::types::AnyArrayDtype::I16 => AnyArray::I16(Array::zeros(shape)),
        wit::types::AnyArrayDtype::I32 => AnyArray::I32(Array::zeros(shape)),
        wit::types::AnyArrayDtype::I64 => AnyArray::I64(Array::zeros(shape)),
        wit::types::AnyArrayDtype::F32 => AnyArray::F32(Array::zeros(shape)),
        wit::types::AnyArrayDtype::F64 => AnyArray::F64(Array::zeros(shape)),
    }
}

pub fn into_wit_any_array(
    array: AnyArray,
) -> Result<wit::types::AnyArray, AnyArrayConversionError> {
    fn array_into_standard_layout_vec<T>(array: ArrayD<T>) -> Vec<T> {
        if array.is_standard_layout() {
            array.into_raw_vec_and_offset().0
        } else {
            array.into_iter().collect()
        }
    }

    let shape = usize_as_u32_slice(array.shape());

    let data = match array {
        AnyArray::U8(array) => wit::types::AnyArrayData::U8(array_into_standard_layout_vec(array)),
        AnyArray::U16(array) => {
            wit::types::AnyArrayData::U16(array_into_standard_layout_vec(array))
        }
        AnyArray::U32(array) => {
            wit::types::AnyArrayData::U32(array_into_standard_layout_vec(array))
        }
        AnyArray::U64(array) => {
            wit::types::AnyArrayData::U64(array_into_standard_layout_vec(array))
        }
        AnyArray::I8(array) => wit::types::AnyArrayData::I8(array_into_standard_layout_vec(array)),
        AnyArray::I16(array) => {
            wit::types::AnyArrayData::I16(array_into_standard_layout_vec(array))
        }
        AnyArray::I32(array) => {
            wit::types::AnyArrayData::I32(array_into_standard_layout_vec(array))
        }
        AnyArray::I64(array) => {
            wit::types::AnyArrayData::I64(array_into_standard_layout_vec(array))
        }
        AnyArray::F32(array) => {
            wit::types::AnyArrayData::F32(array_into_standard_layout_vec(array))
        }
        AnyArray::F64(array) => {
            wit::types::AnyArrayData::F64(array_into_standard_layout_vec(array))
        }
        array => {
            return Err(AnyArrayConversionError::UnsupportedDtype {
                dtype: array.dtype(),
            });
        }
    };

    Ok(wit::types::AnyArray { data, shape })
}

pub fn into_wit_any_array_dtype(
    dtype: AnyArrayDType,
) -> Result<wit::types::AnyArrayDtype, AnyArrayConversionError> {
    let dtype = match dtype {
        AnyArrayDType::U8 => wit::types::AnyArrayDType::U8,
        AnyArrayDType::U16 => wit::types::AnyArrayDType::U16,
        AnyArrayDType::U32 => wit::types::AnyArrayDType::U32,
        AnyArrayDType::U64 => wit::types::AnyArrayDType::U64,
        AnyArrayDType::I8 => wit::types::AnyArrayDType::I8,
        AnyArrayDType::I16 => wit::types::AnyArrayDType::I16,
        AnyArrayDType::I32 => wit::types::AnyArrayDType::I32,
        AnyArrayDType::I64 => wit::types::AnyArrayDType::I64,
        AnyArrayDType::F32 => wit::types::AnyArrayDType::F32,
        AnyArrayDType::F64 => wit::types::AnyArrayDType::F64,
        array => {
            return Err(AnyArrayConversionError::UnsupportedDtype { dtype });
        }
    };

    Ok(dtype)
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
pub fn into_wit_error<T: Error>(err: T) -> wit::types::Error {
    let mut source: Option<&dyn Error> = err.source();

    let mut error = wit::types::Error {
        message: format!("{err}"),
        chain: if source.is_some() {
            Vec::with_capacity(4)
        } else {
            Vec::new()
        },
    };

    while let Some(err) = source.take() {
        error.chain.push(format!("{err}"));
        source = err.source();
    }

    error
}

#[expect(clippy::cast_possible_truncation)]
#[must_use]
pub(crate) fn usize_as_u32_slice(slice: &[usize]) -> Vec<u32> {
    slice.iter().map(|x| *x as u32).collect()
}

#[must_use]
pub(crate) fn u32_as_usize_vec(vec: Vec<u32>) -> Vec<usize> {
    vec.into_iter().map(|x| x as usize).collect()
}
