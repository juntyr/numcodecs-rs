use std::{borrow::Cow, fmt, mem::ManuallyDrop};

use ndarray::{
    ArrayBase, ArrayD, CowRepr, Data, DataMut, IxDyn, OwnedArcRepr, OwnedRepr, RawData,
    RawDataClone, RawDataSubst, ViewRepr,
};
use serde::{Deserialize, Serialize};

/// An array where the data has shared ownership and is copy-on-write.
pub type AnyArcArray = AnyArrayBase<OwnedArcRepr<()>>;
/// An array that owns its data uniquely.
pub type AnyArray = AnyArrayBase<OwnedRepr<()>>;
/// A read-only array view.
pub type AnyArrayView<'a> = AnyArrayBase<ViewRepr<&'a ()>>;
/// A read-write array view.
pub type AnyArrayViewMut<'a> = AnyArrayBase<ViewRepr<&'a mut ()>>;
/// An array with copy-on-write behavior.
pub type AnyCowArray<'a> = AnyArrayBase<CowRepr<'a, ()>>;

/// Numeric n-dimensional arrays with dynamic shapes.
#[non_exhaustive]
#[allow(missing_docs)]
pub enum AnyArrayBase<T: AnyRawData> {
    U8(ArrayBase<T::U8, IxDyn>),
    U16(ArrayBase<T::U16, IxDyn>),
    U32(ArrayBase<T::U32, IxDyn>),
    U64(ArrayBase<T::U64, IxDyn>),
    I8(ArrayBase<T::I8, IxDyn>),
    I16(ArrayBase<T::I16, IxDyn>),
    I32(ArrayBase<T::I32, IxDyn>),
    I64(ArrayBase<T::I64, IxDyn>),
    F32(ArrayBase<T::F32, IxDyn>),
    F64(ArrayBase<T::F64, IxDyn>),
}

impl<T: AnyRawData> AnyArrayBase<T> {
    /// Returns the total number of elements in the array.
    pub fn len(&self) -> usize {
        match self {
            Self::U8(a) => a.len(),
            Self::U16(a) => a.len(),
            Self::U32(a) => a.len(),
            Self::U64(a) => a.len(),
            Self::I8(a) => a.len(),
            Self::I16(a) => a.len(),
            Self::I32(a) => a.len(),
            Self::I64(a) => a.len(),
            Self::F32(a) => a.len(),
            Self::F64(a) => a.len(),
        }
    }

    /// Returns whether the array has any elements.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::U8(a) => a.is_empty(),
            Self::U16(a) => a.is_empty(),
            Self::U32(a) => a.is_empty(),
            Self::U64(a) => a.is_empty(),
            Self::I8(a) => a.is_empty(),
            Self::I16(a) => a.is_empty(),
            Self::I32(a) => a.is_empty(),
            Self::I64(a) => a.is_empty(),
            Self::F32(a) => a.is_empty(),
            Self::F64(a) => a.is_empty(),
        }
    }

    /// Returns the dtype of the array.
    pub const fn dtype(&self) -> AnyArrayDType {
        match self {
            Self::U8(_) => AnyArrayDType::U8,
            Self::U16(_) => AnyArrayDType::U16,
            Self::U32(_) => AnyArrayDType::U32,
            Self::U64(_) => AnyArrayDType::U64,
            Self::I8(_) => AnyArrayDType::I8,
            Self::I16(_) => AnyArrayDType::I16,
            Self::I32(_) => AnyArrayDType::I32,
            Self::I64(_) => AnyArrayDType::I64,
            Self::F32(_) => AnyArrayDType::F32,
            Self::F64(_) => AnyArrayDType::F64,
        }
    }

    /// Returns the shape of the array.
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::U8(a) => a.shape(),
            Self::U16(a) => a.shape(),
            Self::U32(a) => a.shape(),
            Self::U64(a) => a.shape(),
            Self::I8(a) => a.shape(),
            Self::I16(a) => a.shape(),
            Self::I32(a) => a.shape(),
            Self::I64(a) => a.shape(),
            Self::F32(a) => a.shape(),
            Self::F64(a) => a.shape(),
        }
    }

    /// Return the strides of the array.
    pub fn strides(&self) -> &[isize] {
        match self {
            Self::U8(a) => a.strides(),
            Self::U16(a) => a.strides(),
            Self::U32(a) => a.strides(),
            Self::U64(a) => a.strides(),
            Self::I8(a) => a.strides(),
            Self::I16(a) => a.strides(),
            Self::I32(a) => a.strides(),
            Self::I64(a) => a.strides(),
            Self::F32(a) => a.strides(),
            Self::F64(a) => a.strides(),
        }
    }
}

impl<T: AnyRawData> AnyArrayBase<T>
where
    T::U8: Data,
    T::U16: Data,
    T::U32: Data,
    T::U64: Data,
    T::I8: Data,
    T::I16: Data,
    T::I32: Data,
    T::I64: Data,
    T::F32: Data,
    T::F64: Data,
{
    #[must_use]
    /// Returns a read-only view of the array.
    pub fn view(&self) -> AnyArrayView {
        match self {
            Self::U8(a) => AnyArrayView::U8(a.view()),
            Self::U16(a) => AnyArrayView::U16(a.view()),
            Self::U32(a) => AnyArrayView::U32(a.view()),
            Self::U64(a) => AnyArrayView::U64(a.view()),
            Self::I8(a) => AnyArrayView::I8(a.view()),
            Self::I16(a) => AnyArrayView::I16(a.view()),
            Self::I32(a) => AnyArrayView::I32(a.view()),
            Self::I64(a) => AnyArrayView::I64(a.view()),
            Self::F32(a) => AnyArrayView::F32(a.view()),
            Self::F64(a) => AnyArrayView::F64(a.view()),
        }
    }

    #[must_use]
    /// Returns a copy-on-write view of the array.
    pub fn cow(&self) -> AnyCowArray {
        match self {
            Self::U8(a) => AnyCowArray::U8(a.into()),
            Self::U16(a) => AnyCowArray::U16(a.into()),
            Self::U32(a) => AnyCowArray::U32(a.into()),
            Self::U64(a) => AnyCowArray::U64(a.into()),
            Self::I8(a) => AnyCowArray::I8(a.into()),
            Self::I16(a) => AnyCowArray::I16(a.into()),
            Self::I32(a) => AnyCowArray::I32(a.into()),
            Self::I64(a) => AnyCowArray::I64(a.into()),
            Self::F32(a) => AnyCowArray::F32(a.into()),
            Self::F64(a) => AnyCowArray::F64(a.into()),
        }
    }

    #[must_use]
    /// Turns the array into a uniquely owned array, cloning the array elements
    /// if necessary.
    pub fn into_owned(self) -> AnyArray {
        match self {
            Self::U8(a) => AnyArray::U8(a.into_owned()),
            Self::U16(a) => AnyArray::U16(a.into_owned()),
            Self::U32(a) => AnyArray::U32(a.into_owned()),
            Self::U64(a) => AnyArray::U64(a.into_owned()),
            Self::I8(a) => AnyArray::I8(a.into_owned()),
            Self::I16(a) => AnyArray::I16(a.into_owned()),
            Self::I32(a) => AnyArray::I32(a.into_owned()),
            Self::I64(a) => AnyArray::I64(a.into_owned()),
            Self::F32(a) => AnyArray::F32(a.into_owned()),
            Self::F64(a) => AnyArray::F64(a.into_owned()),
        }
    }

    #[must_use]
    /// Returns the array's data as a byte slice.
    ///
    /// If the array is contiguous and in standard order, i.e. if the element
    /// order in memory corresponds to the logical order of the array's
    /// elements, a view of the data is returned without cloning.
    ///
    /// Otherwise, the data is cloned and put into standard order first.
    pub fn as_bytes(&self) -> Cow<[u8]> {
        fn array_into_bytes<T: Copy, S: Data<Elem = T>>(x: &ArrayBase<S, IxDyn>) -> Cow<[u8]> {
            #[allow(clippy::option_if_let_else)]
            if let Some(x) = x.as_slice() {
                #[allow(unsafe_code)]
                // Safety: casting to a byte slice is only safe since this
                //         private helper function is only called for plain-
                //         old-data types and the slice's length is adjusted
                Cow::Borrowed(unsafe {
                    std::slice::from_raw_parts(x.as_ptr().cast::<u8>(), std::mem::size_of_val(x))
                })
            } else {
                let x = x.into_iter().copied().collect::<Vec<T>>();
                let mut x = ManuallyDrop::new(x);
                let (ptr, len, capacity) = (x.as_mut_ptr(), x.len(), x.capacity());
                #[allow(unsafe_code)]
                // Safety: transmuting to a byte vec is only safe since this
                //         private helper function is only called for plain-
                //         old-data types and the vec's length and capacity
                //         are adjusted
                let x = unsafe {
                    Vec::from_raw_parts(
                        ptr.cast::<u8>(),
                        len * std::mem::size_of::<T>(),
                        capacity * std::mem::size_of::<T>(),
                    )
                };
                Cow::Owned(x)
            }
        }

        match self {
            Self::U8(a) => array_into_bytes(a),
            Self::U16(a) => array_into_bytes(a),
            Self::U32(a) => array_into_bytes(a),
            Self::U64(a) => array_into_bytes(a),
            Self::I8(a) => array_into_bytes(a),
            Self::I16(a) => array_into_bytes(a),
            Self::I32(a) => array_into_bytes(a),
            Self::I64(a) => array_into_bytes(a),
            Self::F32(a) => array_into_bytes(a),
            Self::F64(a) => array_into_bytes(a),
        }
    }
}

impl<T: AnyRawData> AnyArrayBase<T>
where
    T::U8: DataMut,
    T::U16: DataMut,
    T::U32: DataMut,
    T::U64: DataMut,
    T::I8: DataMut,
    T::I16: DataMut,
    T::I32: DataMut,
    T::I64: DataMut,
    T::F32: DataMut,
    T::F64: DataMut,
{
    #[must_use]
    /// Returns a read-write view of the array.
    pub fn view_mut(&mut self) -> AnyArrayViewMut {
        match self {
            Self::U8(a) => AnyArrayViewMut::U8(a.view_mut()),
            Self::U16(a) => AnyArrayViewMut::U16(a.view_mut()),
            Self::U32(a) => AnyArrayViewMut::U32(a.view_mut()),
            Self::U64(a) => AnyArrayViewMut::U64(a.view_mut()),
            Self::I8(a) => AnyArrayViewMut::I8(a.view_mut()),
            Self::I16(a) => AnyArrayViewMut::I16(a.view_mut()),
            Self::I32(a) => AnyArrayViewMut::I32(a.view_mut()),
            Self::I64(a) => AnyArrayViewMut::I64(a.view_mut()),
            Self::F32(a) => AnyArrayViewMut::F32(a.view_mut()),
            Self::F64(a) => AnyArrayViewMut::F64(a.view_mut()),
        }
    }

    #[must_use]
    /// Provides access to the array's data as a mutable byte slice.
    ///
    /// If the array is contiguous and in standard order, i.e. if the element
    /// order in memory corresponds to the logical order of the array's
    /// elements, a mutable view of the data is returned without cloning.
    ///
    /// Otherwise, the data is cloned and put into standard order first, and
    /// later copied back into the array.
    pub fn with_bytes_mut<O>(&mut self, with: impl FnOnce(&mut [u8]) -> O) -> O {
        fn array_with_bytes_mut<T: Copy, S: DataMut<Elem = T>, O>(
            x: &mut ArrayBase<S, IxDyn>,
            with: impl FnOnce(&mut [u8]) -> O,
        ) -> O {
            if let Some(x) = x.as_slice_mut() {
                #[allow(unsafe_code)]
                // Safety: casting to a byte slice is only safe since this
                //         private helper function is only called for plain-
                //         old-data types and the slice's length is adjusted
                with(unsafe {
                    std::slice::from_raw_parts_mut(
                        x.as_mut_ptr().cast::<u8>(),
                        std::mem::size_of_val(x),
                    )
                })
            } else {
                let mut x_vec: Vec<T> = x.into_iter().map(|x| *x).collect::<Vec<T>>();

                #[allow(unsafe_code)]
                // Safety: casting to a byte slice is only safe since this
                //         private helper function is only called for plain-
                //         old-data types and the slice's length is adjusted
                let result = with(unsafe {
                    std::slice::from_raw_parts_mut(
                        x_vec.as_mut_ptr().cast::<u8>(),
                        std::mem::size_of_val(x_vec.as_slice()),
                    )
                });

                x.iter_mut().zip(x_vec).for_each(|(x, x_new)| *x = x_new);
                result
            }
        }

        match self {
            Self::U8(a) => array_with_bytes_mut(a, with),
            Self::U16(a) => array_with_bytes_mut(a, with),
            Self::U32(a) => array_with_bytes_mut(a, with),
            Self::U64(a) => array_with_bytes_mut(a, with),
            Self::I8(a) => array_with_bytes_mut(a, with),
            Self::I16(a) => array_with_bytes_mut(a, with),
            Self::I32(a) => array_with_bytes_mut(a, with),
            Self::I64(a) => array_with_bytes_mut(a, with),
            Self::F32(a) => array_with_bytes_mut(a, with),
            Self::F64(a) => array_with_bytes_mut(a, with),
        }
    }
}

impl AnyArray {
    #[must_use]
    /// Create an array with zeros of `dtype` and shape `shape`.
    pub fn zeros(dtype: AnyArrayDType, shape: &[usize]) -> Self {
        match dtype {
            AnyArrayDType::U8 => Self::U8(ArrayD::zeros(shape)),
            AnyArrayDType::U16 => Self::U16(ArrayD::zeros(shape)),
            AnyArrayDType::U32 => Self::U32(ArrayD::zeros(shape)),
            AnyArrayDType::U64 => Self::U64(ArrayD::zeros(shape)),
            AnyArrayDType::I8 => Self::I8(ArrayD::zeros(shape)),
            AnyArrayDType::I16 => Self::I16(ArrayD::zeros(shape)),
            AnyArrayDType::I32 => Self::I32(ArrayD::zeros(shape)),
            AnyArrayDType::I64 => Self::I64(ArrayD::zeros(shape)),
            AnyArrayDType::F32 => Self::F32(ArrayD::zeros(shape)),
            AnyArrayDType::F64 => Self::F64(ArrayD::zeros(shape)),
        }
    }

    /// Create an array with zeros of `dtype` and shape `shape`, and provide
    /// mutable access to the bytes of the array.
    ///
    /// The array is created to be contiguous and in standard order, i.e. its
    /// element order in memory corresponds to the logical order of the array's
    /// elements.
    pub fn with_zeros_bytes<T>(
        dtype: AnyArrayDType,
        shape: &[usize],
        with: impl FnOnce(&mut [u8]) -> T,
    ) -> (Self, T) {
        fn standard_array_as_bytes_mut<T: Copy>(x: &mut ArrayD<T>) -> &mut [u8] {
            #[allow(unsafe_code)]
            // Safety: casting to a byte slice is only safe since this
            //         private helper function is only called for plain-
            //         old-data types, the slice's length is adjusted,
            //         and the array is already in contiguous standard-
            //         order layout
            unsafe {
                std::slice::from_raw_parts_mut(
                    x.as_mut_ptr().cast::<u8>(),
                    x.len() * std::mem::size_of::<T>(),
                )
            }
        }

        let mut array = Self::zeros(dtype, shape);

        let result = match &mut array {
            Self::U8(a) => with(standard_array_as_bytes_mut(a)),
            Self::U16(a) => with(standard_array_as_bytes_mut(a)),
            Self::U32(a) => with(standard_array_as_bytes_mut(a)),
            Self::U64(a) => with(standard_array_as_bytes_mut(a)),
            Self::I8(a) => with(standard_array_as_bytes_mut(a)),
            Self::I16(a) => with(standard_array_as_bytes_mut(a)),
            Self::I32(a) => with(standard_array_as_bytes_mut(a)),
            Self::I64(a) => with(standard_array_as_bytes_mut(a)),
            Self::F32(a) => with(standard_array_as_bytes_mut(a)),
            Self::F64(a) => with(standard_array_as_bytes_mut(a)),
        };

        (array, result)
    }
}

impl<T: AnyRawData> Clone for AnyArrayBase<T>
where
    T::U8: RawDataClone,
    T::U16: RawDataClone,
    T::U32: RawDataClone,
    T::U64: RawDataClone,
    T::I8: RawDataClone,
    T::I16: RawDataClone,
    T::I32: RawDataClone,
    T::I64: RawDataClone,
    T::F32: RawDataClone,
    T::F64: RawDataClone,
{
    fn clone(&self) -> Self {
        match self {
            Self::U8(a) => Self::U8(a.clone()),
            Self::U16(a) => Self::U16(a.clone()),
            Self::U32(a) => Self::U32(a.clone()),
            Self::U64(a) => Self::U64(a.clone()),
            Self::I8(a) => Self::I8(a.clone()),
            Self::I16(a) => Self::I16(a.clone()),
            Self::I32(a) => Self::I32(a.clone()),
            Self::I64(a) => Self::I64(a.clone()),
            Self::F32(a) => Self::F32(a.clone()),
            Self::F64(a) => Self::F64(a.clone()),
        }
    }
}

impl<T: AnyRawData> fmt::Debug for AnyArrayBase<T>
where
    T::U8: Data,
    T::U16: Data,
    T::U32: Data,
    T::U64: Data,
    T::I8: Data,
    T::I16: Data,
    T::I32: Data,
    T::I64: Data,
    T::F32: Data,
    T::F64: Data,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::U8(a) => fmt.debug_tuple("U8").field(a).finish(),
            Self::U16(a) => fmt.debug_tuple("U16").field(a).finish(),
            Self::U32(a) => fmt.debug_tuple("U32").field(a).finish(),
            Self::U64(a) => fmt.debug_tuple("U64").field(a).finish(),
            Self::I8(a) => fmt.debug_tuple("I8").field(a).finish(),
            Self::I16(a) => fmt.debug_tuple("I16").field(a).finish(),
            Self::I32(a) => fmt.debug_tuple("I32").field(a).finish(),
            Self::I64(a) => fmt.debug_tuple("I64").field(a).finish(),
            Self::F32(a) => fmt.debug_tuple("F32").field(a).finish(),
            Self::F64(a) => fmt.debug_tuple("F64").field(a).finish(),
        }
    }
}

impl<T: AnyRawData> PartialEq for AnyArrayBase<T>
where
    T::U8: Data,
    T::U16: Data,
    T::U32: Data,
    T::U64: Data,
    T::I8: Data,
    T::I16: Data,
    T::I32: Data,
    T::I64: Data,
    T::F32: Data,
    T::F64: Data,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::U8(l), Self::U8(r)) => l == r,
            (Self::U16(l), Self::U16(r)) => l == r,
            (Self::U32(l), Self::U32(r)) => l == r,
            (Self::U64(l), Self::U64(r)) => l == r,
            (Self::I8(l), Self::I8(r)) => l == r,
            (Self::I16(l), Self::I16(r)) => l == r,
            (Self::I32(l), Self::I32(r)) => l == r,
            (Self::I64(l), Self::I64(r)) => l == r,
            (Self::F32(l), Self::F32(r)) => l == r,
            (Self::F64(l), Self::F64(r)) => l == r,
            _ => false,
        }
    }
}

/// Array-representation support for all dtypes included in [`AnyArrayBase`].
#[allow(missing_docs)]
pub trait AnyRawData {
    type U8: RawData<Elem = u8>;
    type U16: RawData<Elem = u16>;
    type U32: RawData<Elem = u32>;
    type U64: RawData<Elem = u64>;
    type I8: RawData<Elem = i8>;
    type I16: RawData<Elem = i16>;
    type I32: RawData<Elem = i32>;
    type I64: RawData<Elem = i64>;
    type F32: RawData<Elem = f32>;
    type F64: RawData<Elem = f64>;
}

impl<
        T: RawDataSubst<u8>
            + RawDataSubst<u16>
            + RawDataSubst<u32>
            + RawDataSubst<u64>
            + RawDataSubst<i8>
            + RawDataSubst<i16>
            + RawDataSubst<i32>
            + RawDataSubst<i64>
            + RawDataSubst<f32>
            + RawDataSubst<f64>,
    > AnyRawData for T
{
    type U8 = <T as RawDataSubst<u8>>::Output;
    type U16 = <T as RawDataSubst<u16>>::Output;
    type U32 = <T as RawDataSubst<u32>>::Output;
    type U64 = <T as RawDataSubst<u64>>::Output;
    type I8 = <T as RawDataSubst<i8>>::Output;
    type I16 = <T as RawDataSubst<i16>>::Output;
    type I32 = <T as RawDataSubst<i32>>::Output;
    type I64 = <T as RawDataSubst<i64>>::Output;
    type F32 = <T as RawDataSubst<f32>>::Output;
    type F64 = <T as RawDataSubst<f64>>::Output;
}

/// Enum of all dtypes included in [`AnyArrayBase`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
#[allow(missing_docs)]
pub enum AnyArrayDType {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
}

impl fmt::Display for AnyArrayDType {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str(match self {
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
        })
    }
}
