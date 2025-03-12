#![expect(unsafe_code)] // FFI

use std::{marker::PhantomData, mem::ManuallyDrop};

use ndarray::{ArrayView, ArrayViewMut, Dimension, IxDyn};
use numcodecs::{AnyArrayViewMut, ArrayDType};

use crate::{ZfpCodecError, ZfpCompressionMode, ZfpDType};

pub struct ZfpField<'a, T: ZfpCompressible> {
    field: *mut zfp_sys::zfp_field,
    dims: u32,
    _marker: PhantomData<&'a T>,
}

impl<'a, T: ZfpCompressible> ZfpField<'a, T> {
    #[expect(clippy::needless_pass_by_value)]
    pub fn new<D: Dimension>(data: ArrayView<'a, T, D>) -> Result<Self, ZfpCodecError> {
        let pointer: *mut std::ffi::c_void = data.as_ptr().cast::<std::ffi::c_void>().cast_mut();

        let (field, dims) = match (data.shape(), data.strides()) {
            ([nx], [sx]) => unsafe {
                let field = zfp_sys::zfp_field_1d(pointer, T::Z_TYPE, *nx);
                zfp_sys::zfp_field_set_stride_1d(field, *sx);
                (field, 1)
            },
            ([ny, nx], [sy, sx]) => unsafe {
                let field = zfp_sys::zfp_field_2d(pointer, T::Z_TYPE, *nx, *ny);
                zfp_sys::zfp_field_set_stride_2d(field, *sx, *sy);
                (field, 2)
            },
            ([nz, ny, nx], [sz, sy, sx]) => unsafe {
                let field = zfp_sys::zfp_field_3d(pointer, T::Z_TYPE, *nx, *ny, *nz);
                zfp_sys::zfp_field_set_stride_3d(field, *sx, *sy, *sz);
                (field, 3)
            },
            ([nw, nz, ny, nx], [sw, sz, sy, sx]) => unsafe {
                let field = zfp_sys::zfp_field_4d(pointer, T::Z_TYPE, *nx, *ny, *nz, *nw);
                zfp_sys::zfp_field_set_stride_4d(field, *sx, *sy, *sz, *sw);
                (field, 4)
            },
            (shape, _strides) => {
                return Err(ZfpCodecError::ExcessiveDimensionality {
                    shape: shape.to_vec(),
                })
            }
        };

        Ok(Self {
            field,
            dims,
            _marker: PhantomData::<&'a T>,
        })
    }
}

impl<T: ZfpCompressible> Drop for ZfpField<'_, T> {
    fn drop(&mut self) {
        unsafe { zfp_sys::zfp_field_free(self.field) };
    }
}

pub struct ZfpCompressionStream<T: ZfpCompressible> {
    stream: *mut zfp_sys::zfp_stream,
    _marker: PhantomData<T>,
}

impl<T: ZfpCompressible> ZfpCompressionStream<T> {
    pub fn new(field: &ZfpField<T>, mode: &ZfpCompressionMode) -> Result<Self, ZfpCodecError> {
        let stream = unsafe { zfp_sys::zfp_stream_open(std::ptr::null_mut()) };
        let stream = Self {
            stream,
            _marker: PhantomData::<T>,
        };

        match mode {
            ZfpCompressionMode::Expert {
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            } => {
                #[expect(clippy::cast_possible_wrap)]
                const ZFP_TRUE: zfp_sys::zfp_bool = zfp_sys::zfp_true as zfp_sys::zfp_bool;

                if unsafe {
                    zfp_sys::zfp_stream_set_params(
                        stream.stream,
                        *min_bits,
                        *max_bits,
                        *max_prec,
                        *min_exp,
                    )
                } != ZFP_TRUE
                {
                    return Err(ZfpCodecError::InvalidExpertMode { mode: mode.clone() });
                }
            }
            ZfpCompressionMode::FixedRate { rate } => {
                let _actual_rate: f64 = unsafe {
                    zfp_sys::zfp_stream_set_rate(stream.stream, *rate, T::Z_TYPE, field.dims, 0)
                };
            }
            ZfpCompressionMode::FixedPrecision { precision } => {
                let _actual_precision: u32 =
                    unsafe { zfp_sys::zfp_stream_set_precision(stream.stream, *precision) };
            }
            ZfpCompressionMode::FixedAccuracy { tolerance } => {
                let _actual_tolerance: f64 =
                    unsafe { zfp_sys::zfp_stream_set_accuracy(stream.stream, *tolerance) };
            }
            ZfpCompressionMode::Reversible => {
                let () = unsafe { zfp_sys::zfp_stream_set_reversible(stream.stream) };
            }
        }

        Ok(stream)
    }

    #[must_use]
    pub fn with_bitstream<'a, 'b>(
        self,
        field: ZfpField<'a, T>,
        buffer: &'b mut Vec<u8>,
    ) -> ZfpCompressionStreamWithBitstream<'a, 'b, T> {
        let this = ManuallyDrop::new(self);
        let field = ManuallyDrop::new(field);

        let capacity = unsafe { zfp_sys::zfp_stream_maximum_size(this.stream, field.field) };
        buffer.reserve(capacity);

        let bitstream = unsafe {
            zfp_sys::stream_open(buffer.spare_capacity_mut().as_mut_ptr().cast(), capacity)
        };

        unsafe { zfp_sys::zfp_stream_set_bit_stream(this.stream, bitstream) };
        unsafe { zfp_sys::zfp_stream_rewind(this.stream) };

        ZfpCompressionStreamWithBitstream {
            stream: this.stream,
            bitstream,
            field: field.field,
            buffer,
            _marker: PhantomData::<&'a T>,
        }
    }
}

impl<T: ZfpCompressible> Drop for ZfpCompressionStream<T> {
    fn drop(&mut self) {
        unsafe { zfp_sys::zfp_stream_close(self.stream) };
    }
}

pub struct ZfpCompressionStreamWithBitstream<'a, 'b, T: ZfpCompressible> {
    stream: *mut zfp_sys::zfp_stream,
    bitstream: *mut zfp_sys::bitstream,
    field: *mut zfp_sys::zfp_field,
    buffer: &'b mut Vec<u8>,
    _marker: PhantomData<&'a T>,
}

impl<T: ZfpCompressible> ZfpCompressionStreamWithBitstream<'_, '_, T> {
    pub fn compress(self) -> Result<(), ZfpCodecError> {
        let compressed_size = unsafe { zfp_sys::zfp_compress(self.stream, self.field) };

        if compressed_size == 0 {
            return Err(ZfpCodecError::ZfpEncodeFailed);
        }

        // Safety: compressed_size bytes of the spare capacity have now been
        //         written to and initialized
        unsafe {
            self.buffer.set_len(self.buffer.len() + compressed_size);
        }

        Ok(())
    }
}

impl<T: ZfpCompressible> Drop for ZfpCompressionStreamWithBitstream<'_, '_, T> {
    fn drop(&mut self) {
        unsafe { zfp_sys::zfp_field_free(self.field) };
        unsafe { zfp_sys::zfp_stream_close(self.stream) };
        unsafe { zfp_sys::stream_close(self.bitstream) };
    }
}

pub struct ZfpDecompressionStream<'a> {
    stream: *mut zfp_sys::zfp_stream,
    bitstream: *mut zfp_sys::bitstream,
    _data: &'a [u8],
}

impl<'a> ZfpDecompressionStream<'a> {
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        let bitstream = unsafe {
            zfp_sys::stream_open(
                data.as_ptr().cast::<std::ffi::c_void>().cast_mut(),
                data.len(),
            )
        };

        let stream = unsafe { zfp_sys::zfp_stream_open(bitstream) };

        Self {
            stream,
            bitstream,
            _data: data,
        }
    }

    pub fn decompress<T: ZfpCompressible>(
        self,
        mode: &ZfpCompressionMode,
        mut decompressed: ArrayViewMut<T, IxDyn>,
    ) -> Result<(), ZfpCodecError> {
        let (shape, strides) = (
            decompressed.shape().to_vec(),
            decompressed.strides().to_vec(),
        );

        AnyArrayViewMut::with_bytes_mut_typed(&mut decompressed, |bytes| {
            let pointer = bytes.as_mut_ptr().cast();

            let (field, dims) = match (shape.as_slice(), strides.as_slice()) {
                ([nx], [sx]) => unsafe {
                    let field = zfp_sys::zfp_field_1d(pointer, T::Z_TYPE, *nx);
                    zfp_sys::zfp_field_set_stride_1d(field, *sx);
                    (field, 1)
                },
                ([ny, nx], [sy, sx]) => unsafe {
                    let field = zfp_sys::zfp_field_2d(pointer, T::Z_TYPE, *nx, *ny);
                    zfp_sys::zfp_field_set_stride_2d(field, *sx, *sy);
                    (field, 2)
                },
                ([nz, ny, nx], [sz, sy, sx]) => unsafe {
                    let field = zfp_sys::zfp_field_3d(pointer, T::Z_TYPE, *nx, *ny, *nz);
                    zfp_sys::zfp_field_set_stride_3d(field, *sx, *sy, *sz);
                    (field, 3)
                },
                ([nw, nz, ny, nx], [sw, sz, sy, sx]) => unsafe {
                    let field = zfp_sys::zfp_field_4d(pointer, T::Z_TYPE, *nx, *ny, *nz, *nw);
                    zfp_sys::zfp_field_set_stride_4d(field, *sx, *sy, *sz, *sw);
                    (field, 4)
                },
                (shape, _strides) => {
                    return Err(ZfpCodecError::ExcessiveDimensionality {
                        shape: shape.to_vec(),
                    })
                }
            };

            match mode {
                ZfpCompressionMode::Expert {
                    min_bits,
                    max_bits,
                    max_prec,
                    min_exp,
                } => {
                    #[expect(clippy::cast_possible_wrap)]
                    const ZFP_TRUE: zfp_sys::zfp_bool = zfp_sys::zfp_true as zfp_sys::zfp_bool;

                    if unsafe {
                        zfp_sys::zfp_stream_set_params(
                            self.stream,
                            *min_bits,
                            *max_bits,
                            *max_prec,
                            *min_exp,
                        )
                    } != ZFP_TRUE
                    {
                        unsafe { zfp_sys::zfp_field_free(field) };
                        return Err(ZfpCodecError::InvalidExpertMode { mode: mode.clone() });
                    }
                }
                ZfpCompressionMode::FixedRate { rate } => {
                    let _actual_rate: f64 = unsafe {
                        zfp_sys::zfp_stream_set_rate(self.stream, *rate, T::Z_TYPE, dims, 0)
                    };
                }
                ZfpCompressionMode::FixedPrecision { precision } => {
                    let _actual_precision: u32 =
                        unsafe { zfp_sys::zfp_stream_set_precision(self.stream, *precision) };
                }
                ZfpCompressionMode::FixedAccuracy { tolerance } => {
                    let _actual_tolerance: f64 =
                        unsafe { zfp_sys::zfp_stream_set_accuracy(self.stream, *tolerance) };
                }
                ZfpCompressionMode::Reversible => {
                    let () = unsafe { zfp_sys::zfp_stream_set_reversible(self.stream) };
                }
            }

            let result = unsafe { zfp_sys::zfp_decompress(self.stream, field) };

            unsafe { zfp_sys::zfp_field_free(field) };

            if result == 0 {
                Err(ZfpCodecError::ZfpDecodeFailed)
            } else {
                Ok(())
            }
        })
    }
}

impl Drop for ZfpDecompressionStream<'_> {
    fn drop(&mut self) {
        unsafe { zfp_sys::zfp_stream_close(self.stream) };
        unsafe { zfp_sys::stream_close(self.bitstream) };
    }
}

pub trait ZfpCompressible: Copy + ArrayDType {
    const D_TYPE: ZfpDType;
    const Z_TYPE: zfp_sys::zfp_type;

    fn is_finite(self) -> bool;
}

impl ZfpCompressible for i32 {
    const D_TYPE: ZfpDType = ZfpDType::I32;
    const Z_TYPE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_int32;

    fn is_finite(self) -> bool {
        true
    }
}

impl ZfpCompressible for i64 {
    const D_TYPE: ZfpDType = ZfpDType::I64;
    const Z_TYPE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_int64;

    fn is_finite(self) -> bool {
        true
    }
}

impl ZfpCompressible for f32 {
    const D_TYPE: ZfpDType = ZfpDType::F32;
    const Z_TYPE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_float;

    fn is_finite(self) -> bool {
        Self::is_finite(self)
    }
}

impl ZfpCompressible for f64 {
    const D_TYPE: ZfpDType = ZfpDType::F64;
    const Z_TYPE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_double;

    fn is_finite(self) -> bool {
        Self::is_finite(self)
    }
}
