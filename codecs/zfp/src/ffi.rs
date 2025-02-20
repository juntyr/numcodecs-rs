#![expect(unsafe_code)] // FFI

use std::{marker::PhantomData, mem::ManuallyDrop};

use ndarray::{ArrayView, Dimension};
use numcodecs::{AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayViewMut};

use crate::{ZfpCodecError, ZfpCompressionMode};

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
    pub fn with_bitstream<'a>(
        self,
        field: ZfpField<'a, T>,
    ) -> ZfpCompressionStreamWithBitstream<'a, T> {
        let this = ManuallyDrop::new(self);
        let field = ManuallyDrop::new(field);

        let capacity = unsafe { zfp_sys::zfp_stream_maximum_size(this.stream, field.field) };
        let mut buffer = vec![0_u8; capacity];

        let bitstream = unsafe { zfp_sys::stream_open(buffer.as_mut_ptr().cast(), buffer.len()) };

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

pub struct ZfpCompressionStreamWithBitstream<'a, T: ZfpCompressible> {
    stream: *mut zfp_sys::zfp_stream,
    bitstream: *mut zfp_sys::bitstream,
    field: *mut zfp_sys::zfp_field,
    buffer: Vec<u8>,
    _marker: PhantomData<&'a T>,
}

impl<'a, T: ZfpCompressible> ZfpCompressionStreamWithBitstream<'a, T> {
    pub fn write_full_header(
        self,
    ) -> Result<ZfpCompressionStreamWithBitstreamWithHeader<'a, T>, ZfpCodecError> {
        if unsafe { zfp_sys::zfp_write_header(self.stream, self.field, zfp_sys::ZFP_HEADER_FULL) }
            == 0
        {
            return Err(ZfpCodecError::HeaderEncodeFailed);
        }

        let mut this = ManuallyDrop::new(self);

        Ok(ZfpCompressionStreamWithBitstreamWithHeader {
            stream: this.stream,
            bitstream: this.bitstream,
            field: this.field,
            buffer: std::mem::take(&mut this.buffer),
            _marker: PhantomData::<&'a T>,
        })
    }
}

impl<T: ZfpCompressible> Drop for ZfpCompressionStreamWithBitstream<'_, T> {
    fn drop(&mut self) {
        unsafe { zfp_sys::zfp_field_free(self.field) };
        unsafe { zfp_sys::zfp_stream_close(self.stream) };
        unsafe { zfp_sys::stream_close(self.bitstream) };
    }
}

pub struct ZfpCompressionStreamWithBitstreamWithHeader<'a, T: ZfpCompressible> {
    stream: *mut zfp_sys::zfp_stream,
    bitstream: *mut zfp_sys::bitstream,
    field: *mut zfp_sys::zfp_field,
    buffer: Vec<u8>,
    _marker: PhantomData<&'a T>,
}

impl<T: ZfpCompressible> ZfpCompressionStreamWithBitstreamWithHeader<'_, T> {
    pub fn compress(mut self) -> Result<Vec<u8>, ZfpCodecError> {
        let compressed_size = unsafe { zfp_sys::zfp_compress(self.stream, self.field) };

        if compressed_size == 0 {
            return Err(ZfpCodecError::ZfpEncodeFailed);
        }

        let mut compressed = std::mem::take(&mut self.buffer);
        compressed.truncate(compressed_size);

        Ok(compressed)
    }
}

impl<T: ZfpCompressible> Drop for ZfpCompressionStreamWithBitstreamWithHeader<'_, T> {
    fn drop(&mut self) {
        unsafe { zfp_sys::zfp_field_free(self.field) };
        unsafe { zfp_sys::zfp_stream_close(self.stream) };
        unsafe { zfp_sys::stream_close(self.bitstream) };
    }
}

pub struct ZfpDecompressionStream<'a> {
    stream: *mut zfp_sys::zfp_stream,
    bitstream: *mut zfp_sys::bitstream,
    data: &'a [u8],
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
            data,
        }
    }

    pub fn read_full_header(self) -> Result<ZfpDecompressionStreamWithHeader<'a>, ZfpCodecError> {
        let this = ManuallyDrop::new(self);

        let field = unsafe { zfp_sys::zfp_field_alloc() };

        let stream = ZfpDecompressionStreamWithHeader {
            stream: this.stream,
            bitstream: this.bitstream,
            field,
            _data: this.data,
        };

        if unsafe { zfp_sys::zfp_read_header(this.stream, field, zfp_sys::ZFP_HEADER_FULL) } == 0 {
            return Err(ZfpCodecError::HeaderDecodeFailed);
        }

        Ok(stream)
    }
}

impl Drop for ZfpDecompressionStream<'_> {
    fn drop(&mut self) {
        unsafe { zfp_sys::zfp_stream_close(self.stream) };
        unsafe { zfp_sys::stream_close(self.bitstream) };
    }
}

pub struct ZfpDecompressionStreamWithHeader<'a> {
    stream: *mut zfp_sys::zfp_stream,
    bitstream: *mut zfp_sys::bitstream,
    field: *mut zfp_sys::zfp_field,
    _data: &'a [u8],
}

impl ZfpDecompressionStreamWithHeader<'_> {
    pub fn decompress(self) -> Result<AnyArray, ZfpCodecError> {
        let dtype = match unsafe { (*self.field).type_ } {
            zfp_sys::zfp_type_zfp_type_int32 => AnyArrayDType::I32,
            zfp_sys::zfp_type_zfp_type_int64 => AnyArrayDType::I64,
            zfp_sys::zfp_type_zfp_type_float => AnyArrayDType::F32,
            zfp_sys::zfp_type_zfp_type_double => AnyArrayDType::F64,
            dtype => return Err(ZfpCodecError::DecodeUnknownDtype(dtype)),
        };

        let shape = vec![
            unsafe { (*self.field).nw },
            unsafe { (*self.field).nz },
            unsafe { (*self.field).ny },
            unsafe { (*self.field).nx },
        ]
        .into_iter()
        .filter(|s| *s > 0)
        .collect::<Vec<usize>>();

        let (decompressed, result) = AnyArray::with_zeros_bytes(dtype, &shape, |bytes| {
            unsafe {
                zfp_sys::zfp_field_set_pointer(self.field, bytes.as_mut_ptr().cast());
            }

            if unsafe { zfp_sys::zfp_decompress(self.stream, self.field) } == 0 {
                Err(ZfpCodecError::ZfpDecodeFailed)
            } else {
                Ok(())
            }
        });

        result.map(|()| decompressed)
    }

    pub fn decompress_into(self, mut decompressed: AnyArrayViewMut) -> Result<(), ZfpCodecError> {
        let dtype = match unsafe { (*self.field).type_ } {
            zfp_sys::zfp_type_zfp_type_int32 => AnyArrayDType::I32,
            zfp_sys::zfp_type_zfp_type_int64 => AnyArrayDType::I64,
            zfp_sys::zfp_type_zfp_type_float => AnyArrayDType::F32,
            zfp_sys::zfp_type_zfp_type_double => AnyArrayDType::F64,
            dtype => return Err(ZfpCodecError::DecodeUnknownDtype(dtype)),
        };

        if decompressed.dtype() != dtype {
            return Err(ZfpCodecError::MismatchedDecodeIntoArray {
                source: AnyArrayAssignError::DTypeMismatch {
                    src: dtype,
                    dst: decompressed.dtype(),
                },
            });
        }

        let shape = vec![
            unsafe { (*self.field).nw },
            unsafe { (*self.field).nz },
            unsafe { (*self.field).ny },
            unsafe { (*self.field).nx },
        ]
        .into_iter()
        .filter(|s| *s > 0)
        .collect::<Vec<usize>>();

        if decompressed.shape() != shape {
            return Err(ZfpCodecError::MismatchedDecodeIntoArray {
                source: AnyArrayAssignError::ShapeMismatch {
                    src: shape,
                    dst: decompressed.shape().to_vec(),
                },
            });
        }

        decompressed.with_bytes_mut(|bytes| {
            unsafe {
                zfp_sys::zfp_field_set_pointer(self.field, bytes.as_mut_ptr().cast());
            }

            if unsafe { zfp_sys::zfp_decompress(self.stream, self.field) } == 0 {
                Err(ZfpCodecError::ZfpDecodeFailed)
            } else {
                Ok(())
            }
        })
    }
}

impl Drop for ZfpDecompressionStreamWithHeader<'_> {
    fn drop(&mut self) {
        unsafe { zfp_sys::zfp_field_free(self.field) };
        unsafe { zfp_sys::zfp_stream_close(self.stream) };
        unsafe { zfp_sys::stream_close(self.bitstream) };
    }
}

pub trait ZfpCompressible: Copy {
    const Z_TYPE: zfp_sys::zfp_type;
}

impl ZfpCompressible for i32 {
    const Z_TYPE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_int32;
}

impl ZfpCompressible for i64 {
    const Z_TYPE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_int64;
}

impl ZfpCompressible for f32 {
    const Z_TYPE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_float;
}

impl ZfpCompressible for f64 {
    const Z_TYPE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_double;
}
