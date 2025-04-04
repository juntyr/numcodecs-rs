#![expect(unsafe_code)] // FFI

use std::{marker::PhantomData, mem::ManuallyDrop};

use ndarray::{ArrayView, ArrayViewMut, Dimension, IxDyn};
use numcodecs::ArrayDType;

use crate::{ZfpClassicCodecError, ZfpCompressionMode, ZfpDType};

const ZFP_HEADER_NO_META: u32 = zfp_sys::ZFP_HEADER_FULL & !zfp_sys::ZFP_HEADER_META;

pub struct ZfpField<'a, T: ZfpCompressible> {
    field: *mut zfp_sys::zfp_field,
    dims: u32,
    _marker: PhantomData<&'a T>,
}

impl<'a, T: ZfpCompressible> ZfpField<'a, T> {
    #[expect(clippy::needless_pass_by_value)]
    pub fn new<D: Dimension>(data: ArrayView<'a, T, D>) -> Result<Self, ZfpClassicCodecError> {
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
                return Err(ZfpClassicCodecError::ExcessiveDimensionality {
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
    pub fn new(
        field: &ZfpField<T>,
        mode: &ZfpCompressionMode,
    ) -> Result<Self, ZfpClassicCodecError> {
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
                    return Err(ZfpClassicCodecError::InvalidExpertMode { mode: mode.clone() });
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

impl<'a, 'b, T: ZfpCompressible> ZfpCompressionStreamWithBitstream<'a, 'b, T> {
    pub fn write_header(
        self,
    ) -> Result<ZfpCompressionStreamWithBitstreamWithHeader<'a, 'b, T>, ZfpClassicCodecError> {
        if unsafe { zfp_sys::zfp_write_header(self.stream, self.field, ZFP_HEADER_NO_META) } == 0 {
            return Err(ZfpClassicCodecError::HeaderEncodeFailed);
        }

        let mut this = ManuallyDrop::new(self);

        Ok(ZfpCompressionStreamWithBitstreamWithHeader {
            stream: this.stream,
            bitstream: this.bitstream,
            field: this.field,
            // Safety: self is consumed, buffer is not read inside drop,
            //         the lifetime is carried on
            buffer: unsafe { &mut *std::ptr::from_mut(this.buffer) },
            _marker: PhantomData::<&'a T>,
        })
    }
}

impl<T: ZfpCompressible> Drop for ZfpCompressionStreamWithBitstream<'_, '_, T> {
    fn drop(&mut self) {
        unsafe { zfp_sys::zfp_field_free(self.field) };
        unsafe { zfp_sys::zfp_stream_close(self.stream) };
        unsafe { zfp_sys::stream_close(self.bitstream) };
    }
}

pub struct ZfpCompressionStreamWithBitstreamWithHeader<'a, 'b, T: ZfpCompressible> {
    stream: *mut zfp_sys::zfp_stream,
    bitstream: *mut zfp_sys::bitstream,
    field: *mut zfp_sys::zfp_field,
    buffer: &'b mut Vec<u8>,
    _marker: PhantomData<&'a T>,
}

impl<T: ZfpCompressible> ZfpCompressionStreamWithBitstreamWithHeader<'_, '_, T> {
    pub fn compress(self) -> Result<(), ZfpClassicCodecError> {
        let compressed_size = unsafe { zfp_sys::zfp_compress(self.stream, self.field) };

        if compressed_size == 0 {
            return Err(ZfpClassicCodecError::ZfpEncodeFailed);
        }

        // Safety: compressed_size bytes of the spare capacity have now been
        //         written to and initialized
        unsafe {
            self.buffer.set_len(self.buffer.len() + compressed_size);
        }

        Ok(())
    }
}

impl<T: ZfpCompressible> Drop for ZfpCompressionStreamWithBitstreamWithHeader<'_, '_, T> {
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

    pub fn read_header(self) -> Result<ZfpDecompressionStreamWithHeader<'a>, ZfpClassicCodecError> {
        let this = ManuallyDrop::new(self);

        let field = unsafe { zfp_sys::zfp_field_alloc() };

        let stream = ZfpDecompressionStreamWithHeader {
            stream: this.stream,
            bitstream: this.bitstream,
            field,
            _data: this.data,
        };

        if unsafe { zfp_sys::zfp_read_header(this.stream, field, ZFP_HEADER_NO_META) } == 0 {
            return Err(ZfpClassicCodecError::HeaderDecodeFailed);
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
    pub fn decompress_into<T: ZfpCompressible>(
        self,
        mut decompressed: ArrayViewMut<T, IxDyn>,
    ) -> Result<(), ZfpClassicCodecError> {
        unsafe { zfp_sys::zfp_field_set_type(self.field, T::Z_TYPE) };

        match (decompressed.shape(), decompressed.strides()) {
            ([nx], [sx]) => unsafe {
                zfp_sys::zfp_field_set_size_1d(self.field, *nx);
                zfp_sys::zfp_field_set_stride_1d(self.field, *sx);
            },
            ([ny, nx], [sy, sx]) => unsafe {
                zfp_sys::zfp_field_set_size_2d(self.field, *nx, *ny);
                zfp_sys::zfp_field_set_stride_2d(self.field, *sx, *sy);
            },
            ([nz, ny, nx], [sz, sy, sx]) => unsafe {
                zfp_sys::zfp_field_set_size_3d(self.field, *nx, *ny, *nz);
                zfp_sys::zfp_field_set_stride_3d(self.field, *sx, *sy, *sz);
            },
            ([nw, nz, ny, nx], [sw, sz, sy, sx]) => unsafe {
                zfp_sys::zfp_field_set_size_4d(self.field, *nx, *ny, *nz, *nw);
                zfp_sys::zfp_field_set_stride_4d(self.field, *sx, *sy, *sz, *sw);
            },
            (shape, _strides) => {
                return Err(ZfpClassicCodecError::ExcessiveDimensionality {
                    shape: shape.to_vec(),
                })
            }
        }

        unsafe {
            zfp_sys::zfp_field_set_pointer(
                self.field,
                decompressed.as_mut_ptr().cast::<std::ffi::c_void>(),
            );
        }

        if unsafe { zfp_sys::zfp_decompress(self.stream, self.field) } == 0 {
            Err(ZfpClassicCodecError::ZfpDecodeFailed)
        } else {
            Ok(())
        }
    }
}

impl Drop for ZfpDecompressionStreamWithHeader<'_> {
    fn drop(&mut self) {
        unsafe { zfp_sys::zfp_field_free(self.field) };
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
