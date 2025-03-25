use std::ptr::NonNull;

use super::Jpeg2000Error;

pub struct Decoder {
    codec: NonNull<openjpeg_sys::opj_codec_t>,
}

impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe {
            openjpeg_sys::opj_destroy_codec(self.codec.as_ptr());
        }
    }
}

impl Decoder {
    pub fn j2k() -> Result<Self, Jpeg2000Error> {
        let codec = NonNull::new(unsafe {
            openjpeg_sys::opj_create_decompress(openjpeg_sys::OPJ_CODEC_FORMAT::OPJ_CODEC_J2K)
        })
        .ok_or(Jpeg2000Error::DecoderSetupError)?;

        unsafe {
            openjpeg_sys::opj_set_info_handler(codec.as_ptr(), Some(log), std::ptr::null_mut());
        }
        unsafe {
            openjpeg_sys::opj_set_warning_handler(codec.as_ptr(), Some(log), std::ptr::null_mut());
        }
        unsafe {
            openjpeg_sys::opj_set_error_handler(codec.as_ptr(), Some(log), std::ptr::null_mut());
        }

        Ok(Self { codec })
    }

    pub fn as_raw(&mut self) -> *mut openjpeg_sys::opj_codec_t {
        self.codec.as_ptr()
    }
}

pub struct Encoder {
    codec: NonNull<openjpeg_sys::opj_codec_t>,
}

impl Drop for Encoder {
    fn drop(&mut self) {
        unsafe {
            openjpeg_sys::opj_destroy_codec(self.codec.as_ptr());
        }
    }
}

impl Encoder {
    pub fn j2k() -> Result<Self, Jpeg2000Error> {
        let codec = NonNull::new(unsafe {
            openjpeg_sys::opj_create_compress(openjpeg_sys::OPJ_CODEC_FORMAT::OPJ_CODEC_J2K)
        })
        .ok_or(Jpeg2000Error::EncoderSetupError)?;

        unsafe {
            openjpeg_sys::opj_set_info_handler(codec.as_ptr(), Some(log), std::ptr::null_mut());
        }
        unsafe {
            openjpeg_sys::opj_set_warning_handler(codec.as_ptr(), Some(log), std::ptr::null_mut());
        }
        unsafe {
            openjpeg_sys::opj_set_error_handler(codec.as_ptr(), Some(log), std::ptr::null_mut());
        }

        Ok(Self { codec })
    }

    pub fn as_raw(&mut self) -> *mut openjpeg_sys::opj_codec_t {
        self.codec.as_ptr()
    }
}

extern "C" fn log(msg: *const std::ffi::c_char, _data: *mut std::ffi::c_void) {
    eprintln!(
        "{}",
        unsafe { std::ffi::CStr::from_ptr(msg) }.to_string_lossy()
    );
}
