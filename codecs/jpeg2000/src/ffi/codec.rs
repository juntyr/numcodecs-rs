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
            setup_codec_logging(codec);
        }

        Ok(Self { codec })
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
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
            setup_codec_logging(codec);
        }

        Ok(Self { codec })
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn as_raw(&mut self) -> *mut openjpeg_sys::opj_codec_t {
        self.codec.as_ptr()
    }
}

unsafe fn setup_codec_logging(codec: NonNull<openjpeg_sys::opj_codec_t>) {
    extern "C" fn log_info(msg: *const std::ffi::c_char, _data: *mut std::ffi::c_void) {
        unsafe { log(msg, log::Level::Info) }
    }

    extern "C" fn log_warning(msg: *const std::ffi::c_char, _data: *mut std::ffi::c_void) {
        unsafe { log(msg, log::Level::Warn) }
    }

    extern "C" fn log_error(msg: *const std::ffi::c_char, _data: *mut std::ffi::c_void) {
        unsafe { log(msg, log::Level::Error) }
    }

    unsafe fn log(msg: *const std::ffi::c_char, level: log::Level) {
        let msg = unsafe { std::ffi::CStr::from_ptr(msg) }.to_string_lossy();

        log::log!(level, "{msg}");
    }

    unsafe {
        openjpeg_sys::opj_set_info_handler(codec.as_ptr(), Some(log_info), std::ptr::null_mut());
    }
    unsafe {
        openjpeg_sys::opj_set_warning_handler(
            codec.as_ptr(),
            Some(log_warning),
            std::ptr::null_mut(),
        );
    }
    unsafe {
        openjpeg_sys::opj_set_error_handler(codec.as_ptr(), Some(log_error), std::ptr::null_mut());
    }
}
