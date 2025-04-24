//! Adapted from the MIT/Apache 2.0 licensed <https://github.com/Neopallium/jpeg2k>

use std::ffi::c_void;

pub struct DecodeStream<'a> {
    stream: *mut openjpeg_sys::opj_stream_t,
    _buf: &'a [u8],
}

impl Drop for DecodeStream<'_> {
    fn drop(&mut self) {
        unsafe {
            openjpeg_sys::opj_stream_destroy(self.stream);
        }
    }
}

impl<'a> DecodeStream<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        let len = buf.len();
        let data = Box::into_raw(Box::new(WrappedSlice::new(buf)));

        let stream = unsafe {
            let stream = openjpeg_sys::opj_stream_default_create(1);
            openjpeg_sys::opj_stream_set_read_function(stream, Some(buf_read_stream_read_fn));
            openjpeg_sys::opj_stream_set_skip_function(stream, Some(buf_read_stream_skip_fn));
            openjpeg_sys::opj_stream_set_seek_function(stream, Some(buf_read_stream_seek_fn));
            openjpeg_sys::opj_stream_set_user_data_length(stream, len as u64);
            openjpeg_sys::opj_stream_set_user_data(
                stream,
                data.cast(),
                Some(buf_read_stream_free_fn),
            );
            stream
        };

        Self { stream, _buf: buf }
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub const fn as_raw(&mut self) -> *mut openjpeg_sys::opj_stream_t {
        self.stream
    }
}

extern "C" fn buf_read_stream_free_fn(p_data: *mut c_void) {
    let ptr: *mut WrappedSlice = p_data.cast();
    drop(unsafe { Box::from_raw(ptr) });
}

extern "C" fn buf_read_stream_read_fn(
    p_buffer: *mut c_void,
    nb_bytes: usize,
    p_data: *mut c_void,
) -> usize {
    if p_buffer.is_null() || nb_bytes == 0 {
        return usize::MAX;
    }

    let slice: &mut WrappedSlice = unsafe { &mut *p_data.cast() };
    slice
        .read_into(p_buffer.cast(), nb_bytes)
        .unwrap_or(usize::MAX)
}

extern "C" fn buf_read_stream_skip_fn(nb_bytes: i64, p_data: *mut c_void) -> i64 {
    let slice: &mut WrappedSlice = unsafe { &mut *p_data.cast() };
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    {
        slice.consume(nb_bytes as usize) as i64
    }
}

extern "C" fn buf_read_stream_seek_fn(nb_bytes: i64, p_data: *mut c_void) -> i32 {
    let slice: &mut WrappedSlice = unsafe { &mut *p_data.cast() };
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let seek_offset = nb_bytes as usize;
    let new_offset = slice.seek(seek_offset);

    // Return true if the seek worked.
    i32::from(seek_offset == new_offset)
}

struct WrappedSlice<'a> {
    offset: usize,
    buf: &'a [u8],
}

impl<'a> WrappedSlice<'a> {
    const fn new(buf: &'a [u8]) -> Self {
        Self { offset: 0, buf }
    }

    const fn remaining(&self) -> usize {
        self.buf.len() - self.offset
    }

    fn seek(&mut self, new_offset: usize) -> usize {
        // Make sure `new_offset <= buf.len()`
        self.offset = std::cmp::min(self.buf.len(), new_offset);
        self.offset
    }

    fn consume(&mut self, n_bytes: usize) -> usize {
        let offset = self.offset.saturating_add(n_bytes);
        // Make sure `offset <= buf.len()`
        self.offset = self.buf.len().min(offset);
        self.offset
    }

    fn read_into(&mut self, out: *mut u8, len: usize) -> Option<usize> {
        // Get the number of remaining bytes
        let remaining = self.remaining();
        if remaining == 0 {
            // No more bytes.
            return None;
        }

        // Try to fill the output buffer
        let n_read = std::cmp::min(remaining, len);
        let offset = self.offset;
        self.consume(n_read);

        unsafe {
            std::ptr::copy_nonoverlapping(self.buf.as_ptr().add(offset), out, n_read);
        }

        Some(n_read)
    }
}

pub struct EncodeStream<'a> {
    stream: *mut openjpeg_sys::opj_stream_t,
    _buf: &'a mut Vec<u8>,
}

impl<'a> EncodeStream<'a> {
    pub fn new(buf: &'a mut Vec<u8>) -> Self {
        let stream = unsafe {
            let stream = openjpeg_sys::opj_stream_default_create(0);
            openjpeg_sys::opj_stream_set_write_function(stream, Some(vec_write_stream_write_fn));
            openjpeg_sys::opj_stream_set_user_data(stream, std::ptr::from_mut(buf).cast(), None);
            stream
        };

        Self { stream, _buf: buf }
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub const fn as_raw(&mut self) -> *mut openjpeg_sys::opj_stream_t {
        self.stream
    }
}

extern "C" fn vec_write_stream_write_fn(
    p_buffer: *mut c_void,
    nb_bytes: usize,
    p_data: *mut c_void,
) -> usize {
    if p_buffer.is_null() {
        return usize::MAX;
    }

    let vec: &mut Vec<u8> = unsafe { &mut *p_data.cast() };

    let data = unsafe { std::slice::from_raw_parts(p_buffer.cast(), nb_bytes) };

    vec.extend_from_slice(data);

    nb_bytes
}
