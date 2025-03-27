//! Adapted from
//! - the MIT/Apache 2.0 licensed <https://github.com/kardeiz/jp2k>, and
//! - the MIT/Apache 2.0 licensed <https://github.com/noritada/grib-rs>

use std::ptr::NonNull;

use super::{codec::Decoder, stream::DecodeStream, Jpeg2000Element, Jpeg2000Error};

pub struct Image {
    image: NonNull<openjpeg_sys::opj_image_t>,
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            openjpeg_sys::opj_image_destroy(self.image.as_ptr());
        }
    }
}

impl Image {
    pub fn from_header(
        stream: &mut DecodeStream,
        decoder: &mut Decoder,
    ) -> Result<Self, Jpeg2000Error> {
        let mut image = std::ptr::null_mut();

        if unsafe { openjpeg_sys::opj_read_header(stream.as_raw(), decoder.as_raw(), &mut image) }
            != 1
        {
            return Err(Jpeg2000Error::InvalidMainHeader);
        }

        let image = NonNull::new(image).ok_or(Jpeg2000Error::InvalidMainHeader)?;

        Ok(Self { image })
    }

    pub fn from_gray_data<T: Jpeg2000Element>(
        data: impl IntoIterator<Item = T>,
        width: u32,
        height: u32,
    ) -> Result<Self, Jpeg2000Error> {
        let mut image_params = openjpeg_sys::opj_image_cmptparm_t {
            dx: 1,
            dy: 1,
            w: width,
            h: height,
            x0: 0,
            y0: 0,
            prec: T::NBITS,
            bpp: T::NBITS,
            sgnd: u32::from(T::SIGNED),
        };

        let image = NonNull::new(unsafe {
            openjpeg_sys::opj_image_create(
                1,
                &mut image_params,
                openjpeg_sys::OPJ_COLOR_SPACE::OPJ_CLRSPC_GRAY,
            )
        })
        .ok_or(Jpeg2000Error::ImageCreateError)?;

        unsafe {
            (*image.as_ptr()).x0 = 0;
            (*image.as_ptr()).y0 = 0;
            (*image.as_ptr()).x1 = width;
            (*image.as_ptr()).y1 = height;
        }

        unsafe {
            let mut idata = (*(*image.as_ptr()).comps).data;
            for d in data {
                *idata = d.into_i32().map_err(|_| Jpeg2000Error::DataOutOfRange)?;
                idata = idata.add(1);
            }
        }

        Ok(Self { image })
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn as_raw(&mut self) -> *mut openjpeg_sys::opj_image_t {
        self.image.as_ptr()
    }

    pub fn width(&self) -> u32 {
        unsafe { (*self.image.as_ptr()).x1 - (*self.image.as_ptr()).x0 }
    }

    pub fn height(&self) -> u32 {
        unsafe { (*self.image.as_ptr()).y1 - (*self.image.as_ptr()).y0 }
    }

    pub fn components(&self) -> &[openjpeg_sys::opj_image_comp_t] {
        let comps_len = unsafe { (*self.image.as_ptr()).numcomps };
        unsafe { std::slice::from_raw_parts((*self.image.as_ptr()).comps, comps_len as usize) }
    }
}
