//! Adapted from the MIT/Apache 2.0 licensed https://github.com/Neopallium/jpeg2k
//! and the MIT/Apache 2.0 licensed https://github.com/noritada/grib-rs

#![allow(unsafe_code)] // FFI

use std::mem::MaybeUninit;

use openjpeg_sys as opj;

mod codec;
mod image;
mod stream;

use codec::{Decoder, Encoder};
use image::Image;
use stream::{DecodeStream, EncodeStream};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Jpeg2000Error {
    NotSupported,
    DecoderSetupError,
    MainHeaderReadError,
    BodyReadError,
    EncoderSetupError,
    CompressError,
    DimensionsTooLarge,
    ImageCreateError,
}

pub fn decode(bytes: &[u8]) -> Result<(Vec<i32>, (usize, usize)), Jpeg2000Error> {
    let mut stream = DecodeStream::new(bytes);
    let mut decoder = Decoder::j2k()?;

    let mut decode_params = MaybeUninit::uninit();
    unsafe { openjpeg_sys::opj_set_default_decoder_parameters(decode_params.as_mut_ptr()) };
    let mut decode_params = unsafe { decode_params.assume_init() };

    if unsafe { openjpeg_sys::opj_setup_decoder(decoder.as_raw(), &mut decode_params) } != 1 {
        return Err(Jpeg2000Error::DecoderSetupError);
    }

    let mut image = Image::from_header(&mut stream, &mut decoder)?;

    if unsafe { opj::opj_decode(decoder.as_raw(), stream.as_raw(), image.as_raw()) } != 1 {
        return Err(Jpeg2000Error::BodyReadError);
    }

    drop(decoder);
    drop(stream);

    let width = image.width();
    let height = image.height();

    assert_eq!(image.factor(), 0);

    if let [comp_gray] = image.components() {
        let vec = unsafe {
            std::slice::from_raw_parts(comp_gray.data, (width * height) as usize).to_vec()
        };
        Ok((vec, (width as usize, height as usize)))
    } else {
        Err(Jpeg2000Error::NotSupported)
    }
}

pub fn encode_into(
    data: &[i32],
    width: usize,
    height: usize,
    psnr: i32,
    encoded: &mut Vec<u8>,
) -> Result<(), Jpeg2000Error> {
    let mut stream = EncodeStream::new(encoded);
    let mut encoder = Encoder::j2k()?;

    let mut encode_params = MaybeUninit::uninit();
    unsafe { openjpeg_sys::opj_set_default_encoder_parameters(encode_params.as_mut_ptr()) };
    let mut encode_params = unsafe { encode_params.assume_init() };
    encode_params.cp_fixed_quality = psnr as _;

    let (Ok(width), Ok(height)) = (u32::try_from(width), u32::try_from(height)) else {
        return Err(Jpeg2000Error::DimensionsTooLarge);
    };
    let mut image = Image::from_gray_data(data, width, height)?;

    if unsafe {
        openjpeg_sys::opj_setup_encoder(encoder.as_raw(), &mut encode_params, image.as_raw())
    } != 1
    {
        return Err(Jpeg2000Error::EncoderSetupError);
    }

    if unsafe {
        openjpeg_sys::opj_start_compress(encoder.as_raw(), image.as_raw(), stream.as_raw())
    } != 1
    {
        return Err(Jpeg2000Error::CompressError);
    }

    if unsafe { openjpeg_sys::opj_encode(encoder.as_raw(), stream.as_raw()) } != 1 {
        return Err(Jpeg2000Error::CompressError);
    }

    if unsafe { openjpeg_sys::opj_end_compress(encoder.as_raw(), stream.as_raw()) } != 1 {
        return Err(Jpeg2000Error::CompressError);
    }

    Ok(())
}
