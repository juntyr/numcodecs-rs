//! Adapted from
//! - the MIT/Apache 2.0 licensed <https://github.com/Neopallium/jpeg2k>, and
//! - the MIT/Apache 2.0 licensed <https://github.com/noritada/grib-rs>

#![allow(unsafe_code)] // FFI

use std::mem::MaybeUninit;

use openjpeg_sys as opj;

mod codec;
mod image;
mod stream;

use codec::{Decoder, Encoder};
use image::Image;
use stream::{DecodeStream, EncodeStream};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Jpeg2000Error {
    #[error("Jpeg2000 can only encode data with a width and height that each fit into a u32")]
    ImageTooLarge,
    #[error("Jpeg2000 failed to create an image from the data to encode")]
    ImageCreateError,
    #[error("Jpeg2000 failed to setup the encoder")]
    EncoderSetupError,
    #[error("Jpeg2000 failed to start compression")]
    StartCompressError,
    #[error("Jpeg2000 failed to compress the data body")]
    CompressBodyError,
    #[error("Jpeg2000 failed to end compression")]
    EndCompressError,
    #[error("Jpeg2000 failed to setup the decoder")]
    DecoderSetupError,
    #[error("Jpeg2000 failed to decode an invalid main header")]
    InvalidMainHeader,
    #[error("Jpeg2000 failed to decode the data body")]
    DecodeBodyError,
    #[error("Jpeg2000 failed to end decompression")]
    EndDecompressError,
    #[error("Jpeg2000 can only decode from single-channel gray images")]
    DecodeNonGrayData,
    #[error("Jpeg2000 can only decode from non-subsampled images")]
    DecodeSubsampledData,
    #[error("Jpeg2000 decoded into an image with an unexpected precision")]
    DecodedDataBitsMismatch,
    #[error("Jpeg2000 decoded into an image with an unexpected sign")]
    DecodedDataSignMismatch,
}

#[allow(clippy::upper_case_acronyms)]
pub enum Jpeg2000Compression {
    PSNR(f32),
    Rate(f32),
    Lossless,
}

#[allow(clippy::needless_pass_by_value)]
pub fn encode_into<T: Jpeg2000Element>(
    data: impl IntoIterator<Item = T>,
    width: usize,
    height: usize,
    compression: Jpeg2000Compression,
    out: &mut Vec<u8>,
) -> Result<(), Jpeg2000Error> {
    let mut stream = EncodeStream::new(out);
    let mut encoder = Encoder::j2k()?;

    let mut encode_params = MaybeUninit::zeroed();
    unsafe { openjpeg_sys::opj_set_default_encoder_parameters(encode_params.as_mut_ptr()) };
    let mut encode_params = unsafe { encode_params.assume_init() };

    encode_params.numresolution = 6;
    while (width < (1 << (encode_params.numresolution - 1)))
        || (height < (1 << (encode_params.numresolution - 1)))
    {
        encode_params.numresolution -= 1;
    }

    encode_params.tcp_numlayers = 1;

    match compression {
        Jpeg2000Compression::PSNR(psnr) => {
            encode_params.cp_fixed_quality = 1;
            encode_params.tcp_distoratio[0] = psnr;
        }
        Jpeg2000Compression::Rate(rate) => {
            encode_params.cp_disto_alloc = 1;
            encode_params.tcp_rates[0] = rate;
        }
        Jpeg2000Compression::Lossless => {
            encode_params.cp_disto_alloc = 1;
            encode_params.tcp_rates[0] = 0.0;
        }
    }

    let (Ok(width), Ok(height)) = (u32::try_from(width), u32::try_from(height)) else {
        return Err(Jpeg2000Error::ImageTooLarge);
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
        return Err(Jpeg2000Error::StartCompressError);
    }

    if unsafe { openjpeg_sys::opj_encode(encoder.as_raw(), stream.as_raw()) } != 1 {
        return Err(Jpeg2000Error::CompressBodyError);
    }

    if unsafe { openjpeg_sys::opj_end_compress(encoder.as_raw(), stream.as_raw()) } != 1 {
        return Err(Jpeg2000Error::EndCompressError);
    }

    Ok(())
}

pub fn decode<T: Jpeg2000Element>(bytes: &[u8]) -> Result<(Vec<T>, (usize, usize)), Jpeg2000Error> {
    let mut stream = DecodeStream::new(bytes);
    let mut decoder = Decoder::j2k()?;

    let mut decode_params = MaybeUninit::zeroed();
    unsafe { openjpeg_sys::opj_set_default_decoder_parameters(decode_params.as_mut_ptr()) };
    let mut decode_params = unsafe { decode_params.assume_init() };
    decode_params.decod_format = 1; // JP2

    if unsafe { openjpeg_sys::opj_setup_decoder(decoder.as_raw(), &mut decode_params) } != 1 {
        return Err(Jpeg2000Error::DecoderSetupError);
    }

    let mut image = Image::from_header(&mut stream, &mut decoder)?;

    if unsafe { opj::opj_decode(decoder.as_raw(), stream.as_raw(), image.as_raw()) } != 1 {
        return Err(Jpeg2000Error::DecodeBodyError);
    }

    if unsafe { openjpeg_sys::opj_end_decompress(decoder.as_raw(), stream.as_raw()) } != 1 {
        return Err(Jpeg2000Error::EndDecompressError);
    }

    drop(decoder);
    drop(stream);

    let width = image.width() as usize;
    let height = image.height() as usize;

    let [gray] = image.components() else {
        return Err(Jpeg2000Error::DecodeNonGrayData);
    };

    if gray.factor != 0 {
        return Err(Jpeg2000Error::DecodeSubsampledData);
    }

    if gray.prec != T::NBITS {
        return Err(Jpeg2000Error::DecodedDataBitsMismatch);
    }

    if (gray.sgnd != 0) != T::SIGNED {
        return Err(Jpeg2000Error::DecodedDataSignMismatch);
    }

    let data = unsafe { std::slice::from_raw_parts(gray.data, width * height) };
    let data = data.iter().copied().map(T::from_i32).collect();

    Ok((data, (width, height)))
}

pub trait Jpeg2000Element: Copy {
    const NBITS: u32;
    const SIGNED: bool;

    fn into_i32(self) -> i32;
    fn from_i32(x: i32) -> Self;
}

impl Jpeg2000Element for i8 {
    const NBITS: u32 = 8;
    const SIGNED: bool = true;

    fn into_i32(self) -> i32 {
        i32::from(self)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_i32(x: i32) -> Self {
        x as Self
    }
}

impl Jpeg2000Element for u8 {
    const NBITS: u32 = 8;
    const SIGNED: bool = false;

    fn into_i32(self) -> i32 {
        i32::from(self)
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn from_i32(x: i32) -> Self {
        x as Self
    }
}

impl Jpeg2000Element for i16 {
    const NBITS: u32 = 16;
    const SIGNED: bool = true;

    fn into_i32(self) -> i32 {
        i32::from(self)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_i32(x: i32) -> Self {
        x as Self
    }
}

impl Jpeg2000Element for u16 {
    const NBITS: u32 = 16;
    const SIGNED: bool = false;

    fn into_i32(self) -> i32 {
        i32::from(self)
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn from_i32(x: i32) -> Self {
        x as Self
    }
}
