//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.64.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-zlib
//! [crates.io]: https://crates.io/crates/numcodecs-zlib
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-zlib
//! [docs.rs]: https://docs.rs/numcodecs-zlib/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs-zlib
//!
//! Zlib codec implementation for the [`numcodecs`] API.

use std::borrow::Cow;

use ndarray::Array1;
use numcodecs::{
    AnyArray, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray, Codec, StaticCodec,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

#[derive(Clone, Serialize, Deserialize)]
/// Codec providing compression using Zlib
pub struct ZlibCodec {
    /// Compression level
    pub level: ZlibLevel,
}

#[derive(Copy, Clone, serde_repr::Serialize_repr, serde_repr::Deserialize_repr)]
#[repr(u8)]
/// Zlib compression level.
/// 
/// The level ranges from 0, no compression, to 9, best compression.
#[allow(missing_docs)]
pub enum ZlibLevel {
    ZNoCompression = 0,
    ZBestSpeed = 1,
    ZLevel2 = 2,
    ZLevel3 = 3,
    ZLevel4 = 4,
    ZLevel5 = 5,
    ZLevel6 = 6,
    ZLevel7 = 7,
    ZLevel8 = 8,
    ZBestCompression = 9,
}

impl Codec for ZlibCodec {
    type Error = ZlibCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        compress(data, self.level).map(|bytes| AnyArray::U8(Array1::from_vec(bytes).into_dyn()))
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        let AnyCowArray::U8(encoded) = encoded else {
            return Err(ZlibCodecError::EncodedDataNotBytes { dtype: encoded.dtype() });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(ZlibCodecError::EncodedDataNotOneDimensional { shape: encoded.shape().to_vec() });
        }

        decompress(&AnyCowArray::U8(encoded).as_bytes())
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        let AnyArrayView::U8(encoded) = encoded else {
            return Err(ZlibCodecError::EncodedDataNotBytes { dtype: encoded.dtype() });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(ZlibCodecError::EncodedDataNotOneDimensional { shape: encoded.shape().to_vec() });
        }

        decompress_into(&AnyArrayView::U8(encoded).as_bytes(), decoded)
    }

    fn get_config<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.serialize(serializer)
    }
}

impl StaticCodec for ZlibCodec {
    const CODEC_ID: &'static str = "zlib";

    fn from_config<'de, D: Deserializer<'de>>(config: D) -> Result<Self, D::Error> {
        Self::deserialize(config)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`ZlibCodec`].
pub enum ZlibCodecError {
    /// [`ZlibCodec`] failed to encode the header
    #[error("Zlib failed to encode the header")]
    HeaderEncodeFailed {
        /// Opaque source error 
        source: ZlibHeaderError,
    },
    /// [`ZlibCodec`] can only decode one-dimensional byte arrays but received
    /// an array of a different dtype
    #[error("Zlib can only decode one-dimensional byte arrays but received an array of dtype {dtype}")]
    EncodedDataNotBytes {
        /// The unexpected dtype of the encoded array
        dtype: AnyArrayDType,
    },
    /// [`ZlibCodec`] can only decode one-dimensional byte arrays but received
    /// an array of a different shape
    #[error("Zlib can only decode one-dimensional byte arrays but received a byte array of shape {shape:?}")]
    EncodedDataNotOneDimensional {
        /// The unexpected shape of the encoded array
        shape: Vec<usize>,
    },
    /// [`ZlibCodec`] failed to encode the header
    #[error("Zlib failed to decode the header")]
    HeaderDecodeFailed {
        /// Opaque source error 
        source: ZlibHeaderError,
    },
    /// [`ZlibCodec`] cannot decode the `decoded` dtype into the `provided`
    /// array
    #[error("Zlib cannot decode the dtype {decoded} into the provided {provided} array")]
    MismatchedDecodeIntoDtype {
        /// Dtype of the `decoded` data
        decoded: AnyArrayDType,
        /// Dtype of the `provided` array into which the data is to be decoded
        provided: AnyArrayDType,
    },
    /// [`ZlibCodec`] cannot decode the decoded array into the provided
    /// array of a different shape
    #[error("Zlib cannot decode the decoded array of shape {decoded:?} into the provided array of shape {provided:?}")]
    MismatchedDecodeIntoShape {
        /// Shape of the `decoded` data
        decoded: Vec<usize>,
        /// Shape of the `provided` array into which the data is to be decoded
        provided: Vec<usize>,
    },
}

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding the header fails
pub struct ZlibHeaderError(postcard::Error);

/// Compress the `array` using Zlib with the provided `level`.
/// 
/// # Errors
/// 
/// Errors with [`ZlibCodecError::HeaderEncodeFailed`] if encoding the header
/// to the output bytevec failed.
pub fn compress(
    array: AnyCowArray,
    level: ZlibLevel,
) -> Result<Vec<u8>, ZlibCodecError> {
    let data = array.as_bytes();

    let mut encoded = postcard::to_extend(
        &CompressionHeader {
            dtype: array.dtype(),
            shape: Cow::Borrowed(array.shape()),
        },
        Vec::new(),
    )
    .map_err(|err| ZlibCodecError::HeaderEncodeFailed { source: ZlibHeaderError(err) })?;

    let mut in_pos = 0;
    let mut out_pos = encoded.len();

    // The comp flags function sets the zlib flag if the window_bits parameter
    //  is > 0.
    let flags =
        miniz_oxide::deflate::core::create_comp_flags_from_zip_params((level as u8).into(), 1, 0);
    let mut compressor = miniz_oxide::deflate::core::CompressorOxide::new(flags);
    encoded.resize(encoded.len() + (data.len() / 2).max(2), 0);

    loop {
        let (Some(data_left), Some(encoded_left)) =
            (data.get(in_pos..), encoded.get_mut(out_pos..))
        else {
            panic!("Zlib encode bug: input or output is out of bounds");
        };

        let (status, bytes_in, bytes_out) = miniz_oxide::deflate::core::compress(
            &mut compressor,
            data_left,
            encoded_left,
            miniz_oxide::deflate::core::TDEFLFlush::Finish,
        );

        out_pos += bytes_out;
        in_pos += bytes_in;

        match status {
            miniz_oxide::deflate::core::TDEFLStatus::Okay => {
                // We need more space, so resize the vector.
                if encoded.len().saturating_sub(out_pos) < 30 {
                    encoded.resize(encoded.len() * 2, 0);
                }
            },
            miniz_oxide::deflate::core::TDEFLStatus::Done => {
                encoded.truncate(out_pos);

                if in_pos != data.len() {
                    panic!("Zlib encode bug: consumed less input than expected");
                }

                return Ok(encoded);
            },
            err => panic!("Zlib encode bug: {err:?}"),
        }
    }
}

pub fn decompress(data: &[u8]) -> Result<AnyArray, ZlibCodecError> {
    let (header, data) = postcard::take_from_bytes::<CompressionHeader>(data)
        .map_err(|err| ZlibCodecError::HeaderDecodeFailed { source: ZlibHeaderError(err) })?;

    let (decoded, result) = AnyArray::with_zeros_bytes(header.dtype, &header.shape, |decoded_bytes| {
        let flags = miniz_oxide::inflate::core::inflate_flags::TINFL_FLAG_PARSE_ZLIB_HEADER
            | miniz_oxide::inflate::core::inflate_flags::TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF;

        let mut decomp = Box::<miniz_oxide::inflate::core::DecompressorOxide>::default();

        let (status, in_consumed, out_consumed) =
            miniz_oxide::inflate::core::decompress(&mut decomp, data, decoded_bytes, 0, flags);

        match status {
            miniz_oxide::inflate::TINFLStatus::Done => {
                if in_consumed != data.len() {
                    Err(String::from(
                        "Zlib::decode consumed less data than expected",
                    ))
                } else if out_consumed == decoded_bytes.len() {
                    Ok(())
                } else {
                    Err(String::from(
                        "Zlib::decode produced more data than expected",
                    ))
                }
            },
            miniz_oxide::inflate::TINFLStatus::HasMoreOutput => Err(String::from(
                "Zlib::decode produced less data than expected",
            )),
            _ => Err(format!(
                "{}",
                miniz_oxide::inflate::DecompressError {
                    status,
                    output: Vec::new()
                }
            )),
        }
    });

    result.map(|()| decoded)
}

pub fn decompress_into(data: &[u8], decoded: AnyArrayViewMut) -> Result<(), ZlibCodecError> {
    let (header, data) = postcard::take_from_bytes::<CompressionHeader>(data)
        .map_err(|err| ZlibCodecError::HeaderDecodeFailed { source: ZlibHeaderError(err) })?;

    if header.dtype != decoded.dtype() {
        return Err(ZlibCodecError::MismatchedDecodeIntoDtype { decoded: header.dtype, provided: decoded.dtype() });
    }

    if header.shape != decoded.shape() {
        return Err(ZlibCodecError::MismatchedDecodeIntoShape { decoded: header.shape.into_owned(), provided: decoded.shape().to_vec() });
    }

    if let Some(decoded_bytes) = decoded.as_bytes_mut() {
        let flags = miniz_oxide::inflate::core::inflate_flags::TINFL_FLAG_PARSE_ZLIB_HEADER
            | miniz_oxide::inflate::core::inflate_flags::TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF;

        let mut decomp = Box::<miniz_oxide::inflate::core::DecompressorOxide>::default();

        let (status, in_consumed, out_consumed) =
            miniz_oxide::inflate::core::decompress(&mut decomp, data, decoded_bytes, 0, flags);

        match status {
            miniz_oxide::inflate::TINFLStatus::Done => {
                if in_consumed != data.len() {
                    Err(String::from(
                        "Zlib::decode consumed less data than expected",
                    ))
                } else if out_consumed == decoded_bytes.len() {
                    Ok(())
                } else {
                    Err(String::from(
                        "Zlib::decode produced more data than expected",
                    ))
                }
            },
            miniz_oxide::inflate::TINFLStatus::HasMoreOutput => Err(String::from(
                "Zlib::decode produced less data than expected",
            )),
            _ => Err(format!(
                "{}",
                miniz_oxide::inflate::DecompressError {
                    status,
                    output: Vec::new()
                }
            )),
        }
    }

    todo!()
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CompressionHeader<'a> {
    dtype: AnyArrayDType,
    #[serde(borrow)]
    shape: std::borrow::Cow<'a, [usize]>,
}
