//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.82.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-bit-round
//! [crates.io]: https://crates.io/crates/numcodecs-bit-round
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-bit-round
//! [docs.rs]: https://docs.rs/numcodecs-bit-round/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_bit_round
//!
//! Bit rounding codec implementation for the [`numcodecs`] API.

use ndarray::{Array, ArrayBase, Data, Dimension};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
/// Codec providing floating-point bit rounding.
///
/// Drops the specified number of bits from the floating point mantissa,
/// leaving an array that is more amenable to compression. The number of
/// bits to keep should be determined by information analysis of the data
/// to be compressed.
///
/// The approach is based on the paper by Klöwer et al. 2021
/// (<https://www.nature.com/articles/s43588-021-00156-2>).
pub struct BitRoundCodec {
    /// The number of bits of the mantissa to keep.
    ///
    /// The valid range depends on the dtype of the input data.
    ///
    /// If keepbits is equal to the bitlength of the dtype's mantissa, no
    /// transformation is performed.
    pub keepbits: u8,
}

impl Codec for BitRoundCodec {
    type Error = BitRoundCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            AnyCowArray::F32(data) => Ok(AnyArray::F32(bit_round(data, self.keepbits)?)),
            AnyCowArray::F64(data) => Ok(AnyArray::F64(bit_round(data, self.keepbits)?)),
            encoded => Err(BitRoundCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match encoded {
            AnyCowArray::F32(encoded) => Ok(AnyArray::F32(encoded.into_owned())),
            AnyCowArray::F64(encoded) => Ok(AnyArray::F64(encoded.into_owned())),
            encoded => Err(BitRoundCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        mut decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        if !matches!(encoded.dtype(), AnyArrayDType::F32 | AnyArrayDType::F64) {
            return Err(BitRoundCodecError::UnsupportedDtype(encoded.dtype()));
        }

        Ok(decoded.assign(&encoded)?)
    }
}

impl StaticCodec for BitRoundCodec {
    const CODEC_ID: &'static str = "bit-round";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<Self> {
        StaticCodecConfig::from(self)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`BitRoundCodec`].
pub enum BitRoundCodecError {
    /// [`BitRoundCodec`] does not support the dtype
    #[error("BitRound does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`BitRoundCodec`] encode `keepbits` exceed the mantissa size for `dtype`
    #[error("BitRound encode {keepbits} bits exceed the mantissa size for {dtype}")]
    ExcessiveKeepBits {
        /// The number of bits of the mantissa to keep
        keepbits: u8,
        /// The `dtype` of the data to encode
        dtype: AnyArrayDType,
    },
    /// [`BitRoundCodec`] cannot decode into the provided array
    #[error("BitRound cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

/// Floating-point bit rounding, which drops the specified number of bits from
/// the floating point mantissa.
///
/// See <https://github.com/milankl/BitInformation.jl> for the the original
/// implementation in Julia.
///
/// # Errors
///
/// Errors with [`BitRoundCodecError::ExcessiveKeepBits`] if `keepbits` exceeds
/// [`T::MANITSSA_BITS`][`Float::MANITSSA_BITS`].
pub fn bit_round<T: Float, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
    keepbits: u8,
) -> Result<Array<T, D>, BitRoundCodecError> {
    if u32::from(keepbits) > T::MANITSSA_BITS {
        return Err(BitRoundCodecError::ExcessiveKeepBits {
            keepbits,
            dtype: T::TY,
        });
    }

    let mut encoded = data.into_owned();

    // Early return if no bit rounding needs to happen
    // - required since the ties to even impl does not work in this case
    if u32::from(keepbits) == T::MANITSSA_BITS {
        return Ok(encoded);
    }

    // half of unit in last place (ulp)
    let ulp_half = T::MANTISSA_MASK >> (u32::from(keepbits) + 1);
    // mask to zero out trailing mantissa bits
    let keep_mask = !(T::MANTISSA_MASK >> u32::from(keepbits));
    // shift to extract the least significant bit of the exponent
    let shift = T::MANITSSA_BITS - u32::from(keepbits);

    encoded.mapv_inplace(|x| {
        let mut bits = T::to_binary(x);

        // add ulp/2 with ties to even
        bits += ulp_half + ((bits >> shift) & T::BINARY_ONE);

        // set the trailing bits to zero
        bits &= keep_mask;

        T::from_binary(bits)
    });

    Ok(encoded)
}

/// Floating point types.
pub trait Float: Sized + Copy {
    /// Number of significant digits in base 2
    const MANITSSA_BITS: u32;
    /// Binary mask to extract only the mantissa bits
    const MANTISSA_MASK: Self::Binary;
    /// Binary `0x1`
    const BINARY_ONE: Self::Binary;

    /// Dtype of this type
    const TY: AnyArrayDType;

    /// Binary representation of this type
    type Binary: Copy
        + std::ops::Not<Output = Self::Binary>
        + std::ops::Shr<u32, Output = Self::Binary>
        + std::ops::Add<Self::Binary, Output = Self::Binary>
        + std::ops::AddAssign<Self::Binary>
        + std::ops::BitAnd<Self::Binary, Output = Self::Binary>
        + std::ops::BitAndAssign<Self::Binary>;

    /// Bit-cast the floating point value to its binary representation
    fn to_binary(self) -> Self::Binary;
    /// Bit-cast the binary representation into a floating point value
    fn from_binary(u: Self::Binary) -> Self;
}

impl Float for f32 {
    type Binary = u32;

    const BINARY_ONE: Self::Binary = 1;
    const MANITSSA_BITS: u32 = Self::MANTISSA_DIGITS - 1;
    const MANTISSA_MASK: Self::Binary = (1 << Self::MANITSSA_BITS) - 1;
    const TY: AnyArrayDType = AnyArrayDType::F32;

    fn to_binary(self) -> Self::Binary {
        self.to_bits()
    }

    fn from_binary(u: Self::Binary) -> Self {
        Self::from_bits(u)
    }
}

impl Float for f64 {
    type Binary = u64;

    const BINARY_ONE: Self::Binary = 1;
    const MANITSSA_BITS: u32 = Self::MANTISSA_DIGITS - 1;
    const MANTISSA_MASK: Self::Binary = (1 << Self::MANITSSA_BITS) - 1;
    const TY: AnyArrayDType = AnyArrayDType::F64;

    fn to_binary(self) -> Self::Binary {
        self.to_bits()
    }

    fn from_binary(u: Self::Binary) -> Self {
        Self::from_bits(u)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use ndarray::{Array1, ArrayView1};

    use super::*;

    #[test]
    fn no_mantissa() {
        assert_eq!(
            bit_round(ArrayView1::from(&[0.0_f32]), 0).unwrap(),
            Array1::from_vec(vec![0.0_f32])
        );
        assert_eq!(
            bit_round(ArrayView1::from(&[1.0_f32]), 0).unwrap(),
            Array1::from_vec(vec![1.0_f32])
        );
        // tie to even rounds up as the offset exponent is odd
        assert_eq!(
            bit_round(ArrayView1::from(&[1.5_f32]), 0).unwrap(),
            Array1::from_vec(vec![2.0_f32])
        );
        assert_eq!(
            bit_round(ArrayView1::from(&[2.0_f32]), 0).unwrap(),
            Array1::from_vec(vec![2.0_f32])
        );
        assert_eq!(
            bit_round(ArrayView1::from(&[2.5_f32]), 0).unwrap(),
            Array1::from_vec(vec![2.0_f32])
        );
        // tie to even rounds down as the offset exponent is even
        assert_eq!(
            bit_round(ArrayView1::from(&[3.0_f32]), 0).unwrap(),
            Array1::from_vec(vec![2.0_f32])
        );
        assert_eq!(
            bit_round(ArrayView1::from(&[3.5_f32]), 0).unwrap(),
            Array1::from_vec(vec![4.0_f32])
        );
        assert_eq!(
            bit_round(ArrayView1::from(&[4.0_f32]), 0).unwrap(),
            Array1::from_vec(vec![4.0_f32])
        );
        assert_eq!(
            bit_round(ArrayView1::from(&[5.0_f32]), 0).unwrap(),
            Array1::from_vec(vec![4.0_f32])
        );
        // tie to even rounds up as the offset exponent is odd
        assert_eq!(
            bit_round(ArrayView1::from(&[6.0_f32]), 0).unwrap(),
            Array1::from_vec(vec![8.0_f32])
        );
        assert_eq!(
            bit_round(ArrayView1::from(&[7.0_f32]), 0).unwrap(),
            Array1::from_vec(vec![8.0_f32])
        );
        assert_eq!(
            bit_round(ArrayView1::from(&[8.0_f32]), 0).unwrap(),
            Array1::from_vec(vec![8.0_f32])
        );

        assert_eq!(
            bit_round(ArrayView1::from(&[0.0_f64]), 0).unwrap(),
            Array1::from_vec(vec![0.0_f64])
        );
        assert_eq!(
            bit_round(ArrayView1::from(&[1.0_f64]), 0).unwrap(),
            Array1::from_vec(vec![1.0_f64])
        );
        // tie to even rounds up as the offset exponent is odd
        assert_eq!(
            bit_round(ArrayView1::from(&[1.5_f64]), 0).unwrap(),
            Array1::from_vec(vec![2.0_f64])
        );
        assert_eq!(
            bit_round(ArrayView1::from(&[2.0_f64]), 0).unwrap(),
            Array1::from_vec(vec![2.0_f64])
        );
        assert_eq!(
            bit_round(ArrayView1::from(&[2.5_f64]), 0).unwrap(),
            Array1::from_vec(vec![2.0_f64])
        );
        // tie to even rounds down as the offset exponent is even
        assert_eq!(
            bit_round(ArrayView1::from(&[3.0_f64]), 0).unwrap(),
            Array1::from_vec(vec![2.0_f64])
        );
        assert_eq!(
            bit_round(ArrayView1::from(&[3.5_f64]), 0).unwrap(),
            Array1::from_vec(vec![4.0_f64])
        );
        assert_eq!(
            bit_round(ArrayView1::from(&[4.0_f64]), 0).unwrap(),
            Array1::from_vec(vec![4.0_f64])
        );
        assert_eq!(
            bit_round(ArrayView1::from(&[5.0_f64]), 0).unwrap(),
            Array1::from_vec(vec![4.0_f64])
        );
        // tie to even rounds up as the offset exponent is odd
        assert_eq!(
            bit_round(ArrayView1::from(&[6.0_f64]), 0).unwrap(),
            Array1::from_vec(vec![8.0_f64])
        );
        assert_eq!(
            bit_round(ArrayView1::from(&[7.0_f64]), 0).unwrap(),
            Array1::from_vec(vec![8.0_f64])
        );
        assert_eq!(
            bit_round(ArrayView1::from(&[8.0_f64]), 0).unwrap(),
            Array1::from_vec(vec![8.0_f64])
        );
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn full_mantissa() {
        fn full<T: Float>(x: T) -> T {
            T::from_binary(T::to_binary(x) + T::MANTISSA_MASK)
        }

        for v in [0.0_f32, 1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32] {
            assert_eq!(
                bit_round(ArrayView1::from(&[full(v)]), f32::MANITSSA_BITS as u8).unwrap(),
                Array1::from_vec(vec![full(v)])
            );
        }

        for v in [0.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64] {
            assert_eq!(
                bit_round(ArrayView1::from(&[full(v)]), f64::MANITSSA_BITS as u8).unwrap(),
                Array1::from_vec(vec![full(v)])
            );
        }
    }
}
