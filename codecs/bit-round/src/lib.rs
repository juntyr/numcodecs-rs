//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.88.0-blue
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

use std::borrow::Cow;

use ndarray::{Array, ArrayBase, Data, Dimension};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig, StaticCodecVersion,
};
use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
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
    /// Bit rounding mode.
    #[serde(flatten)]
    pub mode: BitRoundMode,
    /// The codec's encoding format version. Do not provide this parameter explicitly.
    #[serde(default, rename = "_version")]
    pub version: StaticCodecVersion<2, 0, 0>,
}

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "mode")]
#[serde(deny_unknown_fields)]
/// Bit rounding mode
pub enum BitRoundMode {
    /// Directly specify the number of bits of the mantissa to keep.
    #[serde(rename = "keepbits")]
    Keepbits {
        /// The number of bits of the mantissa to keep.
        ///
        /// The valid range depends on the dtype of the input data.
        ///
        /// If keepbits is equal to the bitlength of the dtype's mantissa, no
        /// transformation is performed.
        keepbits: u8,
    },
    /// Pointwise absolute error.
    #[serde(rename = "abs")]
    AbsoluteError {
        /// The pointwise absolute error bound to preserve.
        ///
        /// This error bound guarantees that
        /// `$|x - \hat{x}| \leq \epsilon_{abs}$`.
        eb_abs: NonNegative<f64>,
    },
    /// Pointwise relative error.
    #[serde(rename = "rel")]
    RelativeError {
        /// The pointwise relative error bound to preserve.
        ///
        /// This error bound guarantees that
        /// `$|x - \hat{x}| \leq |x| \cdot \epsilon_{rel}$`.
        eb_rel: NonNegative<f64>,
    },
}

impl Codec for BitRoundCodec {
    type Error = BitRoundCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            AnyCowArray::F32(data) => Ok(AnyArray::F32(bit_round(data, &self.mode)?)),
            AnyCowArray::F64(data) => Ok(AnyArray::F64(bit_round(data, &self.mode)?)),
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
    const CODEC_ID: &'static str = "bit-round.rs";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<'_, Self> {
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

#[expect(clippy::derive_partial_eq_without_eq)] // floats are not Eq
#[derive(Copy, Clone, PartialEq, PartialOrd, Hash)]
/// Non-negative floating point number
pub struct NonNegative<T: Float>(T);

impl Serialize for NonNegative<f64> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(self.0)
    }
}

impl<'de> Deserialize<'de> for NonNegative<f64> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let x = f64::deserialize(deserializer)?;

        if x >= 0.0 {
            Ok(Self(x))
        } else {
            Err(serde::de::Error::invalid_value(
                serde::de::Unexpected::Float(x),
                &"a non-negative value",
            ))
        }
    }
}

impl JsonSchema for NonNegative<f64> {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> Cow<'static, str> {
        Cow::Borrowed("NonNegativeF64")
    }

    fn schema_id() -> Cow<'static, str> {
        Cow::Borrowed(concat!(module_path!(), "::", "NonNegative<f64>"))
    }

    fn json_schema(_gen: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "type": "number",
            "minimum": 0.0
        })
    }
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
    mode: &BitRoundMode,
) -> Result<Array<T, D>, BitRoundCodecError> {
    let (keepbits, keep_non_normal) = match mode {
        BitRoundMode::Keepbits { keepbits } => {
            let keepbits = *keepbits;
            if u32::from(keepbits) > T::MANITSSA_BITS {
                return Err(BitRoundCodecError::ExcessiveKeepBits {
                    keepbits,
                    dtype: T::TY,
                });
            }
            (u32::from(keepbits), false)
        }
        BitRoundMode::AbsoluteError { eb_abs } => {
            let eb_abs = T::from_f64(eb_abs.0);

            let mut encoded = data.into_owned();

            encoded.mapv_inplace(|x| {
                // subnormal, infinite, and NaN values are hard so just keep
                // them as is
                if !x.is_normal() {
                    return x;
                }

                let keepbits = BitRounder::keepbits_from_eb_rel(NonNegative(eb_abs / x.abs()));
                let bit_round = BitRounder::new(keepbits);

                bit_round.apply(x)
            });

            return Ok(encoded);
        }
        BitRoundMode::RelativeError { eb_rel } => (BitRounder::keepbits_from_eb_rel(*eb_rel), true),
    };

    let mut encoded = data.into_owned();

    // Early return if no bit rounding needs to happen
    // - required since the ties to even impl does not work in this case
    if keepbits == T::MANITSSA_BITS {
        return Ok(encoded);
    }

    let bit_round = BitRounder::new(keepbits);

    encoded.mapv_inplace(|x| {
        // subnormal, infinite, and NaN values are hard so just keep them as is
        if keep_non_normal && !x.is_normal() {
            return x;
        }

        bit_round.apply(x)
    });

    Ok(encoded)
}

struct BitRounder<T: Float> {
    ulp_half: T::Binary,
    keep_mask: T::Binary,
    shift: u32,
}

impl<T: Float> BitRounder<T> {
    #[inline]
    fn new(keepbits: u32) -> Self {
        // half of unit in last place (ulp)
        let ulp_half = T::MANTISSA_MASK >> (keepbits + 1);
        // mask to zero out trailing mantissa bits
        let keep_mask = !(T::MANTISSA_MASK >> keepbits);
        // shift to extract the least significant bit of the exponent
        let shift = T::MANITSSA_BITS - keepbits;

        Self {
            ulp_half,
            keep_mask,
            shift,
        }
    }

    fn keepbits_from_eb_rel(eb_rel: NonNegative<T>) -> u32 {
        let keepbits = -(eb_rel.0.normal_log2_floor()) - 1;
        // keepbits must be within the range of the mantissa bits of single precision.
        #[expect(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        // no sign loss or truncation since we clamp to between 0 and a u32
        let keepbits = i64::from(keepbits).clamp(0, i64::from(T::MANITSSA_BITS)) as u32;
        keepbits
    }

    #[inline]
    fn apply(&self, x: T) -> T {
        let mut bits = T::to_binary(x);

        // add ulp/2 with ties to even
        bits += self.ulp_half + ((bits >> self.shift) & T::BINARY_ONE);

        // set the trailing bits to zero
        bits &= self.keep_mask;

        T::from_binary(bits)
    }
}

/// Floating point types.
pub trait Float: Sized + Copy + std::ops::Div<Self, Output = Self> {
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

    /// Returns the floating point category of the number
    fn is_normal(self) -> bool;

    /// Returns the floor of the base-2 logarithm as a signed integer
    fn normal_log2_floor(self) -> i16;

    /// Computes the absolute value
    #[must_use]
    fn abs(self) -> Self;

    /// Convert from an [`f64`] value
    fn from_f64(x: f64) -> Self;
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

    fn is_normal(self) -> bool {
        self.is_normal()
    }

    fn normal_log2_floor(self) -> i16 {
        (((self.to_bits() >> 23) & 0xff) as i16) - 127
    }

    fn abs(self) -> Self {
        self.abs()
    }

    #[expect(clippy::cast_possible_truncation)]
    fn from_f64(x: f64) -> Self {
        x as Self
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

    fn is_normal(self) -> bool {
        self.is_normal()
    }

    fn normal_log2_floor(self) -> i16 {
        (((self.to_bits() >> 52) & 0x7ff) as i16) - 1023
    }

    fn abs(self) -> Self {
        self.abs()
    }

    fn from_f64(x: f64) -> Self {
        x
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use ndarray::{Array1, ArrayView1};

    use super::*;

    #[test]
    #[expect(clippy::too_many_lines)]
    fn no_mantissa() {
        assert_eq!(
            bit_round(
                ArrayView1::from(&[0.0_f32]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![0.0_f32])
        );
        assert_eq!(
            bit_round(
                ArrayView1::from(&[1.0_f32]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![1.0_f32])
        );
        // tie to even rounds up as the offset exponent is odd
        assert_eq!(
            bit_round(
                ArrayView1::from(&[1.5_f32]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![2.0_f32])
        );
        assert_eq!(
            bit_round(
                ArrayView1::from(&[2.0_f32]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![2.0_f32])
        );
        assert_eq!(
            bit_round(
                ArrayView1::from(&[2.5_f32]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![2.0_f32])
        );
        // tie to even rounds down as the offset exponent is even
        assert_eq!(
            bit_round(
                ArrayView1::from(&[3.0_f32]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![2.0_f32])
        );
        assert_eq!(
            bit_round(
                ArrayView1::from(&[3.5_f32]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![4.0_f32])
        );
        assert_eq!(
            bit_round(
                ArrayView1::from(&[4.0_f32]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![4.0_f32])
        );
        assert_eq!(
            bit_round(
                ArrayView1::from(&[5.0_f32]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![4.0_f32])
        );
        // tie to even rounds up as the offset exponent is odd
        assert_eq!(
            bit_round(
                ArrayView1::from(&[6.0_f32]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![8.0_f32])
        );
        assert_eq!(
            bit_round(
                ArrayView1::from(&[7.0_f32]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![8.0_f32])
        );
        assert_eq!(
            bit_round(
                ArrayView1::from(&[8.0_f32]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![8.0_f32])
        );

        assert_eq!(
            bit_round(
                ArrayView1::from(&[0.0_f64]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![0.0_f64])
        );
        assert_eq!(
            bit_round(
                ArrayView1::from(&[1.0_f64]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![1.0_f64])
        );
        // tie to even rounds up as the offset exponent is odd
        assert_eq!(
            bit_round(
                ArrayView1::from(&[1.5_f64]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![2.0_f64])
        );
        assert_eq!(
            bit_round(
                ArrayView1::from(&[2.0_f64]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![2.0_f64])
        );
        assert_eq!(
            bit_round(
                ArrayView1::from(&[2.5_f64]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![2.0_f64])
        );
        // tie to even rounds down as the offset exponent is even
        assert_eq!(
            bit_round(
                ArrayView1::from(&[3.0_f64]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![2.0_f64])
        );
        assert_eq!(
            bit_round(
                ArrayView1::from(&[3.5_f64]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![4.0_f64])
        );
        assert_eq!(
            bit_round(
                ArrayView1::from(&[4.0_f64]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![4.0_f64])
        );
        assert_eq!(
            bit_round(
                ArrayView1::from(&[5.0_f64]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![4.0_f64])
        );
        // tie to even rounds up as the offset exponent is odd
        assert_eq!(
            bit_round(
                ArrayView1::from(&[6.0_f64]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![8.0_f64])
        );
        assert_eq!(
            bit_round(
                ArrayView1::from(&[7.0_f64]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![8.0_f64])
        );
        assert_eq!(
            bit_round(
                ArrayView1::from(&[8.0_f64]),
                &BitRoundMode::Keepbits { keepbits: 0 }
            )
            .unwrap(),
            Array1::from_vec(vec![8.0_f64])
        );
    }

    #[test]
    #[expect(clippy::cast_possible_truncation)]
    fn full_mantissa() {
        fn full<T: Float>(x: T) -> T {
            T::from_binary(T::to_binary(x) + T::MANTISSA_MASK)
        }

        for v in [0.0_f32, 1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32] {
            assert_eq!(
                bit_round(
                    ArrayView1::from(&[full(v)]),
                    &BitRoundMode::Keepbits {
                        keepbits: f32::MANITSSA_BITS as u8
                    }
                )
                .unwrap(),
                Array1::from_vec(vec![full(v)])
            );
        }

        for v in [0.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64] {
            assert_eq!(
                bit_round(
                    ArrayView1::from(&[full(v)]),
                    &BitRoundMode::Keepbits {
                        keepbits: f64::MANITSSA_BITS as u8
                    }
                )
                .unwrap(),
                Array1::from_vec(vec![full(v)])
            );
        }
    }

    #[test]
    fn normal_log2_floor_f32() {
        for e in -100_i16..100 {
            let b = f32::from(e).exp2();
            for f in [0.55, 0.75, 0.9, 1.0, 1.1, 1.5, 1.95] {
                let x = b * f;

                #[expect(clippy::cast_possible_truncation)]
                let math = x.log2().floor() as i16;
                let binary = x.normal_log2_floor();

                assert_eq!(math, binary, "{x}");
            }
        }

        assert_eq!(i32::from(0.0_f32.normal_log2_floor()), f32::MIN_EXP - 2);
    }

    #[test]
    fn normal_log2_floor_f64() {
        for e in -100_i32..100 {
            let b = f64::from(e).exp2();
            for f in [0.55, 0.75, 0.9, 1.0, 1.1, 1.5, 1.95] {
                let x = b * f;

                #[expect(clippy::cast_possible_truncation)]
                let math = x.log2().floor() as i16;
                let binary = x.normal_log2_floor();

                assert_eq!(math, binary, "{x}");
            }
        }

        assert_eq!(i32::from(0.0_f64.normal_log2_floor()), f64::MIN_EXP - 2);
    }
}
