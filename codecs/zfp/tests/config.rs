#![expect(missing_docs)]

use ::{ndarray as _, postcard as _, schemars as _, thiserror as _, zfp_sys as _};

use numcodecs::StaticCodec;
use numcodecs_zfp::{ZfpCodec, ZfpCompressionMode};
use serde::Deserialize;
use serde_json::json;

#[test]
#[should_panic(expected = "missing field `mode`")]
fn empty_config() {
    let _ = ZfpCodec::from_config(Deserialize::deserialize(json!({})).unwrap());
}

#[test]
fn expert_config() {
    let codec = ZfpCodec::from_config(
        Deserialize::deserialize(json!({
            "mode": "expert",
            "min_bits": 1,
            "max_bits": 4,
            "max_prec": 2,
            "min_exp": 3,
        }))
        .unwrap(),
    );

    assert!(matches!(
        codec.mode,
        ZfpCompressionMode::Expert {
            min_bits: 1,
            max_bits: 4,
            max_prec: 2,
            min_exp: 3,
        }
    ));
}

#[test]
fn fixed_rate_config() {
    let codec = ZfpCodec::from_config(
        Deserialize::deserialize(json!({
            "mode": "fixed-rate",
            "rate": 4.0,
        }))
        .unwrap(),
    );

    assert!(matches!(
        codec.mode,
        ZfpCompressionMode::FixedRate { rate: 4.0 }
    ));
}

#[test]
fn fixed_precision_config() {
    let codec = ZfpCodec::from_config(
        Deserialize::deserialize(json!({
            "mode": "fixed-precision",
            "precision": 4,
        }))
        .unwrap(),
    );

    assert!(matches!(
        codec.mode,
        ZfpCompressionMode::FixedPrecision { precision: 4 }
    ));
}

#[test]
fn fixed_accuracy_config() {
    let codec = ZfpCodec::from_config(
        Deserialize::deserialize(json!({
            "mode": "fixed-accuracy",
            "tolerance": 0.5,
        }))
        .unwrap(),
    );

    assert!(matches!(
        codec.mode,
        ZfpCompressionMode::FixedAccuracy { tolerance: 0.5 }
    ));
}

#[test]
fn reversible_config() {
    let codec = ZfpCodec::from_config(
        Deserialize::deserialize(json!({
            "mode": "reversible",
        }))
        .unwrap(),
    );

    assert!(matches!(codec.mode, ZfpCompressionMode::Reversible));
}
