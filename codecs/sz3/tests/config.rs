#![allow(missing_docs)]

use ::{ndarray as _, postcard as _, schemars as _, sz3 as _, thiserror as _, zstd_sys as _};

use numcodecs::StaticCodec;
use numcodecs_sz3::Sz3Codec;
use serde::Deserialize;
use serde_json::json;

#[test]
#[should_panic(expected = "missing field `eb_mode`")]
fn empty_config() {
    let _ = Sz3Codec::from_config(Deserialize::deserialize(json!({})).unwrap());
}

#[test]
fn abs_config() {
    let codec = Sz3Codec::from_config(
        Deserialize::deserialize(json!({
            "eb_mode": "abs",
            "eb_abs": 1.0,
        }))
        .unwrap(),
    );

    assert!(codec.predictor.is_some());
    assert!(codec.encoder.is_some());
    assert!(codec.lossless.is_some());
}

#[test]
#[should_panic(expected = "unknown field `eb_rel`, expected `eb_abs`")]
fn abs_config_with_rel() {
    let _ = Sz3Codec::from_config(
        Deserialize::deserialize(json!({
            "eb_mode": "abs",
            "eb_abs": 1.0,
            "eb_rel": 1.0,
        }))
        .unwrap(),
    );
}

#[test]
fn config_only_abs() {
    let codec = Sz3Codec::from_config(
        Deserialize::deserialize(json!({
            "eb_mode": "abs",
            "eb_abs": 1.0,
            "predictor": null,
            "encoder": null,
            "lossless": null,
        }))
        .unwrap(),
    );

    assert!(codec.predictor.is_none());
    assert!(codec.encoder.is_none());
    assert!(codec.lossless.is_none());
}

#[test]
fn config_predictor() {
    let _ = Sz3Codec::from_config(
        Deserialize::deserialize(json!({
            "eb_mode": "rel",
            "eb_rel": 1.0,
            "predictor": "linear-interpolation",
        }))
        .unwrap(),
    );

    let _ = Sz3Codec::from_config(
        Deserialize::deserialize(json!({
            "eb_mode": "rel",
            "eb_rel": 1.0,
            "predictor": "cubic-interpolation-lorenzo",
        }))
        .unwrap(),
    );

    let _ = Sz3Codec::from_config(
        Deserialize::deserialize(json!({
            "eb_mode": "psnr",
            "eb_psnr": 1.0,
            "predictor": "lorenzo-regression",
        }))
        .unwrap(),
    );

    let _ = Sz3Codec::from_config(
        Deserialize::deserialize(json!({
            "eb_mode": "l2",
            "eb_l2": 1.0,
            "predictor": null,
        }))
        .unwrap(),
    );
}

#[test]
fn config_encoder() {
    let _ = Sz3Codec::from_config(
        Deserialize::deserialize(json!({
            "eb_mode": "rel",
            "eb_rel": 1.0,
            "encoder": "huffman",
        }))
        .unwrap(),
    );

    let _ = Sz3Codec::from_config(
        Deserialize::deserialize(json!({
            "eb_mode": "abs-and-rel",
            "eb_abs": 1.0,
            "eb_rel": 1.0,
            "encoder": "arithmetic",
        }))
        .unwrap(),
    );

    let _ = Sz3Codec::from_config(
        Deserialize::deserialize(json!({
            "eb_mode": "abs-or-rel",
            "eb_abs": 1.0,
            "eb_rel": 1.0,
            "encoder": null,
        }))
        .unwrap(),
    );
}

#[test]
fn config_lossless() {
    let _ = Sz3Codec::from_config(
        Deserialize::deserialize(json!({
            "eb_mode": "psnr",
            "eb_psnr": 1.0,
            "lossless": "zstd",
        }))
        .unwrap(),
    );

    let _ = Sz3Codec::from_config(
        Deserialize::deserialize(json!({
            "eb_mode": "abs",
            "eb_abs": 1.0,
            "lossless": null,
        }))
        .unwrap(),
    );
}
