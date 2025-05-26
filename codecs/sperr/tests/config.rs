#![expect(missing_docs)]

use ::{
    log as _, ndarray as _, num_traits as _, numcodecs_jpeg2000::Jpeg2000CompressionMode,
    openjpeg_sys as _, postcard as _, schemars as _, simple_logger as _, thiserror as _,
};

use numcodecs::StaticCodec;
use numcodecs_jpeg2000::Jpeg2000Codec;
use serde::Deserialize;
use serde_json::json;

#[test]
#[should_panic(expected = "missing field `mode`")]
fn empty_config() {
    let _ = Jpeg2000Codec::from_config(Deserialize::deserialize(json!({})).unwrap());
}

#[test]
fn psnr_config() {
    let codec = Jpeg2000Codec::from_config(
        Deserialize::deserialize(json!({
            "mode": "psnr",
            "psnr": 42.0,
        }))
        .unwrap(),
    );

    assert!(matches!(
        codec.mode,
        Jpeg2000CompressionMode::PSNR { psnr: 42.0 }
    ));
}

#[test]
fn rate_config() {
    let codec = Jpeg2000Codec::from_config(
        Deserialize::deserialize(json!({
            "mode": "rate",
            "rate": 10.0,
        }))
        .unwrap(),
    );

    assert!(matches!(
        codec.mode,
        Jpeg2000CompressionMode::Rate { rate: 10.0 }
    ));
}

#[test]
fn lossless_config() {
    let codec = Jpeg2000Codec::from_config(
        Deserialize::deserialize(json!({
            "mode": "lossless",
        }))
        .unwrap(),
    );

    assert!(matches!(codec.mode, Jpeg2000CompressionMode::Lossless));
}
