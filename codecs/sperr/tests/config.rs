#![expect(missing_docs, clippy::unwrap_used)]

use ::{ndarray as _, num_traits as _, postcard as _, schemars as _, sperr as _, thiserror as _};

use numcodecs::StaticCodec;
use numcodecs_sperr::{SperrCodec, SperrCompressionMode};
use serde::Deserialize;
use serde_json::json;

#[test]
#[should_panic(expected = "missing field `mode`")]
fn empty_config() {
    let _ = SperrCodec::from_config(Deserialize::deserialize(json!({})).unwrap());
}

#[test]
#[expect(clippy::float_cmp)]
fn bpp_config() {
    let codec = SperrCodec::from_config(
        Deserialize::deserialize(json!({
            "mode": "bpp",
            "bpp": 1.0,
        }))
        .unwrap(),
    );

    assert!(matches!(
        codec.mode,
        SperrCompressionMode::BitsPerPixel { bpp } if bpp.get() == 1.0
    ));
}

#[test]
#[expect(clippy::float_cmp)]
fn psnr_config() {
    let codec = SperrCodec::from_config(
        Deserialize::deserialize(json!({
            "mode": "psnr",
            "psnr": 42.0,
        }))
        .unwrap(),
    );

    assert!(matches!(
        codec.mode,
        SperrCompressionMode::PeakSignalToNoiseRatio { psnr } if psnr.get() == 42.0
    ));
}

#[test]
#[expect(clippy::float_cmp)]
fn pwe_config() {
    let codec = SperrCodec::from_config(
        Deserialize::deserialize(json!({
            "mode": "pwe",
            "pwe": 0.1,
        }))
        .unwrap(),
    );

    assert!(matches!(
        codec.mode,
        SperrCompressionMode::PointwiseError { pwe } if pwe.get() == 0.1
    ));
}

#[test]
#[expect(clippy::float_cmp)]
fn q_config() {
    let codec = SperrCodec::from_config(
        Deserialize::deserialize(json!({
            "mode": "q",
            "q": 1.5,
        }))
        .unwrap(),
    );

    assert!(matches!(
        codec.mode,
        SperrCompressionMode::QuantisationStep { q } if q.get() == 1.5
    ));
}
