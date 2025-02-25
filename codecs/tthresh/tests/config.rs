#![expect(missing_docs)]

use ::{ndarray as _, num_traits as _, schemars as _, thiserror as _, tthresh as _};

use numcodecs::StaticCodec;
use numcodecs_tthresh::{TthreshCodec, TthreshErrorBound};
use serde::Deserialize;
use serde_json::json;

#[test]
#[should_panic(expected = "missing field `eb_mode`")]
fn empty_config() {
    let _ = TthreshCodec::from_config(Deserialize::deserialize(json!({})).unwrap());
}

#[test]
fn eps_config() {
    let codec = TthreshCodec::from_config(
        Deserialize::deserialize(json!({
            "eb_mode": "eps",
            "eb_eps": 1.0,
        }))
        .unwrap(),
    );

    assert!(matches!(codec.error_bound, TthreshErrorBound::Eps { .. }));
}

#[test]
fn rmse_config() {
    let codec = TthreshCodec::from_config(
        Deserialize::deserialize(json!({
            "eb_mode": "rmse",
            "eb_rmse": 1.0,
        }))
        .unwrap(),
    );

    assert!(matches!(codec.error_bound, TthreshErrorBound::RMSE { .. }));
}

#[test]
fn psnr_config() {
    let codec = TthreshCodec::from_config(
        Deserialize::deserialize(json!({
            "eb_mode": "psnr",
            "eb_psnr": 1.0,
        }))
        .unwrap(),
    );

    assert!(matches!(codec.error_bound, TthreshErrorBound::PSNR { .. }));
}
