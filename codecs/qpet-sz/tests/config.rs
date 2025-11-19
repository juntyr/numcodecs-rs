#![expect(missing_docs)]

use ::{
    ndarray as _, num_traits as _, postcard as _, qpet_sz as _, schemars as _, thiserror as _,
    zstd_sys as _,
};

use numcodecs::StaticCodec;
use numcodecs_qpet_sz::{QpetSzCodec, QpetSzCompressionMode};
use serde::Deserialize;
use serde_json::json;

#[test]
#[should_panic(expected = "missing field `mode`")]
fn empty_config() {
    let _ = QpetSzCodec::from_config(Deserialize::deserialize(json!({})).unwrap());
}

#[test]
fn symbolic_qoi_config() {
    let codec = QpetSzCodec::from_config(
        Deserialize::deserialize(json!({
            "mode": "qoi-symbolic",
            "qoi": "x^2",
            "qoi_eb_mode": "abs",
            "qoi_eb_abs": 0.1,
            // "data_eb_mode": "abs",
            // "data_eb_abs": 1000,
        }))
        .unwrap(),
    );

    assert!(matches!(
        codec.mode,
        QpetSzCompressionMode::SymbolicQuantityOfInterest { qoi, .. } if qoi == "x^2"
    ));
}
