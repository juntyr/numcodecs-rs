#![expect(missing_docs)]

use ::{
    ndarray as _, num_traits as _, postcard as _, qpet_sperr as _, schemars as _, thiserror as _,
    zstd_sys as _,
};

#[cfg(target_arch = "wasm32")]
use ::gmp_mpfr_sys as _;

use numcodecs::StaticCodec;
use numcodecs_qpet_sperr::{QpetSperrCodec, QpetSperrCompressionMode};
use serde::Deserialize;
use serde_json::json;

#[test]
#[should_panic(expected = "missing field `mode`")]
fn empty_config() {
    let _ = QpetSperrCodec::from_config(Deserialize::deserialize(json!({})).unwrap());
}

#[test]
fn symbolic_qoi_config() {
    let codec = QpetSperrCodec::from_config(
        Deserialize::deserialize(json!({
            "mode": "qoi-symbolic",
            "qoi": "x^2",
            "qoi_pwe": 0.1,
        }))
        .unwrap(),
    );

    assert!(matches!(
        codec.mode,
        QpetSperrCompressionMode::SymbolicQuantityOfInterest { qoi, .. } if qoi == "x^2"
    ));
}
