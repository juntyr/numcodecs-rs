#![expect(missing_docs)]
#![expect(clippy::unwrap_used)]

use ::{ebcc as _, ndarray as _, num_traits as _, postcard as _, schemars as _, thiserror as _};

use numcodecs::StaticCodec;
use numcodecs_ebcc::{EbccCodec, EbccResidualType};
use serde::Deserialize;
use serde_json::json;

#[test]
#[should_panic(expected = "missing field `base_cr`")]
fn empty_config() {
    let _ = EbccCodec::from_config(Deserialize::deserialize(json!({})).unwrap());
}

#[test]
#[should_panic(expected = "expected a positive value")]
fn negative_base_cr() {
    let _ = EbccCodec::from_config(
        Deserialize::deserialize(json!({
            "base_cr": -1.0,
        }))
        .unwrap(),
    );
}

#[test]
fn jpeg2000_only() {
    let codec = EbccCodec::from_config(
        Deserialize::deserialize(json!({
            "base_cr": 10.0,
            "residual": "jpeg2000-only",
        }))
        .unwrap(),
    );

    assert_eq!(codec.base_cr, 10.0);
    assert_eq!(codec.residual, EbccResidualType::Jpeg2000Only);
}

#[test]
fn absolute_error() {
    let codec = EbccCodec::from_config(
        Deserialize::deserialize(json!({
            "base_cr": 10.0,
            "residual": "absolute",
            "error": 0.1,
        }))
        .unwrap(),
    );

    assert_eq!(codec.base_cr, 10.0);
    assert!(matches!(codec.residual, EbccResidualType::AbsoluteError { error } if error == 0.1));
}

#[test]
fn relative_error() {
    let codec = EbccCodec::from_config(
        Deserialize::deserialize(json!({
            "base_cr": 10.0,
            "residual": "relative",
            "error": 2.4,
        }))
        .unwrap(),
    );

    assert_eq!(codec.base_cr, 10.0);
    assert!(matches!(codec.residual, EbccResidualType::RelativeError { error } if error == 2.4));
}

#[test]
#[should_panic(
    expected = "unknown variant `invalid`, expected one of `jpeg2000-only`, `absolute`, `relative`"
)]
fn invalid_residual() {
    let _ = EbccCodec::from_config(
        Deserialize::deserialize(json!({
            "base_cr": 10.0,
            "residual": "invalid",
        }))
        .unwrap(),
    );
}
