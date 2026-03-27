#![expect(missing_docs)]
#![expect(clippy::unwrap_used)]

use ::{
    lc_framework as _, ndarray as _, postcard as _, schemars as _, serde_repr as _, thiserror as _,
};

use numcodecs::StaticCodec;
use numcodecs_lc::LcCodec;
use serde::Deserialize;
use serde_json::json;

#[test]
#[should_panic(expected = "missing field `components`")]
fn empty_config() {
    let _ = LcCodec::from_config(Deserialize::deserialize(json!({})).unwrap());
}

#[test]
#[should_panic(expected = "expected at least one component")]
fn no_components() {
    let _ = LcCodec::from_config(
        Deserialize::deserialize(json!({
            "components": []
        }))
        .unwrap(),
    );
}

#[test]
#[should_panic(expected = "expected at most 8 components")]
fn too_many_components() {
    let _ = LcCodec::from_config(
        Deserialize::deserialize(json!({
            "components": [
                { "id": "NUL" }, { "id": "NUL" }, { "id": "NUL" }, { "id": "NUL" },
                { "id": "NUL" }, { "id": "NUL" }, { "id": "NUL" }, { "id": "NUL" },
                { "id": "NUL" }, { "id": "NUL" }, { "id": "NUL" }, { "id": "NUL" },
            ],
        }))
        .unwrap(),
    );
}

#[test]
fn with_preprocessors() {
    let _ = LcCodec::from_config(Deserialize::deserialize(json!({
        "preprocessors": [
            { "id": "QUANT", "dtype": "f32", "kind": "REL", "error_bound": 0.1, "decorrelation": "R" },
            { "id": "LOR", "dtype": "i32" },
        ],
        "components": [
            { "id": "BIT", "size": 4 },
            { "id": "RLE", "size": 4 },
        ],
    })).unwrap());
}
