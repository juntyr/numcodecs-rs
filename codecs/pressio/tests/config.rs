#![expect(missing_docs, clippy::unwrap_used)]

use ::{
    fragile as _, libpressio as _, ndarray as _, schemars as _, serde as _, serde_json as _,
    serde_ndim as _, thiserror as _,
};

use numcodecs::StaticCodec;
use numcodecs_pressio::PressioCodec;
use serde::Deserialize;
use serde_json::json;

#[test]
#[should_panic(expected = "missing field `compressor_id`")]
fn empty_config() {
    let _ = PressioCodec::from_config(Deserialize::deserialize(json!({})).unwrap());
}

#[test]
#[should_panic(expected = "invalid compressor id ???, choose one of")]
fn invalid_compressor_id() {
    let _ = PressioCodec::from_config(
        Deserialize::deserialize(json!({
            "compressor_id": "???",
        }))
        .unwrap(),
    );
}

#[test]
#[should_panic(expected = "unknown compressor configuration option: `abc`, use one of")]
fn unknown_compressor_config() {
    let _ = PressioCodec::from_config(
        Deserialize::deserialize(json!({
            "compressor_id": "linear_quantizer",
            "compressor_config": {
                "abc": 42,
            }
        }))
        .unwrap(),
    );
}

#[test]
#[should_panic(expected = "failed to cast option `pressio:abs`")]
fn option_cast_failure() {
    let _ = PressioCodec::from_config(
        Deserialize::deserialize(json!({
            "compressor_id": "linear_quantizer",
            "compressor_config": {
                "pressio:abs": "abc",
            },
        }))
        .unwrap(),
    );
}

#[test]
fn bool_array_data_option() {
    let _ = PressioCodec::from_config(
        Deserialize::deserialize(json!({
            "compressor_id": "mask_interpolation",
            "compressor_config": {
                "mask_interpolation:mask": [[true, false], [false, true]],
            },
        }))
        .unwrap(),
    );
}
