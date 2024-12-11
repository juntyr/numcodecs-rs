#![allow(missing_docs)]

use numcodecs::StaticCodec;
use numcodecs_random_projection::RandomProjectionCodec;
use serde::Deserialize;
use serde_json::json;

use ::{
    ndarray as _, ndarray_rand as _, num_traits as _, schemars as _, thiserror as _,
};

#[test]
#[should_panic(expected = "missing field `seed`")]
fn empty_config() {
    let _ = RandomProjectionCodec::from_config(Deserialize::deserialize(json!({})).unwrap());
}

#[test]
fn config_gaussian_projection() {
    let _ = RandomProjectionCodec::from_config(
        Deserialize::deserialize(json!({
            "seed": 42,
            "reduction": "johnson-lindenstrauss",
            "epsilon": 0.5,
            "projection": "gaussian",
        }))
        .unwrap(),
    );

    let _ = RandomProjectionCodec::from_config(
        Deserialize::deserialize(json!({
            "seed": 42,
            "reduction": "explicit",
            "k": 24,
            "projection": "gaussian",
        }))
        .unwrap(),
    );
}

#[test]
fn config_sparse_projection() {
    let _ = RandomProjectionCodec::from_config(
        Deserialize::deserialize(json!({
            "seed": 42,
            "reduction": "johnson-lindenstrauss",
            "epsilon": 0.5,
            "projection": "sparse",
        }))
        .unwrap(),
    );

    let _ = RandomProjectionCodec::from_config(
        Deserialize::deserialize(json!({
            "seed": 42,
            "reduction": "explicit",
            "k": 24,
            "projection": "sparse",
            "density": 0.5,
        }))
        .unwrap(),
    );
}
