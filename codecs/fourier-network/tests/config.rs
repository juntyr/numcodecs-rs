#![expect(missing_docs)]

use numcodecs::StaticCodec;
use numcodecs_fourier_network::FourierNetworkCodec;
use serde::Deserialize;
use serde_json::json;

use ::{
    burn as _, bytemuck as _, itertools as _, log as _, ndarray as _, num_traits as _,
    schemars as _, simple_logger as _, thiserror as _,
};

#[test]
#[should_panic(expected = "missing field `fourier_features`")]
fn empty_config() {
    let _ = FourierNetworkCodec::from_config(Deserialize::deserialize(json!({})).unwrap());
}

#[test]
#[should_panic(expected = "missing field `mini_batch_size`")]
fn config_missing_mini_batch_size() {
    let _ = FourierNetworkCodec::from_config(
        Deserialize::deserialize(json!({
            "fourier_features": 16,
            "fourier_scale": 10.0,
            "num_blocks": 2,
            "learning_rate": 1e-4,
            "num_epochs": 100,
            "seed": 42,
        }))
        .unwrap(),
    );
}

#[test]
fn config_no_mini_batching() {
    let codec = FourierNetworkCodec::from_config(
        Deserialize::deserialize(json!({
            "fourier_features": 16,
            "fourier_scale": 10.0,
            "num_blocks": 2,
            "learning_rate": 1e-4,
            "num_epochs": 100,
            "mini_batch_size": null,
            "seed": 42,
        }))
        .unwrap(),
    );

    assert!(codec.mini_batch_size.is_none());
}

#[test]
fn config_mini_batch_size() {
    let codec = FourierNetworkCodec::from_config(
        Deserialize::deserialize(json!({
            "fourier_features": 16,
            "fourier_scale": 10.0,
            "num_blocks": 2,
            "learning_rate": 1e-4,
            "num_epochs": 100,
            "mini_batch_size": 10,
            "seed": 42,
        }))
        .unwrap(),
    );

    assert!(codec.mini_batch_size.is_some());
}
