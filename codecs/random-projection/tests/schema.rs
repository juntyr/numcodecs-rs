#![allow(missing_docs)]

use numcodecs::{DynCodecType, StaticCodecType};
use numcodecs_random_projection::RandomProjectionCodec;

use ::{
    ndarray as _, ndarray_rand as _, num_traits as _, schemars as _, thiserror as _, serde as _, serde_json as _,
};

#[test]
fn schema() {
    let schema = format!(
        "{:#}",
        StaticCodecType::<RandomProjectionCodec>::of()
            .codec_config_schema()
            .to_value()
    );

    if schema != include_str!("schema.json") {
        panic!("RandomProjection schema has changed\n===\n{schema}\n===");
    }
}
