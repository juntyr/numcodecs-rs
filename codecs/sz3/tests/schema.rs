#![expect(missing_docs)]

use ::{
    ndarray as _, postcard as _, schemars as _, serde as _, serde_json as _, sz3 as _,
    thiserror as _, zstd_sys as _,
};

use numcodecs::{DynCodecType, StaticCodecType};
use numcodecs_sz3::Sz3Codec;

#[test]
fn schema() {
    let schema = format!(
        "{:#}",
        StaticCodecType::<Sz3Codec>::of()
            .codec_config_schema()
            .to_value()
    );

    if schema != include_str!("schema.json") {
        panic!("Sz3 schema has changed\n===\n{schema}\n===");
    }
}
