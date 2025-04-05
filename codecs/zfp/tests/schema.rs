#![expect(missing_docs)]

use ::{
    ndarray as _, postcard as _, schemars as _, serde as _, serde_json as _, thiserror as _,
    zfp_sys as _,
};

use numcodecs::{DynCodecType, StaticCodecType};
use numcodecs_zfp::ZfpCodec;

#[test]
fn schema() {
    let schema = format!(
        "{:#}",
        StaticCodecType::<ZfpCodec>::of()
            .codec_config_schema()
            .to_value()
    );

    if schema != include_str!("schema.json") {
        panic!("Zfp schema has changed\n===\n{schema}\n===");
    }
}
