#![expect(missing_docs)]

use ::{
    log as _, ndarray as _, num_traits as _, openjpeg_sys as _, postcard as _, schemars as _,
    serde as _, serde_json as _, simple_logger as _, thiserror as _,
};

use numcodecs::{DynCodecType, StaticCodecType};
use numcodecs_jpeg2000::Jpeg2000Codec;

#[test]
fn schema() {
    let schema = format!(
        "{:#}",
        StaticCodecType::<Jpeg2000Codec>::of()
            .codec_config_schema()
            .to_value()
    );

    if schema != include_str!("schema.json") {
        panic!("Jpeg2000 schema has changed\n===\n{schema}\n===");
    }
}
