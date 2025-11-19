#![expect(missing_docs)]

use ::{
    ndarray as _, num_traits as _, postcard as _, qpet_sz as _, schemars as _, serde as _,
    serde_json as _, thiserror as _, zstd_sys as _,
};

use numcodecs::{DynCodecType, StaticCodecType};
use numcodecs_qpet_sz::QpetSzCodec;

#[test]
fn schema() {
    let schema = format!(
        "{:#}",
        StaticCodecType::<QpetSzCodec>::of()
            .codec_config_schema()
            .to_value()
    );

    if schema != include_str!("schema.json") {
        panic!("QPET-SZ schema has changed\n===\n{schema}\n===");
    }
}
