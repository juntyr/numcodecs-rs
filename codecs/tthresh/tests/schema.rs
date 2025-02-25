#![expect(missing_docs)]

use ::{
    ndarray as _, num_traits as _, schemars as _, serde as _, serde_json as _, thiserror as _,
    tthresh as _,
};

use numcodecs::{DynCodecType, StaticCodecType};
use numcodecs_tthresh::TthreshCodec;

#[test]
fn schema() {
    let schema = format!(
        "{:#}",
        StaticCodecType::<TthreshCodec>::of()
            .codec_config_schema()
            .to_value()
    );

    if schema != include_str!("schema.json") {
        panic!("Tthresh schema has changed\n===\n{schema}\n===");
    }
}
