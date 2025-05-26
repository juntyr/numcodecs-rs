#![expect(missing_docs)]

use ::{
    ndarray as _, num_traits as _, postcard as _, schemars as _, serde as _, serde_json as _,
    sperr as _, thiserror as _,
};

use numcodecs::{DynCodecType, StaticCodecType};
use numcodecs_sperr::SperrCodec;

#[test]
fn schema() {
    let schema = format!(
        "{:#}",
        StaticCodecType::<SperrCodec>::of()
            .codec_config_schema()
            .to_value()
    );

    if schema != include_str!("schema.json") {
        panic!("Sperr schema has changed\n===\n{schema}\n===");
    }
}
