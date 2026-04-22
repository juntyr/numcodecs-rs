#![expect(missing_docs)]

use ::{
    ndarray as _, postcard as _, schemars as _, serde as _, serde_json as _, thiserror as _,
    zfp_sys as _,
};

use numcodecs::{DynCodecType, StaticCodecType};
use numcodecs_zfp_classic::ZfpClassicCodec;

#[test]
fn schema() {
    let schema = format!(
        "{:#}",
        StaticCodecType::<ZfpClassicCodec>::of()
            .codec_config_schema()
            .to_value()
    );

    #[expect(clippy::manual_assert, clippy::panic)]
    if schema != include_str!("schema.json") {
        panic!("ZfpClassic schema has changed\n===\n{schema}\n===");
    }
}
