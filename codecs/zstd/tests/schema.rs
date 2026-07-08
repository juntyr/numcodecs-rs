#![expect(missing_docs)]

use ::{
    ndarray as _, postcard as _, schemars as _, serde as _, thiserror as _, zstd as _,
    zstd_sys as _,
};

use numcodecs::{DynCodecType, StaticCodecType};
use numcodecs_zstd::ZstdCodec;

#[test]
fn schema() {
    let schema = format!(
        "{:#}",
        StaticCodecType::<ZstdCodec>::of()
            .codec_config_schema()
            .to_value()
    );

    #[expect(clippy::manual_assert, clippy::panic)]
    if schema != include_str!("schema.json") {
        panic!("Zstd schema has changed\n===\n{schema}\n===");
    }
}
