#![expect(missing_docs)]

use ::{
    lc_framework as _, ndarray as _, postcard as _, schemars as _, serde as _, serde_json as _,
    serde_repr as _, thiserror as _,
};

use numcodecs::{DynCodecType, StaticCodecType};
use numcodecs_lc::LcCodec;

#[test]
fn schema() {
    let schema = format!(
        "{:#}",
        StaticCodecType::<LcCodec>::of()
            .codec_config_schema()
            .to_value()
    );

    #[expect(clippy::manual_assert, clippy::panic)]
    if schema != include_str!("schema.json") {
        panic!("Lc schema has changed\n===\n{schema}\n===");
    }
}
