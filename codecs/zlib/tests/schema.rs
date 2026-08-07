#![expect(missing_docs)]

use ::{
    miniz_oxide as _, ndarray as _, postcard as _, schemars as _, serde as _, serde_repr as _,
    thiserror as _,
};

use numcodecs::{DynCodecType, StaticCodecType};
use numcodecs_zlib::ZlibCodec;

#[test]
fn schema() {
    let schema = format!(
        "{:#}",
        StaticCodecType::<ZlibCodec>::of()
            .codec_config_schema()
            .to_value()
    );

    #[expect(clippy::manual_assert, clippy::panic)]
    if schema != include_str!("schema.json") {
        panic!("Zlib schema has changed\n===\n{schema}\n===");
    }
}
