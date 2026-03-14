#![expect(missing_docs)]

use ::{libpressio as _, ndarray as _, schemars as _, serde as _, serde_ndim as _, thiserror as _};

use numcodecs::{DynCodecType, StaticCodecType};
use numcodecs_pressio::PressioCodec;

#[test]
fn schema() {
    let schema = format!(
        "{:#}",
        StaticCodecType::<PressioCodec>::of()
            .codec_config_schema()
            .to_value()
    );

    if schema != include_str!("schema.json") {
        panic!("Pressio schema has changed\n===\n{schema}\n===");
    }
}
