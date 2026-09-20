#![expect(missing_docs)]

use ::{ndarray as _, num_traits as _, schemars as _, serde as _, thiserror as _};

use numcodecs::{DynCodecType, StaticCodecType};
use numcodecs_log::LogCodec;

#[test]
fn schema() {
    let schema = format!(
        "{:#}",
        StaticCodecType::<LogCodec>::of()
            .codec_config_schema()
            .to_value()
    );

    #[expect(clippy::manual_assert, clippy::panic)]
    if schema != include_str!("schema.json") {
        panic!("Log schema has changed\n===\n{schema}\n===");
    }
}
