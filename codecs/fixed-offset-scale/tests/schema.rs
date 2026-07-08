#![expect(missing_docs)]

use ::{ndarray as _, num_traits as _, schemars as _, serde as _, thiserror as _};

use numcodecs::{DynCodecType, StaticCodecType};
use numcodecs_fixed_offset_scale::FixedOffsetScaleCodec;

#[test]
fn schema() {
    let schema = format!(
        "{:#}",
        StaticCodecType::<FixedOffsetScaleCodec>::of()
            .codec_config_schema()
            .to_value()
    );

    #[expect(clippy::manual_assert, clippy::panic)]
    if schema != include_str!("schema.json") {
        panic!("FixedOffsetScale schema has changed\n===\n{schema}\n===");
    }
}
