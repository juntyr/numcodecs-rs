#![expect(missing_docs)]

use ::{ndarray as _, schemars as _, serde as _, thiserror as _};

use numcodecs::{DynCodecType, StaticCodecType};
use numcodecs_bit_round::BitRoundCodec;

#[test]
fn schema() {
    let schema = format!(
        "{:#}",
        StaticCodecType::<BitRoundCodec>::of()
            .codec_config_schema()
            .to_value()
    );

    if schema != include_str!("schema.json") {
        panic!("BitRound schema has changed\n===\n{schema}\n===");
    }
}
