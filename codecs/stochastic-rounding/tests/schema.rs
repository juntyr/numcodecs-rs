#![expect(missing_docs)]

use ::{
    ndarray as _, num_traits as _, rand as _, schemars as _, serde as _, thiserror as _,
    wyhash as _,
};

use numcodecs::{DynCodecType, StaticCodecType};
use numcodecs_stochastic_rounding::StochasticRoundingCodec;

#[test]
fn schema() {
    let schema = format!(
        "{:#}",
        StaticCodecType::<StochasticRoundingCodec>::of()
            .codec_config_schema()
            .to_value()
    );

    #[expect(clippy::manual_assert, clippy::panic)]
    if schema != include_str!("schema.json") {
        panic!("StochasticRounding schema has changed\n===\n{schema}\n===");
    }
}
