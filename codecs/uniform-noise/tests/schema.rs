#![expect(missing_docs)]

use ::{
    ndarray as _, num_traits as _, rand as _, schemars as _, serde as _, thiserror as _,
    wyhash as _,
};

use numcodecs::{DynCodecType, StaticCodecType};
use numcodecs_uniform_noise::UniformNoiseCodec;

#[test]
fn schema() {
    let schema = format!(
        "{:#}",
        StaticCodecType::<UniformNoiseCodec>::of()
            .codec_config_schema()
            .to_value()
    );

    #[expect(clippy::manual_assert, clippy::panic)]
    if schema != include_str!("schema.json") {
        panic!("UniformNoise schema has changed\n===\n{schema}\n===");
    }
}
