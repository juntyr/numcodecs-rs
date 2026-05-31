#![expect(missing_docs)]

use ::{numcodecs_registry as _, schemars as _, serde as _, thiserror as _};

use numcodecs::{DynCodecType, StaticCodecType};
use numcodecs_onion::OnionCodec;

#[test]
fn schema() {
    let schema = format!(
        "{:#}",
        StaticCodecType::<OnionCodec>::of()
            .codec_config_schema()
            .to_value()
    );

    if schema != include_str!("schema.json") {
        panic!("Onion schema has changed\n===\n{schema}\n===");
    }
}
