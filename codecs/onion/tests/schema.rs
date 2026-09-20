#![expect(missing_docs)]

use ::{schemars as _, serde as _, thiserror as _};

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

    #[expect(clippy::manual_assert, clippy::panic)]
    if schema != include_str!("schema.json") {
        panic!("Onion schema has changed\n===\n{schema}\n===");
    }
}

numcodecs_registry::export_global! {
    static REGISTRY: numcodecs_registry::EmptyRegistry = numcodecs_registry::EmptyRegistry;
}
