#![expect(missing_docs)]

use ::{schemars as _, serde as _, thiserror as _};

use numcodecs::{DynCodecType, StaticCodecType};
use numcodecs_identity::IdentityCodec;

#[test]
fn schema() {
    let schema = format!(
        "{:#}",
        StaticCodecType::<IdentityCodec>::of()
            .codec_config_schema()
            .to_value()
    );

    #[expect(clippy::manual_assert, clippy::panic)]
    if schema != include_str!("schema.json") {
        panic!("Identity schema has changed\n===\n{schema}\n===");
    }
}
