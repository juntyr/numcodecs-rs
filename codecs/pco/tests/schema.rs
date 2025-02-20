#![expect(missing_docs)]

use ::{
    ndarray as _, pco as _, postcard as _, schemars as _, serde as _, serde_json as _,
    serde_repr as _, thiserror as _,
};

use numcodecs::{DynCodecType, StaticCodecType};
use numcodecs_pco::Pcodec;

#[test]
fn schema() {
    let schema = format!(
        "{:#}",
        StaticCodecType::<Pcodec>::of()
            .codec_config_schema()
            .to_value()
    );

    if schema != include_str!("schema.json") {
        panic!("Pcodec schema has changed\n===\n{schema}\n===");
    }
}
