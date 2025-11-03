#![expect(missing_docs)]

use ::{
    ndarray as _, num_traits as _, postcard as _, qpet_sperr as _, schemars as _, serde as _,
    serde_json as _, thiserror as _,
};

#[cfg(target_arch = "wasm32")]
use ::gmp_mpfr_sys as _;

use numcodecs::{DynCodecType, StaticCodecType};
use numcodecs_qpet_sperr::QpetSperrCodec;

#[test]
fn schema() {
    let schema = format!(
        "{:#}",
        StaticCodecType::<QpetSperrCodec>::of()
            .codec_config_schema()
            .to_value()
    );

    if schema != include_str!("schema.json") {
        panic!("QPET-SPERR schema has changed\n===\n{schema}\n===");
    }
}
