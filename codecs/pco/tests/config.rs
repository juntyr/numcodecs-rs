#![allow(missing_docs)]

use std::num::NonZero;

use ::{ndarray as _, postcard as _, schemars as _, serde_repr as _, thiserror as _};

use numcodecs::StaticCodec;
use numcodecs_pco::{PcoDeltaEncoding, PcoLevel, PcoMode, Pcodec};
use serde::Deserialize;
use serde_json::json;

#[test]
fn empty_config() {
    let codec = Pcodec::from_config(Deserialize::deserialize(json!({})).unwrap());

    assert_eq!(codec.level, PcoLevel::PcoLevel8);
    assert_eq!(codec.mode, PcoMode::Auto);
    assert_eq!(codec.delta, PcoDeltaEncoding::Auto);
    assert_eq!(
        codec.equal_pages_up_to,
        NonZero::new(pco::DEFAULT_MAX_PAGE_N).unwrap(),
    );
}

#[test]
fn minimal_config() {
    let codec = Pcodec::from_config(
        Deserialize::deserialize(json!({
            "level": 0,
            "mode": "classic",
            "delta": "none",
            "equal_pages_up_to": 1000,
        }))
        .unwrap(),
    );

    assert_eq!(codec.level, PcoLevel::PcoLevel0);
    assert_eq!(codec.mode, PcoMode::Classic);
    assert_eq!(codec.delta, PcoDeltaEncoding::None);
    assert_eq!(codec.equal_pages_up_to, NonZero::new(1000).unwrap(),);
}
