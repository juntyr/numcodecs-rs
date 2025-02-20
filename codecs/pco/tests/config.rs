#![expect(missing_docs)]

use std::num::{NonZero, NonZeroUsize};

use ::{ndarray as _, postcard as _, schemars as _, serde_repr as _, thiserror as _};

use numcodecs::StaticCodec;
use numcodecs_pco::{PcoCompressionLevel, PcoDeltaSpec, PcoModeSpec, PcoPagingSpec, Pcodec};
use serde::Deserialize;
use serde_json::json;

#[test]
#[should_panic(expected = "missing field `level`")]
fn empty_config() {
    let _ = Pcodec::from_config(Deserialize::deserialize(json!({})).unwrap());
}

#[test]
fn minimal_config() {
    let codec = Pcodec::from_config(
        Deserialize::deserialize(json!({
            "level": 0,
            "mode": "classic",
            "delta": "none",
            "paging": "equal-pages-up-to",
            "equal_pages_up_to": 1000,
        }))
        .unwrap(),
    );

    assert_eq!(codec.level, PcoCompressionLevel::Level0);
    assert_eq!(codec.mode, PcoModeSpec::Classic);
    assert_eq!(codec.delta, PcoDeltaSpec::None);
    assert_eq!(
        codec.paging,
        PcoPagingSpec::EqualPagesUpTo {
            equal_pages_up_to: NonZero::new(1000).unwrap()
        }
    );
}

#[test]
fn mode_config() {
    let _ = Pcodec::from_config(
        Deserialize::deserialize(json!({
            "level": 1,
            "mode": "auto",
            "delta": "none",
            "paging": "equal-pages-up-to",
        }))
        .unwrap(),
    );

    let _ = Pcodec::from_config(
        Deserialize::deserialize(json!({
            "level": 2,
            "mode": "classic",
            "delta": "none",
            "paging": "equal-pages-up-to",
        }))
        .unwrap(),
    );

    let _ = Pcodec::from_config(
        Deserialize::deserialize(json!({
            "level": 3,
            "mode": "try-float-mult",
            "float_mult_base": 42.0,
            "delta": "none",
            "paging": "equal-pages-up-to",
        }))
        .unwrap(),
    );

    let _ = Pcodec::from_config(
        Deserialize::deserialize(json!({
            "level": 3,
            "mode": "try-float-quant",
            "float_quant_bits": 12,
            "delta": "none",
            "paging": "equal-pages-up-to",
        }))
        .unwrap(),
    );

    let _ = Pcodec::from_config(
        Deserialize::deserialize(json!({
            "level": 4,
            "mode": "try-int-mult",
            "int_mult_base": 24,
            "delta": "none",
            "paging": "equal-pages-up-to",
        }))
        .unwrap(),
    );
}

#[test]
fn delta_config() {
    let _ = Pcodec::from_config(
        Deserialize::deserialize(json!({
            "level": 5,
            "mode": "classic",
            "delta": "auto",
            "paging": "equal-pages-up-to",
        }))
        .unwrap(),
    );

    let _ = Pcodec::from_config(
        Deserialize::deserialize(json!({
            "level": 6,
            "mode": "classic",
            "delta": "none",
            "paging": "equal-pages-up-to",
        }))
        .unwrap(),
    );

    let _ = Pcodec::from_config(
        Deserialize::deserialize(json!({
            "level": 7,
            "mode": "classic",
            "delta": "try-consecutive",
            "delta_encoding_order": 0,
            "paging": "equal-pages-up-to",
        }))
        .unwrap(),
    );

    let _ = Pcodec::from_config(
        Deserialize::deserialize(json!({
            "level": 8,
            "mode": "classic",
            "delta": "try-lookback",
            "paging": "equal-pages-up-to",
        }))
        .unwrap(),
    );
}

#[test]
fn paging_config() {
    let codec = Pcodec::from_config(
        Deserialize::deserialize(json!({
            "level": 9,
            "mode": "classic",
            "delta": "none",
            "paging": "equal-pages-up-to",
        }))
        .unwrap(),
    );

    assert_eq!(
        codec.paging,
        PcoPagingSpec::EqualPagesUpTo {
            equal_pages_up_to: NonZeroUsize::new(pco::DEFAULT_MAX_PAGE_N).unwrap()
        }
    );

    let codec = Pcodec::from_config(
        Deserialize::deserialize(json!({
            "level": 10,
            "mode": "classic",
            "delta": "none",
            "paging": "equal-pages-up-to",
            "equal_pages_up_to": 1000,
        }))
        .unwrap(),
    );

    assert_eq!(
        codec.paging,
        PcoPagingSpec::EqualPagesUpTo {
            equal_pages_up_to: NonZero::new(1000).unwrap()
        }
    );
}
