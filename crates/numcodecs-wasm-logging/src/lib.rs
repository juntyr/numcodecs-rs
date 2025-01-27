//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.82.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-wasm-logging
//! [crates.io]: https://crates.io/crates/numcodecs-wasm-logging
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-wasm-logging
//! [docs.rs]: https://docs.rs/numcodecs-wasm-logging/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_wasm_logging
//!
//! Enable logging for wasm32-compiled [`StaticCodec`]s for the [`numcodecs`]
//! API.

use std::sync::Once;

use numcodecs::{
    AnyArray, AnyArrayView, AnyArrayViewMut, AnyCowArray, Codec, StaticCodec, StaticCodecConfig,
};

#[derive(Clone)]
/// Wrapper for a [`StaticCodec`] that automatically installs a logger.
pub struct LoggingCodec<T: StaticCodec>(pub T);

impl<T: StaticCodec> Codec for LoggingCodec<T> {
    type Error = T::Error;

    #[inline]
    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        ensure_logger();

        self.0.encode(data)
    }

    #[inline]
    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        ensure_logger();

        self.0.decode(encoded)
    }

    #[inline]
    fn decode_into(
        &self,
        encoded: AnyArrayView,
        decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        ensure_logger();

        self.0.decode_into(encoded, decoded)
    }
}

impl<T: StaticCodec> StaticCodec for LoggingCodec<T> {
    type Config<'de> = T::Config<'de>;

    const CODEC_ID: &'static str = T::CODEC_ID;

    #[inline]
    fn from_config(config: Self::Config<'_>) -> Self {
        ensure_logger();

        Self(T::from_config(config))
    }

    #[inline]
    fn get_config(&self) -> StaticCodecConfig<Self> {
        ensure_logger();

        StaticCodecConfig::new(self.0.get_config().config)
    }
}

// The logger init could also be implemented using a no-mangle `_initialize`
// function, but that would require unsafe and be less explicit
// https://github.com/bytecodealliance/wasm-tools/pull/1747
fn ensure_logger() {
    static LOGGER_INIT: Once = Once::new();

    LOGGER_INIT.call_once(|| {
        #[expect(clippy::expect_used)]
        // failing to install the logger is a bug and we cannot continue
        wasi_logger::Logger::install().expect("failed to install wasi_logger::Logger");

        log::set_max_level(log::LevelFilter::Trace);
    });
}
