#![allow(missing_docs)] // FIXME
#![allow(clippy::missing_errors_doc)] // FIXME
#![allow(clippy::multiple_crate_versions)] // FIXME

#[macro_use]
extern crate log;

mod codec;
mod engine;
mod logging;
mod stdio;
mod transform;

pub use codec::{ReproducibleWasmCodec, ReproducibleWasmCodecError, ReproducibleWasmCodecType};
