#![allow(missing_docs)] // FIXME
#![allow(clippy::missing_errors_doc)] // FIXME

// postcard depends on embedded-io 0.4 and 0.6
#![allow(clippy::multiple_crate_versions)]

mod codec;
mod component;
mod error;
mod wit;

pub use codec::WasmCodec;
pub use component::WasmCodecComponent;
pub use error::{GuestError, RuntimeError};
pub use wit::NumcodecsWitInterfaces;
