//! # EBCC Rust Bindings
//! 
//! This crate provides Rust bindings for EBCC (Error Bounded Climate Compressor),
//! a multi-layer compression algorithm for scientific data that combines JPEG2000 
//! base compression with optional wavelet-based residual compression.
//! 
//! ## Features
//! 
//! - Safe Rust API wrapping the C library
//! - Integration with the `numcodecs` crate for array compression
//! - Support for multiple compression modes and error bounds
//! - Configurable logging and error handling
//! 
//! ## Examples
//! 
//! ```rust,no_run
//! use numcodecs_ebcc::{EBCCConfig, ResidualType, encode_climate_variable, decode_climate_variable};
//! use ndarray::Array2;
//! 
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a 2D array of climate data
//!     let data = Array2::<f32>::zeros((721, 1440)); // ERA5-like dimensions
//!     
//!     // Configure the codec
//!     let config = EBCCConfig {
//!         dims: [1, 721, 1440],
//!         base_cr: 30.0,
//!         residual_compression_type: ResidualType::MaxError,
//!         error: 0.01,
//!     };
//!     
//!     // Compress the data
//!     let compressed = encode_climate_variable(data.as_slice().unwrap(), &config)?;
//!     
//!     // Decompress the data
//!     let decompressed = decode_climate_variable(&compressed)?;
//!     
//!     Ok(())
//! }
//! ```

pub mod error;
pub mod ffi;
pub mod config;
pub mod codec;

pub mod numcodecs_impl;

// Re-export main types and functions
pub use config::{EBCCConfig, ResidualType};
pub use codec::{encode_climate_variable, decode_climate_variable};
pub use error::{EBCCError, EBCCResult};

pub use numcodecs_impl::{EBCCCodec, EBCCCodecError, ebcc_codec_from_config};

/// Initialize logging from environment variables.
/// 
/// This function sets the log level based on the `EBCC_LOG_LEVEL` environment variable.
/// The log levels are: 0=TRACE, 1=DEBUG, 2=INFO, 3=WARN, 4=ERROR, 5=FATAL.
/// 
/// In debug builds, the default level is TRACE (0). In release builds, it's WARN (3).
pub fn init_logging() {
    unsafe {
        ffi::log_set_level_from_env();
    }
}