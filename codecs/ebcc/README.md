# EBCC Rust Bindings

This directory contains Rust bindings for EBCC (Error Bounded Climate Compressor), providing a safe and efficient interface to the EBCC compression library with integration support for the `numcodecs` ecosystem.

## Features

- **Safe Rust API**: Memory-safe wrappers around the C library with automatic error handling
- **numcodecs Integration**: Compatible with the Rust numcodecs ecosystem for array compression
- **Multiple Compression Modes**: Support for JPEG2000-only and error-bounded compression


## Quick Start


### Basic Usage

```rust
use numcodecs_ebcc::{encode_climate_variable, decode_climate_variable, EBCCConfig, ResidualType};

// Create climate data (e.g., ERA5-like temperature field)
let data = vec![273.15; 721 * 1440]; // 721x1440 grid at 0°C

// Configure compression with 0.01K maximum error bound
let config = EBCCConfig::max_error_bounded([1, 721, 1440], 30.0, 0.01);

// Compress the data
let compressed = encode_climate_variable(&data, &config)?;
println!("Compressed {} bytes to {} bytes", 
         data.len() * 4, compressed.len());

// Decompress the data
let decompressed = decode_climate_variable(&compressed)?;
assert_eq!(data.len(), decompressed.len());
```

### Configuration Types

```rust
use numcodecs_ebcc::{EBCCConfig, ResidualType};

// JPEG2000-only compression
let config = EBCCConfig::jpeg2000_only([1, 721, 1440], 20.0);

// Maximum absolute error bound (e.g., 0.1 Kelvin)
let config = EBCCConfig::max_error_bounded([1, 721, 1440], 15.0, 0.1);

// Relative error bound (e.g., 0.1% of data range)
let config = EBCCConfig::relative_error_bounded([1, 721, 1440], 15.0, 0.001);

// Custom configuration
let config = EBCCConfig {
    dims: [2, 721, 1440],  // 2 time steps, 721x1440 spatial grid
    base_cr: 25.0,         // JPEG2000 compression ratio
    residual_compression_type: ResidualType::MaxError,
    error: 0.05,           // 0.05 unit maximum error
};
```

## Build System

The Rust bindings use CMake to build the underlying C library as a static library (`ebcc.a`) that includes statically linked OpenJPEG and Zstd dependencies.

### Building

```bash
# Build with default features (debug)
cargo build
# or release mode
cargo build --release

# Build with bindgen (regenerates C bindings)
cargo build --features bindgen

### Testing

```bash
# Run all tests
cargo test

# Run tests with logging
EBCC_LOG_LEVEL=2 cargo test

# Run integration tests only
cargo test --test integration_tests

# Run with bindgen feature
cargo test --features bindgen
```

### Examples

```bash
# Basic compression example
cargo run --example basic_compression

# numcodecs integration
cargo run --example numcodecs_integration
```

## API Documentation

### Core Functions

#### `encode_climate_variable(data: &[f32], config: &EBCCConfig) -> EBCCResult<Vec<u8>>`

Compresses climate data using EBCC.

**Parameters:**
- `data`: Input data as f32 slice
- `config`: Compression configuration

**Returns:** Compressed data bytes

#### `decode_climate_variable(compressed_data: &[u8]) -> EBCCResult<Vec<f32>>`

Decompresses EBCC-compressed data.

**Parameters:**
- `compressed_data`: Compressed bytes from `encode_climate_variable`

**Returns:** Decompressed f32 values

### Configuration

#### `EBCCConfig`

Main configuration struct with the following fields:

- `dims: [usize; 3]` - Data dimensions as [frames, height, width]
- `base_cr: f32` - Base JPEG2000 compression ratio
- `residual_compression_type: ResidualType` - Type of residual compression
- `error: f32` - Error bound (for error-bounded modes)

#### `ResidualType`

Compression modes:
- `None` - JPEG2000 only
- `MaxError` - Absolute error bound
- `RelativeError` - Relative error bound

### Error Handling

All functions return `EBCCResult<T>` which is `Result<T, EBCCError>`. Error types include:

- `InvalidInput` - Invalid input data (NaN, wrong size, etc.)
- `InvalidConfig` - Invalid configuration parameters
- `CompressionError` - Compression failed
- `DecompressionError` - Decompression failed
- `MemoryError` - Memory allocation failed

### numcodecs Integration


```rust
use numcodecs_ebcc::numcodecs_impl::{EBCCCodec, ebcc_codec_from_config};
use std::collections::HashMap;

// Create codec directly
let config = EBCCConfig::max_error_bounded([1, 100, 100], 20.0, 0.1);
let codec = EBCCCodec::new(config)?;

// Create codec from configuration map
let mut config_map = HashMap::new();
config_map.insert("dims".to_string(), serde_json::json!([1, 100, 100]));
config_map.insert("base_cr".to_string(), serde_json::json!(20.0));
config_map.insert("residual_type".to_string(), serde_json::json!("max_error"));
config_map.insert("error".to_string(), serde_json::json!(0.1));

let codec = ebcc_codec_from_config(config_map)?;
```

## Environment Variables

- `EBCC_LOG_LEVEL` - Set log level (0=TRACE, 1=DEBUG, 2=INFO, 3=WARN, 4=ERROR, 5=FATAL)
- `EBCC_INIT_BASE_ERROR_QUANTILE` - Initial base error quantile (default: 1e-6)
- `EBCC_DISABLE_PURE_BASE_COMPRESSION_FALLBACK` - Disable pure JPEG2000 fallback
- `EBCC_DISABLE_MEAN_ADJUSTMENT` - Disable mean error adjustment

## Architecture

```
┌─────────────────────┐
│   User Application  │
├─────────────────────┤
│   numcodecs API     │  ← Codec + StaticCodec traits
├─────────────────────┤
│  Safe Rust Wrapper  │  ← Memory management, error handling
├─────────────────────┤
│    Raw C Bindings   │  ← Generated by bindgen
├─────────────────────┤
│     ebcc.a          │  ← Static library (OpenJPEG + Zstd + SPIHT)
└─────────────────────┘
```

## Contributing

### Development Setup

1. Install Rust toolchain
2. Install CMake and C compiler (clang)
3. Clone repository with submodules:
   ```bash
   git clone --recurse-submodules <repository>
   ```
4. Build Rust bindings:
   ```bash
   cargo build --features bindgen
   ```

### Testing

- Run `cargo test` for unit tests 
- Use `EBCC_LOG_LEVEL=0` for verbose logging during development

## Citation

If you use EBCC in your research, please cite the original paper and software.
