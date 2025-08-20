//! Configuration types for EBCC compression.

use serde::{Deserialize, Serialize};
use crate::error::{EBCCError, EBCCResult};
use crate::ffi;

use schemars::JsonSchema;

/// The number of dimensions supported by EBCC (matches NDIMS from C header).
pub const NDIMS: usize = 3;

/// Residual compression types supported by EBCC.
#[derive(JsonSchema, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResidualType {
    /// No residual compression - base JPEG2000 only
    None,
    /// Residual compression with absolute maximum error bound
    MaxError,
    /// Residual compression with relative error bound
    RelativeError,
}

impl From<ResidualType> for ffi::residual_t::Type {
    fn from(rt: ResidualType) -> Self {
        match rt {
            ResidualType::None => ffi::residual_t::NONE,
            ResidualType::MaxError => ffi::residual_t::MAX_ERROR,
            ResidualType::RelativeError => ffi::residual_t::RELATIVE_ERROR,
        }
    }
}

impl From<ffi::residual_t::Type> for ResidualType {
    fn from(rt: ffi::residual_t::Type) -> Self {
        match rt {
            ffi::residual_t::NONE => ResidualType::None,
            ffi::residual_t::MAX_ERROR => ResidualType::MaxError,
            ffi::residual_t::RELATIVE_ERROR => ResidualType::RelativeError,
            // Deprecated types map to None for backward compatibility
            ffi::residual_t::SPARSIFICATION_FACTOR | ffi::residual_t::QUANTILE => ResidualType::None,
            _ => ResidualType::None, // Default case for unknown values
        }
    }
}

/// Configuration for EBCC compression.
/// 
/// This struct mirrors the `codec_config_t` struct from the C library.
#[derive(JsonSchema, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EBCCConfig {
    /// Data dimensions [frames, height, width] - must be exactly 3 dimensions
    pub dims: [usize; NDIMS],
    
    /// Base compression ratio for JPEG2000 layer
    pub base_cr: f32,
    
    /// Type of residual compression to apply
    pub residual_compression_type: ResidualType,
    
    /// Maximum allowed error (used with MaxError and RelativeError)
    pub error: f32,
}

impl EBCCConfig {
    /// Create a new EBCC configuration with default values.
    pub fn new(dims: [usize; NDIMS]) -> Self {
        Self {
            dims,
            base_cr: 10.0,
            residual_compression_type: ResidualType::None,
            error: 0.01,
        }
    }
    
    /// Create a configuration for JPEG2000-only compression.
    pub fn jpeg2000_only(dims: [usize; NDIMS], compression_ratio: f32) -> Self {
        Self {
            dims,
            base_cr: compression_ratio,
            residual_compression_type: ResidualType::None,
            error: 0.0,
        }
    }
    
    /// Create a configuration for maximum error bounded compression.
    pub fn max_error_bounded(
        dims: [usize; NDIMS],
        base_cr: f32,
        max_error: f32,
    ) -> Self {
        Self {
            dims,
            base_cr,
            residual_compression_type: ResidualType::MaxError,
            error: max_error,
        }
    }
    
    /// Create a configuration for relative error bounded compression.
    pub fn relative_error_bounded(
        dims: [usize; NDIMS],
        base_cr: f32,
        relative_error: f32,
    ) -> Self {
        Self {
            dims,
            base_cr,
            residual_compression_type: ResidualType::RelativeError,
            error: relative_error,
        }
    }
    
    /// Validate the configuration parameters.
    pub fn validate(&self) -> EBCCResult<()> {
        // Check dimensions
        if self.dims.iter().any(|&d| d == 0) {
            return Err(EBCCError::invalid_config("All dimensions must be > 0"));
        }
        
        // Check total size doesn't overflow
        let total_elements = self.dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| EBCCError::invalid_config("Dimension overflow"))?;
        
        if total_elements > (isize::MAX as usize) / std::mem::size_of::<f32>() {
            return Err(EBCCError::invalid_config("Data too large"));
        }
        
        // EBCC requires last two dimensions to be at least 32x32
        if self.dims[1] < 32 || self.dims[2] < 32 {
            return Err(EBCCError::invalid_config(
                format!("EBCC requires last two dimensions to be at least 32x32, got {}x{}", 
                        self.dims[1], self.dims[2])
            ));
        }
        
        // Check compression ratio
        if self.base_cr <= 0.0 {
            return Err(EBCCError::invalid_config("Base compression ratio must be > 0"));
        }
        
        // Check residual-specific parameters
        match self.residual_compression_type {
            ResidualType::MaxError | ResidualType::RelativeError => {
                if self.error <= 0.0 {
                    return Err(EBCCError::invalid_config("Error bound must be > 0"));
                }
            }
            ResidualType::None => {
                // No additional validation needed
            }
        }
        
        Ok(())
    }
    
    /// Get the total number of elements in the data array.
    pub fn total_elements(&self) -> usize {
        self.dims.iter().product()
    }
    
    /// Convert to the C FFI configuration struct.
    pub(crate) fn to_ffi(&self) -> ffi::codec_config_t {
        ffi::codec_config_t {
            dims: self.dims,
            base_cr: self.base_cr,
            residual_compression_type: self.residual_compression_type.into(),
            residual_cr: 1.0, // Default value for removed field
            error: self.error,
            quantile: 1e-6, // Default value for removed field
        }
    }
    
    /// Create from a C FFI configuration struct.
    #[allow(dead_code)]
    pub(crate) fn from_ffi(config: &ffi::codec_config_t) -> Self {
        Self {
            dims: config.dims,
            base_cr: config.base_cr,
            residual_compression_type: config.residual_compression_type.into(),
            error: config.error,
            // Note: residual_cr and quantile are removed from the Rust struct
        }
    }
}