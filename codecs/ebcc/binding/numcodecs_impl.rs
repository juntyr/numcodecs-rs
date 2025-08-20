//! Implementation of numcodecs traits for EBCC.
//! 
//! This module provides integration with the `numcodecs` crate, allowing EBCC
//! to be used as a compression codec in the numcodecs ecosystem.

use crate::config::{EBCCConfig, ResidualType};
use crate::codec::{encode_climate_variable, decode_climate_variable};
use crate::error::{EBCCError, EBCCResult};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use ndarray::Array;
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig,
};
use schemars::JsonSchema;

// Version tracking for the codec (not needed for this implementation)
const CODEC_VERSION: &str = "0.1.0";

/// EBCC codec implementation for the numcodecs ecosystem.
/// 
/// This struct holds the configuration for EBCC compression and implements
/// the numcodecs codec traits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct EBCCCodec {
    /// EBCC configuration parameters
    #[serde(flatten)]
    pub config: EBCCConfig,
    /// The codec's encoding format version. Do not provide this parameter explicitly.
    #[serde(default, rename = "_version")]
    pub version: String,
}

impl EBCCCodec {
    /// Create a new EBCC codec with the given configuration.
    pub fn new(config: EBCCConfig) -> EBCCResult<Self> {
        config.validate()?;
        Ok(Self { 
            config, 
            version: CODEC_VERSION.to_string(),
        })
    }
    
    /// Create an EBCC codec for JPEG2000-only compression.
    pub fn jpeg2000_only(dims: [usize; 3], compression_ratio: f32) -> EBCCResult<Self> {
        let config = EBCCConfig::jpeg2000_only(dims, compression_ratio);
        Self::new(config)
    }
    
    /// Create an EBCC codec for maximum error bounded compression.
    pub fn max_error_bounded(
        dims: [usize; 3],
        base_cr: f32,
        max_error: f32,
    ) -> EBCCResult<Self> {
        let config = EBCCConfig::max_error_bounded(dims, base_cr, max_error);
        Self::new(config)
    }
    
    /// Create an EBCC codec for relative error bounded compression.
    pub fn relative_error_bounded(
        dims: [usize; 3],
        base_cr: f32,
        relative_error: f32,
    ) -> EBCCResult<Self> {
        let config = EBCCConfig::relative_error_bounded(dims, base_cr, relative_error);
        Self::new(config)
    }
}

impl Codec for EBCCCodec {
    type Error = EBCCCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            AnyCowArray::F32(data) => {
                // Check if data shape matches expected dimensions
                let expected_size = self.config.dims[0] * self.config.dims[1] * self.config.dims[2];
                if data.len() != expected_size {
                    return Err(EBCCCodecError::ShapeMismatch {
                        expected: self.config.dims.to_vec(),
                        actual: vec![data.len()],
                    });
                }
                
                // Check minimum size requirement for EBCC (last two dimensions must be at least 32x32)
                if self.config.dims[1] < 32 || self.config.dims[2] < 32 {
                    return Err(EBCCCodecError::InvalidDimensions {
                        dims: self.config.dims.to_vec(),
                        requirement: "Last two dimensions must be at least 32x32".to_string(),
                    });
                }
                
                let data_slice = data.as_slice().ok_or(EBCCCodecError::NonContiguousArray)?;
                let compressed = encode_climate_variable(data_slice, &self.config)
                    .map_err(|source| EBCCCodecError::CompressionFailed { source })?;
                
                Ok(AnyArray::U8(
                    Array::from(compressed).into_dyn()
                ))
            }
            _ => Err(EBCCCodecError::UnsupportedDtype(data.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        let AnyCowArray::U8(encoded) = encoded else {
            return Err(EBCCCodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(EBCCCodecError::EncodedDataNotOneDimensional {
                shape: encoded.shape().to_vec(),
            });
        }

        let data_slice = encoded.as_slice().ok_or(EBCCCodecError::NonContiguousArray)?;
        
        let decompressed = decode_climate_variable(data_slice)
            .map_err(|source| EBCCCodecError::DecompressionFailed { source })?;
        
        // Reshape to the original dimensions
        Ok(AnyArray::F32(
            Array::from_shape_vec(self.config.dims, decompressed)
                .map_err(|err| EBCCCodecError::ShapeError { message: err.to_string() })?
                .into_dyn()
        ))
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        mut decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        let decoded_data = self.decode(encoded.cow())?;
        Ok(decoded.assign(&decoded_data).map_err(|source| EBCCCodecError::AssignError { source })?)
    }
}

impl StaticCodec for EBCCCodec {
    const CODEC_ID: &'static str = "ebcc.rs";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<'_, Self> {
        StaticCodecConfig::from(self)
    }
}

/// Errors that may occur when applying the [`EBCCCodec`].
#[derive(Debug, thiserror::Error)]
pub enum EBCCCodecError {
    /// EBCC codec does not support the dtype
    #[error("EBCC does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    
    /// EBCC codec failed to encode the header
    #[error("EBCC failed to encode the header")]
    HeaderEncodeFailed {
        /// Opaque source error
        source: postcard::Error,
    },
    
    /// EBCC codec failed to decode the header
    #[error("EBCC failed to decode the header")]
    HeaderDecodeFailed {
        /// Opaque source error
        source: postcard::Error,
    },
    
    /// EBCC codec cannot encode/decode non-contiguous arrays
    #[error("EBCC cannot encode/decode non-contiguous arrays")]
    NonContiguousArray,
    
    /// EBCC codec can only decode one-dimensional byte arrays but received
    /// an array of a different dtype
    #[error(
        "EBCC can only decode one-dimensional byte arrays but received an array of dtype {dtype}"
    )]
    EncodedDataNotBytes {
        /// The unexpected dtype of the encoded array
        dtype: AnyArrayDType,
    },
    
    /// EBCC codec can only decode one-dimensional byte arrays but received
    /// an array of a different shape
    #[error(
        "EBCC can only decode one-dimensional byte arrays but received a byte array of shape {shape:?}"
    )]
    EncodedDataNotOneDimensional {
        /// The unexpected shape of the encoded array
        shape: Vec<usize>,
    },
    
    /// EBCC codec cannot decode into the provided array
    #[error("EBCC cannot decode into the provided array")]
    AssignError {
        /// The source of the error
        source: AnyArrayAssignError,
    },
    
    /// EBCC codec failed during compression
    #[error("EBCC compression failed")]
    CompressionFailed {
        /// The source of the error
        source: EBCCError,
    },
    
    /// EBCC codec failed during decompression
    #[error("EBCC decompression failed")]
    DecompressionFailed {
        /// The source of the error
        source: EBCCError,
    },
    
    /// Data shape mismatch
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape
        expected: Vec<usize>,
        /// Actual shape
        actual: Vec<usize>,
    },
    
    /// Invalid dimensions for EBCC compression
    #[error("Invalid dimensions {dims:?}: {requirement}")]
    InvalidDimensions {
        /// The invalid dimensions
        dims: Vec<usize>,
        /// The requirement that was not met
        requirement: String,
    },
    
    /// Shape error when creating arrays
    #[error("Shape error when creating arrays: {message}")]
    ShapeError {
        /// The error message
        message: String,
    },
}

/// Create an EBCC codec from a configuration dictionary.
/// 
/// This function provides a way to create EBCC codecs from configuration
/// data, similar to how other numcodecs codecs are created.
/// 
/// # Arguments
/// 
/// * `config` - Configuration parameters as key-value pairs
/// 
/// # Configuration Parameters
/// 
/// - `dims`: Array dimensions as [frames, height, width]
/// - `base_cr`: Base JPEG2000 compression ratio (default: 10.0)
/// - `residual_type`: Residual compression type ("none", "max_error", "relative_error")
/// - `error`: Error bound for error-bounded modes (default: 0.01)
/// 
/// # Returns
/// 
/// An EBCC codec configured with the specified parameters.
/// 
/// # Examples
/// 
/// ```rust,no_run
/// use std::collections::HashMap;
/// use numcodecs_ebcc::numcodecs_impl::ebcc_codec_from_config;
/// 
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let mut config = HashMap::new();
///     config.insert("dims".to_string(), serde_json::json!([1, 721, 1440]));
///     config.insert("base_cr".to_string(), serde_json::json!(30.0));
///     config.insert("residual_type".to_string(), serde_json::json!("max_error"));
///     config.insert("error".to_string(), serde_json::json!(0.01));
///     
///     let codec = ebcc_codec_from_config(config)?;
///     Ok(())
/// }
/// ```
pub fn ebcc_codec_from_config(
    config_map: HashMap<String, serde_json::Value>
) -> EBCCResult<EBCCCodec> {
    // Extract dimensions (required)
    let dims_value = config_map.get("dims")
        .ok_or_else(|| EBCCError::invalid_config("Missing required parameter 'dims'"))?;
    
    let dims_array: Vec<usize> = serde_json::from_value(dims_value.clone())
        .map_err(|e| EBCCError::invalid_config(format!("Invalid dims format: {}", e)))?;
    
    if dims_array.len() != 3 {
        return Err(EBCCError::invalid_config("dims must have exactly 3 elements"));
    }
    
    let dims = [dims_array[0], dims_array[1], dims_array[2]];
    
    // Extract other parameters with defaults
    let base_cr = config_map.get("base_cr")
        .and_then(|v| v.as_f64())
        .unwrap_or(10.0) as f32;
    
    let residual_type_str = config_map.get("residual_type")
        .and_then(|v| v.as_str())
        .unwrap_or("none");
    
    let residual_type = match residual_type_str {
        "none" => ResidualType::None,
        "max_error" => ResidualType::MaxError,
        "relative_error" => ResidualType::RelativeError,
        // Deprecated types are ignored and default to None
        "sparsification" | "quantile" => {
            return Err(EBCCError::invalid_config(format!(
                "Residual type '{}' is deprecated and no longer supported", residual_type_str
            )));
        },
        _ => return Err(EBCCError::invalid_config(format!(
            "Unknown residual type: {}", residual_type_str
        ))),
    };
    
    let error = config_map.get("error")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.01) as f32;
    
    let config = EBCCConfig {
        dims,
        base_cr,
        residual_compression_type: residual_type,
        error,
    };
    
    EBCCCodec::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use numcodecs::Codec;
    use ndarray::Array1;
    
    #[test]
    fn test_codec_creation() {
        let config = EBCCConfig::new([1, 32, 32]);
        let codec = EBCCCodec::new(config).unwrap();
        assert_eq!(codec.config.dims, [1, 32, 32]);
    }
    
    #[test]
    fn test_codec_from_config() {
        let mut config_map = HashMap::new();
        config_map.insert("dims".to_string(), serde_json::json!([1, 32, 32]));
        config_map.insert("base_cr".to_string(), serde_json::json!(15.0));
        config_map.insert("residual_type".to_string(), serde_json::json!("max_error"));
        config_map.insert("error".to_string(), serde_json::json!(0.05));
        
        let codec = ebcc_codec_from_config(config_map).unwrap();
        assert_eq!(codec.config.dims, [1, 32, 32]);
        assert_eq!(codec.config.base_cr, 15.0);
        assert_eq!(codec.config.residual_compression_type, ResidualType::MaxError);
        assert_eq!(codec.config.error, 0.05);
    }
    
    #[test]
    fn test_missing_dims() {
        let config_map = HashMap::new();
        let result = ebcc_codec_from_config(config_map);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_invalid_residual_type() {
        let mut config_map = HashMap::new();
        config_map.insert("dims".to_string(), serde_json::json!([1, 32, 32]));
        config_map.insert("residual_type".to_string(), serde_json::json!("invalid"));
        
        let result = ebcc_codec_from_config(config_map);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_deprecated_residual_types() {
        let mut config_map = HashMap::new();
        config_map.insert("dims".to_string(), serde_json::json!([1, 32, 32]));
        
        // Test sparsification is rejected
        config_map.insert("residual_type".to_string(), serde_json::json!("sparsification"));
        let result = ebcc_codec_from_config(config_map.clone());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("deprecated"));
        
        // Test quantile is rejected
        config_map.insert("residual_type".to_string(), serde_json::json!("quantile"));
        let result = ebcc_codec_from_config(config_map);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("deprecated"));
    }
    
    #[test]
    fn test_unsupported_dtype() {
        let config = EBCCConfig::new([1, 32, 32]);
        let codec = EBCCCodec::new(config).unwrap();
        
        let data = Array1::<i32>::zeros(100);
        let result = codec.encode(AnyCowArray::I32(data.into_dyn().into()));
        
        assert!(matches!(result, Err(EBCCCodecError::UnsupportedDtype(_))));
    }
    
    #[test]
    fn test_invalid_dimensions() {
        // Test dimensions too small (16x16 < 32x32 requirement)
        let result = EBCCCodec::new(EBCCConfig::new([1, 16, 16]));
        assert!(result.is_err());
        
        // Test mixed valid/invalid dimensions
        let result = EBCCCodec::new(EBCCConfig::new([1, 32, 16]));
        assert!(result.is_err());
        
        // Test valid dimensions
        let result = EBCCCodec::new(EBCCConfig::new([1, 32, 32]));
        assert!(result.is_ok());
    }
}