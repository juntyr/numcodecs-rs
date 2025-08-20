//! Safe wrapper functions for EBCC compression and decompression.

use std::ptr;
use std::slice;
use crate::config::EBCCConfig;
use crate::error::{EBCCError, EBCCResult};
use crate::ffi;

/// Encode climate variable data using EBCC compression.
/// 
/// # Arguments
/// 
/// * `data` - Input data as a slice of f32 values
/// * `config` - EBCC configuration parameters
/// 
/// # Returns
/// 
/// A vector containing the compressed data bytes.
/// 
/// # Errors
/// 
/// Returns an error if:
/// - Configuration is invalid
/// - Input data size doesn't match configuration dimensions
/// - Compression fails
/// - Memory allocation fails
/// 
/// # Examples
/// 
/// ```rust,no_run
/// use numcodecs_ebcc::{encode_climate_variable, EBCCConfig, ResidualType};
/// 
/// // 2D ERA5-like data: 721x1440
/// let data = vec![0.0f32; 721 * 1440];
/// let config = EBCCConfig::max_error_bounded([1, 721, 1440], 30.0, 0.01);
/// 
/// let compressed = encode_climate_variable(&data, &config)?;
/// println!("Compressed {} bytes to {} bytes", 
///          data.len() * 4, compressed.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn encode_climate_variable(data: &[f32], config: &EBCCConfig) -> EBCCResult<Vec<u8>> {
    // Validate configuration
    config.validate()?;
    
    // Check data size matches configuration
    let expected_size = config.total_elements();
    if data.len() != expected_size {
        return Err(EBCCError::invalid_input(format!(
            "Data size {} doesn't match config dimensions (expected {})",
            data.len(), expected_size
        )));
    }
    
    // Check for NaN or infinity values
    for (i, &value) in data.iter().enumerate() {
        if !value.is_finite() {
            return Err(EBCCError::invalid_input(format!(
                "Non-finite value {} at index {}", value, i
            )));
        }
    }
    
    // Convert to FFI types
    let mut ffi_config = config.to_ffi();
    let mut data_copy = data.to_vec(); // C function may modify the input
    
    // Call the C function
    let mut out_buffer: *mut u8 = ptr::null_mut();
    let compressed_size = unsafe {
        ffi::encode_climate_variable(
            data_copy.as_mut_ptr(),
            &mut ffi_config,
            &mut out_buffer,
        )
    };
    
    // Check for errors
    if compressed_size == 0 || out_buffer.is_null() {
        return Err(EBCCError::compression_error("C function returned null or zero size"));
    }
    
    // Copy the compressed data to a Vec and free the C-allocated memory
    let compressed_data = unsafe {
        let slice = slice::from_raw_parts(out_buffer, compressed_size);
        let vec = slice.to_vec();
        libc::free(out_buffer as *mut libc::c_void);
        vec
    };
    
    Ok(compressed_data)
}

/// Decode climate variable data using EBCC decompression.
/// 
/// # Arguments
/// 
/// * `compressed_data` - Compressed data bytes from `encode_climate_variable`
/// 
/// # Returns
/// 
/// A vector containing the decompressed f32 values.
/// 
/// # Errors
/// 
/// Returns an error if:
/// - Compressed data is invalid or corrupted
/// - Decompression fails
/// - Memory allocation fails
/// 
/// # Examples
/// 
/// ```rust,no_run
/// use numcodecs_ebcc::{encode_climate_variable, decode_climate_variable, EBCCConfig};
/// 
/// let data = vec![1.0f32; 100];
/// let config = EBCCConfig::new([1, 10, 10]);
/// 
/// let compressed = encode_climate_variable(&data, &config)?;
/// let decompressed = decode_climate_variable(&compressed)?;
/// 
/// assert_eq!(data.len(), decompressed.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn decode_climate_variable(compressed_data: &[u8]) -> EBCCResult<Vec<f32>> {
    if compressed_data.is_empty() {
        return Err(EBCCError::invalid_input("Compressed data is empty"));
    }
    
    // Call the C function
    let mut out_buffer: *mut f32 = ptr::null_mut();
    let decompressed_size = unsafe {
        ffi::decode_climate_variable(
            compressed_data.as_ptr() as *mut u8, // C function shouldn't modify input
            compressed_data.len(),
            &mut out_buffer,
        )
    };
    
    // Check for errors
    if decompressed_size == 0 || out_buffer.is_null() {
        return Err(EBCCError::decompression_error("C function returned null or zero size"));
    }
    
    // Copy the decompressed data to a Vec and free the C-allocated memory
    let decompressed_data = unsafe {
        let slice = slice::from_raw_parts(out_buffer, decompressed_size);
        let vec = slice.to_vec();
        libc::free(out_buffer as *mut libc::c_void);
        vec
    };
    
    Ok(decompressed_data)
}

/// Print EBCC configuration details to the log.
/// 
/// This function uses the C library's logging system to print configuration details.
/// The output level depends on the log level set via environment variables or `init_logging()`.
/// 
/// # Arguments
/// 
/// * `config` - Configuration to print
pub fn print_config(config: &EBCCConfig) {
    let mut ffi_config = config.to_ffi();
    unsafe {
        ffi::print_config(&mut ffi_config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_encode_decode_roundtrip() {
        // Create test data for 32x32 minimum size requirement
        let data = vec![1.0f32; 32 * 32];
        let config = EBCCConfig::new([1, 32, 32]);
        
        let compressed = encode_climate_variable(&data, &config).unwrap();
        let decompressed = decode_climate_variable(&compressed).unwrap();
        
        assert_eq!(data.len(), decompressed.len());
        // Note: Due to lossy compression, values may not be exactly equal
    }
    
    #[test]
    fn test_invalid_config() {
        let data = vec![1.0f32; 32 * 32];
        let mut config = EBCCConfig::new([1, 32, 32]);
        config.base_cr = -1.0; // Invalid compression ratio
        
        let result = encode_climate_variable(&data, &config);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_mismatched_data_size() {
        let data = vec![1.0f32; 1025]; // Should be 1024 elements (32*32)
        let config = EBCCConfig::new([1, 32, 32]); // Expects 32*32 = 1024 elements
        
        let result = encode_climate_variable(&data, &config);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_nan_input() {
        let mut data = vec![1.0f32; 32 * 32];
        data[100] = f32::NAN; // Insert NaN in the middle
        let config = EBCCConfig::new([1, 32, 32]);
        
        let result = encode_climate_variable(&data, &config);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_empty_compressed_data() {
        let result = decode_climate_variable(&[]);
        assert!(result.is_err());
    }
}