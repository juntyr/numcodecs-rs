//! Error types for EBCC operations.

use thiserror::Error;

/// Result type for EBCC operations.
pub type EBCCResult<T> = Result<T, EBCCError>;

/// Errors that can occur during EBCC compression/decompression.
#[derive(Error, Debug)]
pub enum EBCCError {
    #[error("Invalid input data: {0}")]
    InvalidInput(String),
    
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    
    #[error("Compression failed: {0}")]
    CompressionError(String),
    
    #[error("Decompression failed: {0}")]
    DecompressionError(String),
    
    #[error("Memory allocation failed")]
    MemoryError,
    
    #[error("Buffer too small: expected at least {expected}, got {actual}")]
    BufferTooSmall { expected: usize, actual: usize },
    
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),
    
    #[error("Null pointer returned from C function")]
    NullPointer,
    
    #[error("Array conversion error: {0}")]
    ArrayError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

impl EBCCError {
    /// Create an InvalidInput error with a formatted message.
    pub fn invalid_input<S: Into<String>>(msg: S) -> Self {
        EBCCError::InvalidInput(msg.into())
    }
    
    /// Create an InvalidConfig error with a formatted message.
    pub fn invalid_config<S: Into<String>>(msg: S) -> Self {
        EBCCError::InvalidConfig(msg.into())
    }
    
    /// Create a CompressionError with a formatted message.
    pub fn compression_error<S: Into<String>>(msg: S) -> Self {
        EBCCError::CompressionError(msg.into())
    }
    
    /// Create a DecompressionError with a formatted message.
    pub fn decompression_error<S: Into<String>>(msg: S) -> Self {
        EBCCError::DecompressionError(msg.into())
    }
}