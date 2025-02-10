use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
#[error(transparent)]
/// Opaque error type for errors that occur within the WebAssembly runtime.
pub struct RuntimeError(#[from] anyhow::Error);

#[derive(Debug, thiserror::Error)]
#[error("{msg}")]
/// Opaque error type for errors that occur within the codec implementations
/// inside the WebAssembly component.
///
/// The error preserves the complete causality chain, i.e. calling
/// [`std::error::Error::source`] works, though the types in the chain are
/// erased.
pub struct CodecError {
    msg: Arc<str>,
    source: Option<Box<CodecError>>,
}

impl CodecError {
    pub(crate) fn new(message: Arc<str>, chain: Vec<Arc<str>>) -> Self {
        let mut root = Self {
            msg: message,
            source: None,
        };

        let mut err = &mut root;

        for msg in chain {
            err = &mut *err.source.insert(Box::new(Self { msg, source: None }));
        }

        root
    }
}
