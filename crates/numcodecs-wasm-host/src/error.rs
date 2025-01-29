use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct RuntimeError(#[from] anyhow::Error);

#[derive(Debug, thiserror::Error)]
#[error("{msg}")]
pub struct GuestError {
    msg: Arc<str>,
    source: Option<Box<GuestError>>,
}

impl GuestError {
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
