use std::sync::OnceLock;

use semver::Version;
use wasm_component_layer::{InterfaceIdentifier, PackageIdentifier, PackageName, Value};

use crate::error::{CodecError, RuntimeError};

/// WebAssembly Interface Type (WIT) interfaces for `numcodecs`
#[non_exhaustive]
pub struct NumcodecsWitInterfaces {
    /// The `numcodecs:abc/codec` interface
    pub codec: InterfaceIdentifier,
}

impl NumcodecsWitInterfaces {
    /// Get the once-computed interfaces
    #[must_use]
    pub fn get() -> &'static Self {
        static NUMCODECS_WIT_INTERFACES: OnceLock<NumcodecsWitInterfaces> = OnceLock::new();

        NUMCODECS_WIT_INTERFACES.get_or_init(|| Self {
            codec: InterfaceIdentifier::new(
                PackageIdentifier::new(
                    PackageName::new("numcodecs", "abc"),
                    Some(Version::new(0, 1, 1)),
                ),
                "codec",
            ),
        })
    }
}

pub fn guest_error_from_wasm(err: Option<&Value>) -> Result<CodecError, RuntimeError> {
    let Some(Value::Record(record)) = err else {
        return Err(RuntimeError::from(anyhow::anyhow!(
            "unexpected err value {err:?}"
        )));
    };

    let Some(Value::String(message)) = record.field("message") else {
        return Err(RuntimeError::from(anyhow::anyhow!(
            "numcodecs:abc/codec::error is missing the `message` field"
        )));
    };

    let Some(Value::List(chain)) = record.field("chain") else {
        return Err(RuntimeError::from(anyhow::anyhow!(
            "numcodecs:abc/codec::error is missing the `chain` field"
        )));
    };

    let Ok(chain) = chain
        .iter()
        .map(|msg| match msg {
            Value::String(msg) => Ok(msg),
            _ => Err(()),
        })
        .collect::<Result<Vec<_>, _>>()
    else {
        return Err(RuntimeError::from(anyhow::anyhow!(
            "numcodecs:abc/codec::error chain contains unexpected non-string values: {chain:?}"
        )));
    };

    Ok(CodecError::new(message, chain))
}
