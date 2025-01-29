use std::sync::OnceLock;

use semver::Version;
use wasm_component_layer::{InterfaceIdentifier, PackageIdentifier, PackageName, Value};

use crate::error::{GuestError, RuntimeError};

#[non_exhaustive]
pub struct CodecPluginInterfaces {
    pub codecs: InterfaceIdentifier,
    pub perf: InterfaceIdentifier,
    pub instruction_counter: String,
}

impl CodecPluginInterfaces {
    #[must_use]
    pub fn get() -> &'static Self {
        static CODEC_PLUGIN_INTERFACES: OnceLock<CodecPluginInterfaces> = OnceLock::new();

        CODEC_PLUGIN_INTERFACES.get_or_init(|| Self {
            codecs: InterfaceIdentifier::new(
                PackageIdentifier::new(
                    PackageName::new("numcodecs", "abc"),
                    Some(Version::new(0, 1, 1)),
                ),
                "codec",
            ),
            perf: InterfaceIdentifier::new(
                PackageIdentifier::new(
                    PackageName::new("fcbench", "perf"),
                    Some(Version::new(0, 1, 0)),
                ),
                "perf",
            ),
            instruction_counter: String::from("instruction-counter"),
        })
    }
}

pub fn guest_error_from_wasm(
    err: Option<&Value>,
) -> Result<GuestError, RuntimeError> {
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

    Ok(GuestError::new(message, chain))
}
