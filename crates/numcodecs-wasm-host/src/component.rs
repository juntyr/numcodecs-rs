use std::sync::Arc;

use schemars::Schema;
use serde::Deserializer;
use wasm_component_layer::{
    AsContextMut, ComponentList, ExportInstance, Func, Instance, TypedFunc, Value,
};

use crate::{
    codec::WasmCodec,
    error::RuntimeError,
    wit::{guest_error_from_wasm, NumcodecsWitInterfaces},
};

/// WebAssembly component that exports the `numcodecs:abc/codec` interface.
///
/// `WasmCodecComponent` does not implement the
/// [`DynCodecType`][numcodecs::DynCodecType] trait itself so that it can expose
/// un-opinionated bindings. However, it provides methods with that can be used
/// to implement the trait on a wrapper.
pub struct WasmCodecComponent {
    // precomputed properties
    pub(crate) codec_id: Arc<str>,
    pub(crate) codec_config_schema: Arc<Schema>,
    // wit functions
    // FIXME: make typed instead
    pub(crate) from_config: Func,
    pub(crate) encode: Func,
    pub(crate) decode: Func,
    pub(crate) decode_into: Func,
    pub(crate) get_config: Func,
    // wasm component instance
    pub(crate) instance: Instance,
}

impl WasmCodecComponent {
    // NOTE: the WasmCodecComponent never calls Instance::drop
    /// Import the `numcodecs:abc/codec` interface from a WebAssembly component
    /// `instance`.
    ///
    /// The `ctx` must refer to the same store in which the `instance` was
    /// instantiated.
    ///
    /// # Warning
    /// The `WasmCodecComponent` does *not* own the provided `instance` and
    /// *never* calls [`Instance::drop`]. It is the responsibility of the code
    /// creating the `WasmCodecComponent` to destroy the `instance` after the
    /// component, and all codecs created from it, have been destroyed.
    ///
    /// # Errors
    ///
    /// Errors if the `instance` does not export the `numcodecs:abc/codec`
    /// interface or if interacting with the component fails.
    pub fn new(mut ctx: impl AsContextMut, instance: Instance) -> Result<Self, RuntimeError> {
        fn load_func(interface: &ExportInstance, name: &str) -> Result<Func, RuntimeError> {
            let Some(func) = interface.func(name) else {
                return Err(RuntimeError::from(anyhow::Error::msg(format!(
                    "WASM component interface does not contain a function named `{name}`"
                ))));
            };

            Ok(func)
        }

        fn load_typed_func<P: ComponentList, R: ComponentList>(
            interface: &ExportInstance,
            name: &str,
        ) -> Result<TypedFunc<P, R>, RuntimeError> {
            load_func(interface, name)?
                .typed()
                .map_err(RuntimeError::from)
        }

        let interfaces = NumcodecsWitInterfaces::get();

        let Some(codecs_interface) = instance.exports().instance(&interfaces.codec) else {
            return Err(RuntimeError::from(anyhow::Error::msg(format!(
                "WASM component does not contain an interface named `{}`",
                interfaces.codec
            ))));
        };

        let codec_id = load_typed_func(codecs_interface, "codec-id")?;
        let codec_id = codec_id.call(&mut ctx, ())?;

        let codec_config_schema = load_typed_func(codecs_interface, "codec-config-schema")?;
        let codec_config_schema: Arc<str> = codec_config_schema.call(&mut ctx, ())?;
        let codec_config_schema: Schema =
            serde_json::from_str(&codec_config_schema).map_err(anyhow::Error::new)?;

        Ok(Self {
            codec_id,
            codec_config_schema: Arc::new(codec_config_schema),
            from_config: load_func(codecs_interface, "[static]codec.from-config")?,
            encode: load_func(codecs_interface, "[method]codec.encode")?,
            decode: load_func(codecs_interface, "[method]codec.decode")?,
            decode_into: load_func(codecs_interface, "[method]codec.decode-into")?,
            get_config: load_func(codecs_interface, "[method]codec.get-config")?,
            instance,
        })
    }
}

/// Methods for implementing the [`DynCodecType`][numcodecs::DynCodecType] trait
impl WasmCodecComponent {
    #[allow(clippy::missing_const_for_fn)] // FIXME: false positive, has Arc deref
    /// Codec identifier.
    #[must_use]
    pub fn codec_id(&self) -> &str {
        &self.codec_id
    }

    #[allow(clippy::missing_const_for_fn)] // FIXME: false positive, has Arc deref
    /// JSON schema for the codec's configuration.
    #[must_use]
    pub fn codec_config_schema(&self) -> &Schema {
        &self.codec_config_schema
    }

    /// Instantiate a codec of this type from a serialized `config`uration.
    ///
    /// The `config` must *not* contain an `id` field. If the `config` *may*
    /// contain one, use the
    /// [`codec_from_config_with_id`][numcodecs::codec_from_config_with_id]
    /// helper function.
    ///
    /// The `config` *must* be compatible with JSON encoding.
    ///
    /// # Errors
    ///
    /// Errors if constructing the codec or interacting with the component
    /// fails.
    pub fn codec_from_config<'de, D: Deserializer<'de>>(
        &self,
        mut ctx: impl AsContextMut,
        config: D,
    ) -> Result<WasmCodec, D::Error> {
        let mut config_bytes = Vec::new();
        serde_transcode::transcode(config, &mut serde_json::Serializer::new(&mut config_bytes))
            .map_err(serde::de::Error::custom)?;
        let config = String::from_utf8(config_bytes).map_err(serde::de::Error::custom)?;

        let args = Value::String(config.into());
        let mut result = Value::U8(0);

        self.from_config
            .call(
                &mut ctx,
                std::slice::from_ref(&args),
                std::slice::from_mut(&mut result),
            )
            .map_err(serde::de::Error::custom)?;

        let codec = match result {
            Value::Result(result) => match &*result {
                Ok(Some(Value::Own(resource))) => WasmCodec {
                    resource: resource.clone(),
                    codec_id: self.codec_id.clone(),
                    codec_config_schema: self.codec_config_schema.clone(),
                    from_config: self.from_config.clone(),
                    encode: self.encode.clone(),
                    decode: self.decode.clone(),
                    decode_into: self.decode_into.clone(),
                    get_config: self.get_config.clone(),
                    instance: self.instance.clone(),
                },
                Err(err) => match guest_error_from_wasm(err.as_ref()) {
                    Ok(err) => return Err(serde::de::Error::custom(err)),
                    Err(err) => return Err(serde::de::Error::custom(err)),
                },
                result => {
                    return Err(serde::de::Error::custom(format!(
                        "unexpected from-config result value {result:?}"
                    )))
                }
            },
            value => {
                return Err(serde::de::Error::custom(format!(
                    "unexpected from-config result value {value:?}"
                )))
            }
        };

        Ok(codec)
    }
}
