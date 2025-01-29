use std::borrow::BorrowMut;

use schemars::Schema;
use serde::Deserializer;
use wasm_component_layer::{
    ComponentList, ExportInstance, Func, Instance, Store, TypedFunc, Value,
};
use wasm_runtime_layer::backend::WasmEngine;

use crate::{
    codec::WasmCodec,
    error::RuntimeError,
    store::ErasedWasmStore,
    wit::{guest_error_from_wasm, CodecPluginInterfaces},
};

#[expect(clippy::module_name_repetitions)]
pub struct CodecPlugin {
    // FIXME: make typed instead
    from_config: Func,
    pub(crate) encode: Func,
    pub(crate) decode: Func,
    pub(crate) decode_into: Func,
    codec_id: TypedFunc<(), String>,
    codec_config_schema: TypedFunc<(), String>,
    pub(crate) get_config: Func,
    pub(crate) instruction_counter: TypedFunc<(), u64>,
    instance: Instance,
    pub(crate) ctx: Box<dyn Send + Sync + ErasedWasmStore>,
}

impl CodecPlugin {
    pub fn new<E: WasmEngine>(
        instance: Instance,
        ctx: Store<(), E>,
    ) -> Result<Self, RuntimeError>
    where
        Store<(), E>: Send + Sync,
    {
        fn load_func(
            interface: &ExportInstance,
            name: &str,
        ) -> Result<Func, RuntimeError> {
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

        let interfaces = CodecPluginInterfaces::get();

        let Some(codecs_interface) = instance.exports().instance(&interfaces.codecs) else {
            return Err(RuntimeError::from(anyhow::Error::msg(format!(
                "WASM component does not contain an interface named `{}`",
                interfaces.codecs
            ))));
        };
        let Some(perf_interface) = instance.exports().instance(&interfaces.perf) else {
            return Err(RuntimeError::from(anyhow::Error::msg(format!(
                "WASM component does not contain an interface named `{}`",
                interfaces.perf
            ))));
        };

        Ok(Self {
            from_config: load_func(codecs_interface, "[static]codec.from-config")?,
            encode: load_func(codecs_interface, "[method]codec.encode")?,
            decode: load_func(codecs_interface, "[method]codec.decode")?,
            decode_into: load_func(codecs_interface, "[method]codec.decode-into")?,
            codec_id: load_typed_func(codecs_interface, "codec-id")?,
            codec_config_schema: load_typed_func(codecs_interface, "codec-config-schema")?,
            get_config: load_func(codecs_interface, "[method]codec.get-config")?,
            instruction_counter: load_typed_func(perf_interface, &interfaces.instruction_counter)?,
            instance,
            ctx: Box::new(ctx),
        })
    }

    pub fn codec_id(&mut self) -> Result<String, RuntimeError> {
        self.ctx
            .call_typed_str_func(&self.codec_id)
            .map_err(RuntimeError::from)
    }

    pub fn codec_config_schema(&mut self) -> Result<Schema, RuntimeError> {
        let schema = self
            .ctx
            .call_typed_str_func(&self.codec_config_schema)
            .map_err(RuntimeError::from)?;

        let schema = serde_json::from_str(&schema)
            .map_err(anyhow::Error::new)
            .map_err(RuntimeError::from)?;

        Ok(schema)
    }

    pub fn from_config<'de, P: BorrowMut<Self>, D: Deserializer<'de>>(
        mut plugin: P,
        config: D,
    ) -> Result<WasmCodec<P>, D::Error> {
        let plugin_borrow: &mut Self = plugin.borrow_mut();

        let mut config_bytes = Vec::new();
        serde_transcode::transcode(config, &mut serde_json::Serializer::new(&mut config_bytes))
            .map_err(serde::de::Error::custom)?;
        let config = String::from_utf8(config_bytes).map_err(serde::de::Error::custom)?;

        let args = Value::String(config.into());
        let mut result = Value::U8(0);

        plugin_borrow
            .ctx
            .call_func(
                &plugin_borrow.from_config,
                std::slice::from_ref(&args),
                std::slice::from_mut(&mut result),
            )
            .map_err(serde::de::Error::custom)?;

        let codec = match result {
            Value::Result(result) => match &*result {
                Ok(Some(Value::Own(resource))) => WasmCodec {
                    resource: resource.clone(),
                    plugin,
                    instruction_counter: 0,
                },
                Err(err) => match guest_error_from_wasm(err.as_ref()) {
                    Ok(err) => return Err(serde::de::Error::custom(err)),
                    Err(err) => return Err(serde::de::Error::custom(err)),
                },
                result => {
                    return Err(serde::de::Error::custom(format!(
                        "unexpected from-config result value {result:?}"
                    )))
                },
            },
            value => {
                return Err(serde::de::Error::custom(format!(
                    "unexpected from-config result value {value:?}"
                )))
            },
        };

        Ok(codec)
    }

    pub fn drop(mut self) -> Result<(), RuntimeError> {
        let result = self
            .ctx
            .drop_instance(&self.instance)
            .map_err(RuntimeError::from);

        // We need to forget here instead of using ManuallyDrop since we need
        //  both a mutable borrow to self.plugin and an immutable borrow to
        //  self.resource at the same time
        std::mem::forget(self);

        let mut errors = result?;

        let Some(mut err) = errors.pop() else {
            return Ok(());
        };

        if !errors.is_empty() {
            err = err.context(format!("showing one of {} errors", errors.len() + 1));
        }

        Err(RuntimeError::from(err.context(
            "dropping instance and all of its resources failed",
        )))
    }
}

impl Drop for CodecPlugin {
    fn drop(&mut self) {
        std::mem::drop(self.ctx.drop_instance(&self.instance));
    }
}
