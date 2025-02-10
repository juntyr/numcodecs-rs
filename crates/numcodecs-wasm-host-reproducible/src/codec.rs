use std::num::Wrapping;
use std::sync::{Arc, Mutex};

use numcodecs::{
    AnyArray, AnyArrayView, AnyArrayViewMut, AnyCowArray, Codec, DynCodec, DynCodecType,
};
use numcodecs_wasm_host::{CodecError, RuntimeError, WasmCodec, WasmCodecComponent};
use schemars::Schema;
use serde::Serializer;
use wasm_component_layer::{AsContextMut, Component, Instance, Linker, Store, TypedFunc};
use wasm_runtime_layer::{backend::WasmEngine, Engine};

use crate::transform::instcnt::PerfWitInterfaces;
use crate::transform::transform_wasm_component;
use crate::{engine::ReproducibleEngine, logging, stdio};

#[derive(Debug, thiserror::Error)]
/// Errors that can occur when using the [`ReproducibleWasmCodec`]
pub enum ReproducibleWasmCodecError {
    /// The codec's lock was poisoned
    #[error("{codec_id} codec's lock was poisoned")]
    Poisoned {
        /// The codec's id
        codec_id: Arc<str>,
    },
    /// The codec's WebAssembly runtime raised an error
    #[error("{codec_id} codec's WebAssembly runtime raised an error")]
    Runtime {
        /// The codec's id
        codec_id: Arc<str>,
        /// The runtime error
        source: RuntimeError,
    },
    /// The codec's implementation raised an error
    #[error("{codec_id} codec's implementation raised an error")]
    Codec {
        /// The codec's id
        codec_id: Arc<str>,
        /// The codec error
        source: CodecError,
    },
}

/// Codec instantiated inside a WebAssembly component.
///
/// The codec is loaded such that its execution is reproducible across any
/// platform. Importantly, each codec owns its own component instance such that
/// two codecs cannot interfere.
pub struct ReproducibleWasmCodec<E: WasmEngine>
where
    Store<(), ReproducibleEngine<E>>: Send,
{
    store: Mutex<Store<(), ReproducibleEngine<E>>>,
    instance: Instance,
    codec: WasmCodec,
    ty: ReproducibleWasmCodecType<E>,
    instruction_counter: TypedFunc<(), u64>,
}

impl<E: WasmEngine> ReproducibleWasmCodec<E>
where
    Store<(), ReproducibleEngine<E>>: Send,
{
    /// Try cloning the codec by recreating it from its configuration.
    ///
    /// `ReproducibleWasmCodec` implements [`Clone`] by calling this method and
    /// panicking if it fails.
    ///
    /// # Errors
    ///
    /// Errors if serializing the codec configuration, constructing the new
    /// codec, or interacting with the component fails.
    pub fn try_clone(&self) -> Result<Self, serde_json::Error> {
        let mut config = self.get_config(serde_json::value::Serializer)?;

        if let Some(config) = config.as_object_mut() {
            config.remove("id");
        }

        let codec: Self = self.ty.codec_from_config(config)?;

        Ok(codec)
    }

    /// Try dropping the codec.
    ///
    /// `ReproducibleWasmCodec` implements [`Drop`] by calling this method and
    /// ignoring any errors.
    // That's not quite true but the effect is the same
    ///
    /// # Errors
    ///
    /// Errors if dropping the codec's resource or component, or interacting
    /// with the component fails.
    pub fn try_drop(mut self) -> Result<(), ReproducibleWasmCodecError> {
        // keep in sync with drop
        let mut store = self
            .store
            .get_mut()
            .map_err(|_| ReproducibleWasmCodecError::Poisoned {
                codec_id: self.ty.codec_id.clone(),
            })?;

        let result =
            self.codec
                .try_drop(&mut store)
                .map_err(|source| ReproducibleWasmCodecError::Runtime {
                    codec_id: self.ty.codec_id.clone(),
                    source,
                });
        let results = try_drop_instance(store, &self.instance, &self.ty.codec_id);

        result.and(results)
    }

    #[expect(clippy::significant_drop_tightening)]
    /// Read the codec's instruction counter, which is based on the number of
    /// WebAssembly bytecode instructions executed.
    ///
    /// The instruction counter is never reset and wraps around. Comparisons of
    /// the counter values before and after, e.g. a call to [`Self::encode`],
    /// should thus use wrapping arithmetic.
    ///
    /// # Errors
    ///
    /// Errors if interacting with the component fails.
    pub fn instruction_counter(&self) -> Result<Wrapping<u64>, ReproducibleWasmCodecError> {
        let mut store = self
            .store
            .lock()
            .map_err(|_| ReproducibleWasmCodecError::Poisoned {
                codec_id: self.ty.codec_id.clone(),
            })?;

        let cnt = self
            .instruction_counter
            .call(store.as_context_mut(), ())
            .map_err(|err| ReproducibleWasmCodecError::Runtime {
                codec_id: self.ty.codec_id.clone(),
                source: RuntimeError::from(err),
            })?;

        Ok(Wrapping(cnt))
    }
}

impl<E: WasmEngine> Clone for ReproducibleWasmCodec<E>
where
    Store<(), ReproducibleEngine<E>>: Send,
{
    fn clone(&self) -> Self {
        #[expect(clippy::expect_used)]
        self.try_clone()
            .expect("cloning a wasm codec should not fail")
    }
}

impl<E: WasmEngine> Drop for ReproducibleWasmCodec<E>
where
    Store<(), ReproducibleEngine<E>>: Send,
{
    fn drop(&mut self) {
        // keep in sync with try_drop
        let Ok(mut store) = self.store.get_mut() else {
            return;
        };

        let result = self.codec.try_drop(&mut store);
        std::mem::drop(result);

        let results = self.instance.drop(store);
        std::mem::drop(results);
    }
}

impl<E: WasmEngine> Codec for ReproducibleWasmCodec<E>
where
    Store<(), ReproducibleEngine<E>>: Send,
{
    type Error = ReproducibleWasmCodecError;

    #[expect(clippy::significant_drop_tightening)]
    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        let mut store = self
            .store
            .lock()
            .map_err(|_| ReproducibleWasmCodecError::Poisoned {
                codec_id: self.ty.codec_id.clone(),
            })?;

        let encoded = self
            .codec
            .encode(store.as_context_mut(), data)
            .map_err(|err| ReproducibleWasmCodecError::Runtime {
                codec_id: self.ty.codec_id.clone(),
                source: err,
            })?
            .map_err(|err| ReproducibleWasmCodecError::Codec {
                codec_id: self.ty.codec_id.clone(),
                source: err,
            })?;

        Ok(encoded)
    }

    #[expect(clippy::significant_drop_tightening)]
    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        let mut store = self
            .store
            .lock()
            .map_err(|_| ReproducibleWasmCodecError::Poisoned {
                codec_id: self.ty.codec_id.clone(),
            })?;

        let decoded = self
            .codec
            .decode(store.as_context_mut(), encoded)
            .map_err(|err| ReproducibleWasmCodecError::Runtime {
                codec_id: self.ty.codec_id.clone(),
                source: err,
            })?
            .map_err(|err| ReproducibleWasmCodecError::Codec {
                codec_id: self.ty.codec_id.clone(),
                source: err,
            })?;

        Ok(decoded)
    }

    #[expect(clippy::significant_drop_tightening)]
    fn decode_into(
        &self,
        encoded: AnyArrayView,
        decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        let mut store = self
            .store
            .lock()
            .map_err(|_| ReproducibleWasmCodecError::Poisoned {
                codec_id: self.ty.codec_id.clone(),
            })?;

        self.codec
            .decode_into(store.as_context_mut(), encoded, decoded)
            .map_err(|err| ReproducibleWasmCodecError::Runtime {
                codec_id: self.ty.codec_id.clone(),
                source: err,
            })?
            .map_err(|err| ReproducibleWasmCodecError::Codec {
                codec_id: self.ty.codec_id.clone(),
                source: err,
            })?;

        Ok(())
    }
}

impl<E: WasmEngine> DynCodec for ReproducibleWasmCodec<E>
where
    Store<(), ReproducibleEngine<E>>: Send,
{
    type Type = ReproducibleWasmCodecType<E>;

    fn ty(&self) -> Self::Type {
        ReproducibleWasmCodecType {
            codec_id: self.ty.codec_id.clone(),
            codec_config_schema: self.ty.codec_config_schema.clone(),
            component: self.ty.component.clone(),
            component_instantiater: self.ty.component_instantiater.clone(),
        }
    }

    fn get_config<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut store = self
            .store
            .lock()
            .map_err(|_| ReproducibleWasmCodecError::Poisoned {
                codec_id: self.ty.codec_id.clone(),
            })
            .map_err(serde::ser::Error::custom)?;

        self.codec.get_config(store.as_context_mut(), serializer)
    }
}

/// Type object for a codec instantiated inside a WebAssembly component.
pub struct ReproducibleWasmCodecType<E: WasmEngine>
where
    Store<(), ReproducibleEngine<E>>: Send,
{
    pub(super) codec_id: Arc<str>,
    pub(super) codec_config_schema: Arc<Schema>,
    pub(super) component: Component,
    #[expect(clippy::type_complexity)]
    pub(super) component_instantiater: Arc<
        dyn Send
            + Sync
            + Fn(
                &Component,
                &str,
            ) -> Result<
                (
                    Store<(), ReproducibleEngine<E>>,
                    Instance,
                    WasmCodecComponent,
                ),
                ReproducibleWasmCodecError,
            >,
    >,
}

impl<E: WasmEngine> ReproducibleWasmCodecType<E>
where
    Store<(), ReproducibleEngine<E>>: Send,
{
    /// Load a [`DynCodecType`] from a binary `wasm_component`, which will be
    /// executed in the provided core WebAssembly `engine`.
    ///
    /// # Errors
    ///
    /// Errors if the `wasm_component` does not export the `numcodecs:abc/codec`
    /// interface or if interacting with the component fails.
    pub fn new(
        engine: E,
        wasm_component: impl Into<Vec<u8>>,
    ) -> Result<Self, ReproducibleWasmCodecError>
    where
        E: Send + Sync,
        Store<(), ReproducibleEngine<E>>: Send + Sync,
    {
        let wasm_component = transform_wasm_component(wasm_component).map_err(|err| {
            ReproducibleWasmCodecError::Runtime {
                codec_id: Arc::from("<unknown>"),
                source: RuntimeError::from(err),
            }
        })?;

        let engine = Engine::new(ReproducibleEngine::new(engine));
        let component = Component::new(&engine, &wasm_component).map_err(|err| {
            ReproducibleWasmCodecError::Runtime {
                codec_id: Arc::from("<unknown>"),
                source: RuntimeError::from(err),
            }
        })?;

        let component_instantiater = Arc::new(move |component: &Component, codec_id: &str| {
            let mut store = Store::new(&engine, ());

            let mut linker = Linker::default();
            stdio::add_to_linker(&mut linker, &mut store).map_err(|err| {
                ReproducibleWasmCodecError::Runtime {
                    codec_id: Arc::from(codec_id),
                    source: RuntimeError::from(err),
                }
            })?;
            logging::add_to_linker(&mut linker, &mut store).map_err(|err| {
                ReproducibleWasmCodecError::Runtime {
                    codec_id: Arc::from(codec_id),
                    source: RuntimeError::from(err),
                }
            })?;

            let instance = linker.instantiate(&mut store, component).map_err(|err| {
                ReproducibleWasmCodecError::Runtime {
                    codec_id: Arc::from(codec_id),
                    source: RuntimeError::from(err),
                }
            })?;

            let component =
                WasmCodecComponent::new(&mut store, instance.clone()).map_err(|source| {
                    ReproducibleWasmCodecError::Runtime {
                        codec_id: Arc::from(codec_id),
                        source,
                    }
                })?;

            Ok((store, instance, component))
        });

        let (codec_id, codec_config_schema) = {
            let (mut store, instance, ty): (_, _, WasmCodecComponent) =
                (component_instantiater)(&component, "<unknown>")?;

            let codec_id = Arc::from(ty.codec_id());
            let codec_config_schema = Arc::from(ty.codec_config_schema().clone());

            try_drop_instance(&mut store, &instance, &codec_id)?;

            (codec_id, codec_config_schema)
        };

        Ok(Self {
            codec_id,
            codec_config_schema,
            component,
            component_instantiater,
        })
    }
}

impl<E: WasmEngine> DynCodecType for ReproducibleWasmCodecType<E>
where
    Store<(), ReproducibleEngine<E>>: Send,
{
    type Codec = ReproducibleWasmCodec<E>;

    fn codec_id(&self) -> &str {
        &self.codec_id
    }

    fn codec_from_config<'de, D: serde::Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<Self::Codec, D::Error> {
        let (mut store, instance, component) =
            (self.component_instantiater)(&self.component, &self.codec_id)
                .map_err(serde::de::Error::custom)?;
        let codec = component.codec_from_config(store.as_context_mut(), config)?;

        let PerfWitInterfaces {
            perf: perf_interface,
            instruction_counter,
        } = PerfWitInterfaces::get();
        let Some(perf_interface) = instance.exports().instance(perf_interface) else {
            return Err(serde::de::Error::custom(
                "WASM component does not contain an interface to read the instruction counter",
            ));
        };
        let Some(instruction_counter) = perf_interface.func(instruction_counter) else {
            return Err(serde::de::Error::custom(
                "WASM component interface does not contain a function to read the instruction counter"
            ));
        };
        let instruction_counter = instruction_counter.typed().map_err(|err| {
            serde::de::Error::custom(format!(
                "WASM component instruction counter function has the wrong signature: {err}"
            ))
        })?;

        Ok(ReproducibleWasmCodec {
            store: Mutex::new(store),
            instance,
            codec,
            ty: Self {
                codec_id: self.codec_id.clone(),
                codec_config_schema: self.codec_config_schema.clone(),
                component: self.component.clone(),
                component_instantiater: self.component_instantiater.clone(),
            },
            instruction_counter,
        })
    }

    fn codec_config_schema(&self) -> Schema {
        (*self.codec_config_schema).clone()
    }
}

fn try_drop_instance<T, E: WasmEngine>(
    store: &mut Store<T, E>,
    instance: &Instance,
    codec_id: &str,
) -> Result<(), ReproducibleWasmCodecError> {
    let mut errors = instance
        .drop(store)
        .map_err(|err| ReproducibleWasmCodecError::Runtime {
            codec_id: Arc::from(codec_id),
            source: RuntimeError::from(err),
        })?;

    let Some(mut err) = errors.pop() else {
        return Ok(());
    };

    if !errors.is_empty() {
        err = err.context(format!("showing one of {} errors", errors.len() + 1));
    }

    Err(ReproducibleWasmCodecError::Runtime {
        codec_id: Arc::from(codec_id),
        source: RuntimeError::from(
            err.context("dropping instance and all of its resources failed"),
        ),
    })
}
