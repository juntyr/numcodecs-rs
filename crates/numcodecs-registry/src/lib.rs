//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.87.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-registry
//! [crates.io]: https://crates.io/crates/numcodecs-registry
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-registry
//! [docs.rs]: https://docs.rs/numcodecs-registry/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_registry
//!
//! Registries for compression codecs implementing the [`numcodecs`] API.
//!
//! [`numcodecs`]: https://numcodecs.readthedocs.io/en/stable/

use std::{error::Error, sync::Arc};

use numcodecs::{DynCodec, ErasedDynCodec, ErasedError};
use serde::Deserializer;

/// Registry of codec types.
pub trait Registry: 'static + Send + Sync {
    /// Error type that may be returned during
    /// [`get_codec`][`Registry::get_codec`] and
    /// and [`register_codec`][`Codec::Registry`].
    type Error: 'static + Send + Sync + Error;

    /// Instantiate a codec of any type from its `config`uration.
    ///
    /// The config *must* include the `id` field with the
    /// [`DynCodecType::codec_id`].
    ///
    /// # Errors
    ///
    /// Errors if no codec with a matching `id` has been registered, or if
    /// constructing the codec fails.
    fn get_codec<'de, D: Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<ErasedDynCodec, Self::Error>;

    /// Instantiate a codec with a concrete type from its `config`uration.
    ///
    /// The config *must* include the `id` field with the
    /// [`DynCodecType::codec_id`].
    ///
    /// # Errors
    ///
    /// Errors if no codec with a matching `id` has been registered, if
    /// constructing the codec fails, or if the constructed codec is not of the
    /// concrete type.
    fn get_codec_typed<'de, T: DynCodec, D: Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<Option<T>, Self::Error> {
        self.get_codec(config).map(|codec| codec.downcast().ok())
    }
}

impl<R: Registry> Registry for Box<R> {
    type Error = R::Error;

    fn get_codec<'de, D: Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<ErasedDynCodec, Self::Error> {
        R::get_codec(self, config)
    }

    fn get_codec_typed<'de, T: DynCodec, D: Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<Option<T>, Self::Error> {
        R::get_codec_typed(self, config)
    }
}

impl<R: Registry> Registry for Arc<R> {
    type Error = R::Error;

    fn get_codec<'de, D: Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<ErasedDynCodec, Self::Error> {
        R::get_codec(self, config)
    }

    fn get_codec_typed<'de, T: DynCodec, D: Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<Option<T>, Self::Error> {
        R::get_codec_typed(self, config)
    }
}

/// Type-erased [`Registry`].
pub struct ErasedRegistry {
    registry: Box<dyn ErasedRegistryDispatch>,
}

impl ErasedRegistry {
    /// Erase the type information of the concrete `registry`.
    pub fn new<T: Registry>(registry: T) -> Self {
        Self {
            registry: Box::new(registry),
        }
    }
}

impl Registry for ErasedRegistry {
    type Error = ErasedError;

    fn get_codec<'de, D: Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<ErasedDynCodec, Self::Error> {
        self.registry
            .erased_get_codec(&mut <dyn erased_serde::Deserializer>::erase(config))
    }
}

trait ErasedRegistryDispatch: 'static + Send + Sync {
    fn erased_get_codec(
        &self,
        config: &mut dyn erased_serde::Deserializer,
    ) -> Result<ErasedDynCodec, ErasedError>;
}

impl<T: Registry> ErasedRegistryDispatch for T {
    fn erased_get_codec(
        &self,
        config: &mut dyn erased_serde::Deserializer,
    ) -> Result<ErasedDynCodec, ErasedError> {
        match self.get_codec(config) {
            Ok(codec) => Ok(codec),
            Err(err) => Err(ErasedError::new(err)),
        }
    }
}

/// Global registry singleton.
///
/// If the global registry is used, its backing registry must be provided
/// exactly once using [`export_global`].
///
/// The global registry must not be used to provide the backing of itself,
/// which would result in an infinite loop at runtime.
pub struct GlobalRegistry;

impl GlobalRegistry {
    fn get() -> &'static ErasedRegistry {
        #[expect(unsafe_code)]
        unsafe extern "C" {
            #[expect(improper_ctypes)]
            safe fn _numcodecs_registry_get_global_registry() -> &'static ErasedRegistry;
        }

        _numcodecs_registry_get_global_registry()
    }
}

impl Registry for GlobalRegistry {
    type Error = ErasedError;

    fn get_codec<'de, D: Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<ErasedDynCodec, Self::Error> {
        Self::get().get_codec(config)
    }

    fn get_codec_typed<'de, T: DynCodec, D: Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<Option<T>, Self::Error> {
        Self::get().get_codec_typed(config)
    }
}

#[macro_export]
/// `export_global!(registry: ty = expr)` exports the provided registry as the
/// global registry singleton.
///
/// This macro must only be used at most once in every binary or shared
/// library.
macro_rules! export_global {
    (registry: $ty:ty = $init:expr) => {
        const _: () = {
            use std::sync::LazyLock;

            use $crate::ErasedRegistry;

            static _GLOBAL_REGISTRY: LazyLock<ErasedRegistry> =
                LazyLock::new(|| ErasedRegistry::new($init));

            #[allow(improper_ctypes, unsafe_code)]
            #[unsafe(no_mangle)]
            extern "C" fn _numcodecs_registry_get_global_registry() -> &'static ErasedRegistry {
                LazyLock::force(&_GLOBAL_REGISTRY)
            }
        };
    };
}

#[derive(Debug, thiserror::Error)]
#[error("codec not found")]
/// Codec was not found in the registry
pub struct CodecNotFoundError;

/// Empty registry that contains no codecs
pub struct EmptyRegistry;

impl Registry for EmptyRegistry {
    type Error = CodecNotFoundError;

    fn get_codec<'de, D: Deserializer<'de>>(
        &self,
        _config: D,
    ) -> Result<ErasedDynCodec, Self::Error> {
        Err(CodecNotFoundError)
    }

    fn get_codec_typed<'de, T: DynCodec, D: Deserializer<'de>>(
        &self,
        _config: D,
    ) -> Result<Option<T>, Self::Error> {
        Err(CodecNotFoundError)
    }
}
