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

#![expect(missing_docs, clippy::missing_errors_doc)] // FIXME

use std::{
    collections::HashMap,
    error::Error,
    sync::{LazyLock, RwLock},
};

use numcodecs::{DynCodec, DynCodecType, ErasedDynCodec, ErasedDynCodecType, ErasedError};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Value};

pub trait Registry: 'static + Send + Sync {
    /// Error type that may be returned during
    /// [`get_codec`][`Registry::get_codec`] and
    /// and [`register_codec`][`Codec::Registry`].
    type Error: 'static + Send + Sync + Error;

    fn get_codec<'de, D: Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<ErasedDynCodec, Self::Error>;

    fn get_codec_typed<'de, T: DynCodec, D: Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<Option<T>, Self::Error> {
        self.get_codec(config).map(|codec| codec.downcast().ok())
    }

    fn register_codec<T: DynCodecType>(
        &mut self,
        ty: T,
    ) -> Result<Option<ErasedDynCodecType>, Self::Error>;
}

pub struct ErasedRegistry {
    registry: Box<dyn ErasedRegistryDispatch>,
}

impl ErasedRegistry {
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

    fn register_codec<T: DynCodecType>(
        &mut self,
        ty: T,
    ) -> Result<Option<ErasedDynCodecType>, Self::Error> {
        self.registry
            .erased_register_codec(ErasedDynCodecType::new(ty))
    }
}

trait ErasedRegistryDispatch: 'static + Send + Sync {
    fn erased_get_codec(
        &self,
        config: &mut dyn erased_serde::Deserializer,
    ) -> Result<ErasedDynCodec, ErasedError>;

    fn erased_register_codec(
        &mut self,
        ty: ErasedDynCodecType,
    ) -> Result<Option<ErasedDynCodecType>, ErasedError>;
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

    fn erased_register_codec(
        &mut self,
        ty: ErasedDynCodecType,
    ) -> Result<Option<ErasedDynCodecType>, ErasedError> {
        match self.register_codec(ty) {
            Ok(codec) => Ok(codec),
            Err(err) => Err(ErasedError::new(err)),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LocalRegistryError {
    #[error("codec not found")]
    CodecNotFound,
    #[error("invalid codec config")]
    InvalidCodecConfig,
}

pub struct LocalRegistry {
    tys: HashMap<String, ErasedDynCodecType>,
}

impl LocalRegistry {
    pub fn new() -> Self {
        Self {
            tys: HashMap::new(),
        }
    }
}

impl Registry for LocalRegistry {
    type Error = LocalRegistryError;

    fn get_codec<'de, D: Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<ErasedDynCodec, Self::Error> {
        let mut config = Map::<String, Value>::deserialize(config).unwrap();

        let Some(id) = config.remove("id") else {
            panic!("missing codec `id`");
        };

        let Value::String(id) = id else {
            panic!("codec `id` must be a string");
        };

        let Some(ty) = self.tys.get(&id) else {
            panic!("unknown codec with id `{id}`");
        };

        let codec = ty.codec_from_config(config).unwrap();

        Ok(codec)
    }

    fn register_codec<T: DynCodecType>(
        &mut self,
        ty: T,
    ) -> Result<Option<ErasedDynCodecType>, Self::Error> {
        Ok(self
            .tys
            .insert(String::from(ty.codec_id()), ErasedDynCodecType::new(ty)))
    }
}

static REGISTRY: LazyLock<RwLock<ErasedRegistry>> =
    LazyLock::new(|| RwLock::new(ErasedRegistry::new(LocalRegistry::new())));

#[must_use]
pub fn global_registry() -> &'static RwLock<ErasedRegistry> {
    LazyLock::force(&REGISTRY)
}
