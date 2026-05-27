//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.87.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-wasm-guest
//! [crates.io]: https://crates.io/crates/numcodecs-wasm-guest
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-wasm-guest
//! [docs.rs]: https://docs.rs/numcodecs-wasm-guest/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_wasm_guest
//!
//! wasm32 guest-side bindings for the [`numcodecs`] API, which allows you to
//! export one [`StaticCodec`] from a WASM component.

// Required in docs and the [`export_codec`] macro
#[doc(hidden)]
pub use numcodecs;

#[cfg(doc)]
use numcodecs::StaticCodec;

#[cfg(target_arch = "wasm32")]
use ::{
    numcodecs::{Codec, StaticCodec},
    schemars::schema_for,
    serde::Deserialize,
};

// #[cfg(target_arch = "wasm32")]
mod convert;

#[cfg(target_arch = "wasm32")]
use crate::{
    bindings::exports::numcodecs::abc::codec as wit,
    convert::{
        from_wit_any_array, into_wit_any_array, into_wit_error, zeros_from_wit_any_array_prototype,
    },
};

#[doc(hidden)]
#[expect(clippy::same_length_and_capacity)]
pub mod bindings {
    wit_bindgen::generate!({
        world: "numcodecs:abc/exports@0.1.1",
        with: {
            "numcodecs:abc/codec@0.1.1": generate,
        },
        pub_export_macro: true,
    });
}

#[macro_export]
/// Export a [`StaticCodec`] type using the WASM component model.
///
/// ```rust,ignore
/// # use numcodecs_wasm_guest::export_codec;
///
/// struct MyCodec {
///     // ...
/// }
///
/// impl numcodecs::Codec for MyCodec {
///     // ...
/// }
///
/// impl numcodecs::StaticCodec for MyCodec {
///     // ...
/// }
///
/// export_codec!(MyCodec);
/// ```
macro_rules! export_codec {
    ($codec:ty) => {
        #[cfg(target_arch = "wasm32")]
        const _: () = {
            type Codec = $codec;

            $crate::bindings::export!(
                Codec with_types_in $crate::bindings
            );
        };

        const _: () = {
            const fn can_only_export_static_codec<T: $crate::numcodecs::StaticCodec>() {}

            can_only_export_static_codec::<$codec>()
        };
    };
}

#[cfg(target_arch = "wasm32")]
#[doc(hidden)]
impl<T: StaticCodec> wit::Guest for T {
    type Codec = Self;

    fn codec_id() -> String {
        String::from(<Self as StaticCodec>::CODEC_ID)
    }

    fn codec_config_schema() -> wit::JsonSchema {
        schema_for!(<Self as StaticCodec>::Config<'static>)
            .as_value()
            .to_string()
    }
}

#[cfg(target_arch = "wasm32")]
impl<T: StaticCodec> wit::GuestCodec for T {
    fn from_config(config: String) -> Result<wit::Codec, wit::Error> {
        let err = match <Self as StaticCodec>::Config::deserialize(
            &mut serde_json::Deserializer::from_str(&config),
        ) {
            Ok(config) => return Ok(wit::Codec::new(<Self as StaticCodec>::from_config(config))),
            Err(err) => err,
        };

        let err = format_serde_error::SerdeError::new(config, err);
        Err(into_wit_error(err))
    }

    fn encode(&self, data: wit::AnyArray) -> Result<wit::AnyArray, wit::Error> {
        let data = match from_wit_any_array(data) {
            Ok(data) => data,
            Err(err) => return Err(into_wit_error(err)),
        };

        match <Self as Codec>::encode(self, data.into_cow()) {
            Ok(encoded) => match into_wit_any_array(encoded) {
                Ok(encoded) => Ok(encoded),
                Err(err) => Err(into_wit_error(err)),
            },
            Err(err) => Err(into_wit_error(err)),
        }
    }

    fn decode(&self, encoded: wit::AnyArray) -> Result<wit::AnyArray, wit::Error> {
        let encoded = match from_wit_any_array(encoded) {
            Ok(encoded) => encoded,
            Err(err) => return Err(into_wit_error(err)),
        };

        match <Self as Codec>::decode(self, encoded.into_cow()) {
            Ok(decoded) => match into_wit_any_array(decoded) {
                Ok(decoded) => Ok(decoded),
                Err(err) => Err(into_wit_error(err)),
            },
            Err(err) => Err(into_wit_error(err)),
        }
    }

    fn decode_into(
        &self,
        encoded: wit::AnyArray,
        decoded: wit::AnyArrayPrototype,
    ) -> Result<wit::AnyArray, wit::Error> {
        let encoded = match from_wit_any_array(encoded) {
            Ok(encoded) => encoded,
            Err(err) => return Err(into_wit_error(err)),
        };

        let mut decoded = zeros_from_wit_any_array_prototype(decoded);

        match <Self as Codec>::decode_into(self, encoded.view(), decoded.view_mut()) {
            Ok(()) => match into_wit_any_array(decoded) {
                Ok(decoded) => Ok(decoded),
                Err(err) => Err(into_wit_error(err)),
            },
            Err(err) => Err(into_wit_error(err)),
        }
    }

    fn get_config(&self) -> Result<wit::Json, wit::Error> {
        match serde_json::to_string(&<Self as StaticCodec>::get_config(self)) {
            Ok(config) => Ok(config),
            Err(err) => Err(into_wit_error(err)),
        }
    }
}

pub fn get_codec<'de, D: serde::Deserializer<'de>>(
    config: D,
) -> Result<ExternalCodec, ExternalError> {
    let mut config_bytes = Vec::new();
    serde_transcode::transcode(config, &mut serde_json::Serializer::new(&mut config_bytes))
        .map_err(convert::into_wit_error)
        .map_err(ExternalError::new)?;
    let config = String::from_utf8(config_bytes)
        .map_err(convert::into_wit_error)
        .map_err(ExternalError::new)?;

    let codec =
        bindings::numcodecs::abc::registry::get_codec(&config).map_err(ExternalError::new)?;
    let ty = codec.ty();

    let codec_id = ty.codec_id();
    let schema: schemars::Schema = serde_json::from_str(&ty.codec_config_schema())
        .map_err(convert::into_wit_error)
        .map_err(ExternalError::new)?;

    Ok(ExternalCodec {
        codec,
        ty: ExternalCodecType {
            ty: std::sync::Arc::new(ty),
            codec_id: codec_id.into(),
            schema: std::sync::Arc::new(schema),
        },
    })
}

pub struct ExternalCodec {
    codec: bindings::numcodecs::abc::registry::ErasedDynCodec,
    ty: ExternalCodecType,
}

impl Clone for ExternalCodec {
    fn clone(&self) -> Self {
        Self {
            codec: self.codec.clone(),
            ty: ExternalCodecType {
                ty: self.ty.ty.clone(),
                codec_id: self.ty.codec_id.clone(),
                schema: self.ty.schema.clone(),
            },
        }
    }
}

impl numcodecs::Codec for ExternalCodec {
    type Error = ExternalError;

    fn encode(&self, data: numcodecs::AnyCowArray) -> Result<numcodecs::AnyArray, Self::Error> {
        match self
            .codec
            .encode(&convert::into_wit_any_array(data.into_owned()))
        {
            Ok(encoded) => match convert::from_wit_any_array(encoded) {
                Ok(encoded) => Ok(encoded),
                Err(err) => Err(ExternalError::new(convert::into_wit_error(err))),
            },
            Err(err) => Err(ExternalError::new(err)),
        }
    }

    fn decode(&self, encoded: numcodecs::AnyCowArray) -> Result<numcodecs::AnyArray, Self::Error> {
        match self
            .codec
            .decode(&convert::into_wit_any_array(encoded.into_owned()))
        {
            Ok(decoded) => match convert::from_wit_any_array(decoded) {
                Ok(decoded) => Ok(decoded),
                Err(err) => Err(ExternalError::new(convert::into_wit_error(err))),
            },
            Err(err) => Err(ExternalError::new(err)),
        }
    }

    fn decode_into(
        &self,
        encoded: numcodecs::AnyArrayView,
        mut decoded: numcodecs::AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        match self.codec.decode_into(
            &convert::into_wit_any_array(encoded.into_owned()),
            &bindings::numcodecs::abc::types::AnyArrayPrototype {
                dtype: match convert::into_wit_any_array_dtype(decoded.dtype()) {
                    Ok(dtype) => dtype,
                    Err(err) => return Err(ExternalError::new(convert::into_wit_error(err))),
                },
                shape: convert::usize_as_u32_slice(decoded.shape()),
            },
        ) {
            Ok(dec) => match convert::from_wit_any_array(dec) {
                Ok(dec) => {
                    decoded.assign(&dec);
                    Ok(())
                }
                Err(err) => Err(ExternalError::new(convert::into_wit_error(err))),
            },
            Err(err) => Err(ExternalError::new(err)),
        }
    }
}

impl numcodecs::DynCodec for ExternalCodec {
    type Type = ExternalCodecType;

    fn get_config<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let config = self
            .codec
            .get_config()
            .map_err(ExternalError::new)
            .map_err(serde::ser::Error::custom)?;
        serde_transcode::transcode(&mut serde_json::Deserializer::from_str(&config), serializer)
    }

    fn ty(&self) -> Self::Type {
        ExternalCodecType {
            ty: self.ty.ty.clone(),
            codec_id: self.ty.codec_id.clone(),
            schema: self.ty.schema.clone(),
        }
    }
}

pub struct ExternalCodecType {
    ty: std::sync::Arc<bindings::numcodecs::abc::registry::ErasedDynCodecType>,
    codec_id: std::sync::Arc<str>,
    schema: std::sync::Arc<schemars::Schema>,
}

impl numcodecs::DynCodecType for ExternalCodecType {
    type Codec = ExternalCodec;

    fn codec_id(&self) -> &str {
        &*self.codec_id
    }

    fn codec_config_schema(&self) -> schemars::Schema {
        (*self.schema).clone()
    }

    fn codec_from_config<'de, D: serde::Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<Self::Codec, D::Error> {
        let mut config_bytes = Vec::new();
        serde_transcode::transcode(config, &mut serde_json::Serializer::new(&mut config_bytes))
            .map_err(serde::de::Error::custom)?;
        let config = String::from_utf8(config_bytes).map_err(serde::de::Error::custom)?;

        let codec = self
            .ty
            .from_config(&config)
            .map_err(ExternalError::new)
            .map_err(serde::de::Error::custom)?;

        Ok(ExternalCodec {
            codec,
            ty: ExternalCodecType {
                ty: self.ty.clone(),
                codec_id: self.codec_id.clone(),
                schema: self.schema.clone(),
            },
        })
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{msg}")]
pub struct ExternalError {
    msg: String,
    source: Option<Box<Self>>,
}

impl ExternalError {
    pub(crate) fn new(error: bindings::numcodecs::abc::types::Error) -> Self {
        let mut root = Self {
            msg: error.message,
            source: None,
        };

        let mut err = &mut root;

        for msg in error.chain {
            err = &mut *err.source.insert(Box::new(Self { msg, source: None }));
        }

        root
    }
}
