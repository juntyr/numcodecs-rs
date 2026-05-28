use std::sync::Arc;

use numcodecs::{
    self, AnyArray, AnyArrayView, AnyArrayViewMut, AnyCowArray, Codec, DynCodec, DynCodecType,
    ErasedDynCodec,
};
use numcodecs_registry::{self, Registry, export_global};
use schemars::Schema;
use serde::{self, Deserializer, Serializer};
use serde_transcode::transcode;

use crate::{convert, wit};

pub struct ExternalCodec {
    codec: wit::exports::ErasedDynCodec,
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

impl Codec for ExternalCodec {
    type Error = ExternalError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
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

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
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
        encoded: AnyArrayView,
        mut decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        match self.codec.decode_into(
            &convert::into_wit_any_array(encoded.into_owned()),
            &wit::types::AnyArrayPrototype {
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

impl DynCodec for ExternalCodec {
    type Type = ExternalCodecType;

    fn get_config<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let config = self
            .codec
            .get_config()
            .map_err(ExternalError::new)
            .map_err(serde::ser::Error::custom)?;
        transcode(&mut serde_json::Deserializer::from_str(&config), serializer)
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
    ty: Arc<wit::imports::ErasedDynCodecType>,
    codec_id: Arc<str>,
    schema: Arc<Schema>,
}

impl DynCodecType for ExternalCodecType {
    type Codec = ExternalCodec;

    fn codec_id(&self) -> &str {
        &*self.codec_id
    }

    fn codec_config_schema(&self) -> Schema {
        (*self.schema).clone()
    }

    fn codec_from_config<'de, D: Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<Self::Codec, D::Error> {
        let mut config_bytes = Vec::new();
        transcode(config, &mut serde_json::Serializer::new(&mut config_bytes))
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
    pub(crate) fn new(error: wit::types::Error) -> Self {
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

pub struct ExternalRegistry;

impl Registry for ExternalRegistry {
    type Error = ExternalError;

    fn get_codec<'de, D: Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<ErasedDynCodec, Self::Error> {
        let mut config_bytes = Vec::new();
        transcode(config, &mut serde_json::Serializer::new(&mut config_bytes))
            .map_err(convert::into_wit_error)
            .map_err(ExternalError::new)?;
        let config = String::from_utf8(config_bytes)
            .map_err(convert::into_wit_error)
            .map_err(ExternalError::new)?;

        let codec = wit::imports::get_codec(&config).map_err(ExternalError::new)?;
        let ty = codec.ty();

        let codec_id = ty.codec_id();
        let schema: Schema = serde_json::from_str(&ty.codec_config_schema())
            .map_err(convert::into_wit_error)
            .map_err(ExternalError::new)?;

        let codec = ExternalCodec {
            codec,
            ty: ExternalCodecType {
                ty: std::sync::Arc::new(ty),
                codec_id: codec_id.into(),
                schema: std::sync::Arc::new(schema),
            },
        };

        Ok(ErasedDynCodec::new(codec))
    }
}

export_global! { registry: ExternalRegistry = ExternalRegistry }
