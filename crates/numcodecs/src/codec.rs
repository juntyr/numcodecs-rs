use std::{error::Error, marker::PhantomData};

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::{AnyArray, AnyArrayView, AnyArrayViewMut, AnyCowArray};

/// Compression codec that [`encode`][`Codec::encode`]s and
/// [`decode`][`Codec::decode`]s numeric n-dimensional arrays.
pub trait Codec: 'static + Send + Sync + Clone {
    /// Error type that may be returned during [`encode`][`Codec::encode`]ing
    /// and [`decode`][`Codec::decode`]ing.
    type Error: 'static + Send + Sync + Error;

    /// Encodes the `data` and returns the result.
    ///
    /// # Errors
    ///
    /// Errors if encoding the buffer fails.
    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error>;

    /// Decodes the `encoded` data and returns the result.
    ///
    /// # Errors
    ///
    /// Errors if decoding the buffer fails.
    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error>;

    /// Decodes the `encoded` data and writes the result into the provided
    /// `decoded` output.
    ///
    /// The output must have the correct type and shape.
    ///
    /// # Errors
    ///
    /// Errors if decoding the buffer fails.
    fn decode_into(
        &self,
        encoded: AnyArrayView,
        decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error>;

    /// Serializes the configuration parameters for this codec.
    ///
    /// The config *must* include an `id` field with the
    /// [`DynCodecType::codec_id`], for which the
    /// [`serialize_codec_config_with_id`] helper function may be used.
    ///
    /// The config *must* be compatible with JSON encoding.
    ///
    /// # Errors
    ///
    /// Errors if serializing the codec configuration fails.
    fn get_config<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error>;
}

/// Statically typed compression codec.
pub trait StaticCodec: Codec {
    /// Codec identifier.
    const CODEC_ID: &'static str;

    /// Instantiate a codec from a serialized `config`uration.
    ///
    /// The `config` must *not* contain an `id` field. If the `config` *may*
    /// contain one, use the [`codec_from_config_with_id`] helper function.
    ///
    /// The `config` *must* be compatible with JSON encoding.
    ///
    /// # Errors
    ///
    /// Errors if constructing the codec fails.
    fn from_config<'de, D: Deserializer<'de>>(config: D) -> Result<Self, D::Error>;
}

/// Dynamically typed compression codec.
///
/// Every codec that implements [`StaticCodec`] also implements [`DynCodec`].
pub trait DynCodec: Codec {
    /// Type object type for this codec.
    type Type: DynCodecType;

    /// Returns the type object for this codec.
    fn ty(&self) -> Self::Type;
}

/// Type object for dynamically typed compression codecs.
pub trait DynCodecType: 'static + Send + Sync {
    /// Type of the instances of this codec type object.
    type Codec: DynCodec<Type = Self>;

    /// Codec identifier.
    fn codec_id(&self) -> &str;

    /// Instantiate a codec of this type from a serialized `config`uration.
    ///
    /// The `config` must *not* contain an `id` field. If the `config` *may*
    /// contain one, use the [`codec_from_config_with_id`] helper function.
    ///
    /// The `config` *must* be compatible with JSON encoding.
    ///
    /// # Errors
    ///
    /// Errors if constructing the codec fails.
    fn codec_from_config<'de, D: Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<Self::Codec, D::Error>;
}

impl<T: StaticCodec> DynCodec for T {
    type Type = StaticCodecType<Self>;

    fn ty(&self) -> Self::Type {
        StaticCodecType::of()
    }
}

/// Type object for statically typed compression codecs.
pub struct StaticCodecType<T: StaticCodec> {
    _marker: PhantomData<T>,
}

impl<T: StaticCodec> StaticCodecType<T> {
    /// Statically obtain the type for a statically typed codec.
    #[must_use]
    pub const fn of() -> Self {
        Self {
            _marker: PhantomData::<T>,
        }
    }
}

impl<T: StaticCodec> DynCodecType for StaticCodecType<T> {
    type Codec = T;

    fn codec_id(&self) -> &str {
        T::CODEC_ID
    }

    fn codec_from_config<'de, D: Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<Self::Codec, D::Error> {
        T::from_config(config)
    }
}

/// Utility function to serialize a codec's config together with its
/// [`DynCodecType::codec_id`].
///
/// This function may be useful when implementing the [`Codec::get_config`]
/// method.
///
/// # Errors
///
/// Errors if serializing the codec configuration fails.
pub fn serialize_codec_config_with_id<T: Serialize, C: DynCodec, S: Serializer>(
    config: &T,
    codec: &C,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    #[derive(Serialize)]
    struct CodecConfigWithId<'a, T> {
        id: &'a str,
        #[serde(flatten)]
        config: &'a T,
    }

    CodecConfigWithId {
        id: codec.ty().codec_id(),
        config,
    }
    .serialize(serializer)
}

/// Utility function to instantiate a codec of the given `ty`, where the
/// `config` *may* still contain an `id` field.
///
/// If the `config` does *not* contain an `id` field, use
/// [`DynCodecType::codec_from_config`] instead.
///
/// # Errors
///
/// Errors if constructing the codec fails.
pub fn codec_from_config_with_id<'de, T: DynCodecType, D: Deserializer<'de>>(
    ty: &T,
    config: D,
) -> Result<T::Codec, D::Error> {
    let mut config = Value::deserialize(config)?;

    if let Some(config) = config.as_object_mut() {
        if let Some(id) = config.remove("id") {
            let codec_id = ty.codec_id();

            if !matches!(id, Value::String(ref id) if id == codec_id) {
                return Err(serde::de::Error::custom(format!(
                    "expected codec id {codec_id:?} but found {id}"
                )));
            }
        }
    }

    ty.codec_from_config(config)
        .map_err(serde::de::Error::custom)
}
