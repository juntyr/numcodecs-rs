use std::{any::Any, error::Error, fmt};

use schemars::Schema;
use serde::{Deserializer, Serialize, Serializer};

use crate::{AnyArray, AnyArrayView, AnyArrayViewMut, AnyCowArray, Codec, DynCodec, DynCodecType};

/// Type-erased [`Error`] type.
pub struct ErasedError {
    error: Box<dyn 'static + Error + Send + Sync>,
}

impl ErasedError {
    /// Erase the type information of the concrete `err`or.
    pub fn new<T: 'static + Error + Send + Sync>(err: T) -> Self {
        Self {
            error: Box::new(err),
        }
    }
}

impl fmt::Debug for ErasedError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.error, fmt)
    }
}

impl fmt::Display for ErasedError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.error, fmt)
    }
}

impl Error for ErasedError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.error.source()
    }
}

/// Type-erased dynamically typed compression codec.
pub struct ErasedDynCodec {
    codec: Box<dyn ErasedDynCodecDispatch>,
}

impl ErasedDynCodec {
    /// Erase the type information of the concrete `codec`.
    pub fn new<T: DynCodec>(codec: T) -> Self {
        Self {
            codec: Box::new(codec),
        }
    }

    /// Try to downcast into a concretely-typed codec.
    ///
    /// # Errors
    ///
    /// Returns `self` if the type-erased codec is not of the concrete type.
    pub fn downcast<T: DynCodec>(self) -> Result<T, Self> {
        if self.codec.erased_as_any().is::<T>() {
            let raw = Box::into_raw(self.codec);
            #[expect(unsafe_code)]
            // SAFETY: we have checked that self.codec is of type T
            let codec = unsafe { Box::from_raw(raw.cast::<T>()) };
            Ok(*codec)
        } else {
            Err(self)
        }
    }

    /// Try to downcast to a concretely-typed codec reference.
    #[must_use]
    pub fn downcast_ref<T: DynCodec>(&self) -> Option<&T> {
        self.codec.erased_as_any().downcast_ref()
    }

    /// Try to downcast to a concretely-typed mutable codec reference.
    #[must_use]
    pub fn downcast_mut<T: DynCodec>(&mut self) -> Option<&mut T> {
        self.codec.erased_as_any_mut().downcast_mut()
    }
}

impl Clone for ErasedDynCodec {
    fn clone(&self) -> Self {
        Self {
            codec: self.codec.erased_clone(),
        }
    }
}

impl Codec for ErasedDynCodec {
    type Error = ErasedError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        self.codec.erased_encode(data)
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        self.codec.erased_decode(encoded)
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        self.codec.erased_decode_into(encoded, decoded)
    }
}

impl DynCodec for ErasedDynCodec {
    type Type = ErasedDynCodecType;

    fn ty(&self) -> Self::Type {
        ErasedDynCodecType {
            ty: self.codec.erased_ty(),
        }
    }

    fn get_config<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        erased_serde::serialize(self.codec.erased_as_serialize(), serializer)
    }
}

/// Type-erased dynamically typed compression codec type.
pub struct ErasedDynCodecType {
    ty: Box<dyn ErasedDynCodecTypeDispatch>,
}

impl ErasedDynCodecType {
    /// Erase the type information of the concrete codec `ty`pe.
    pub fn new<T: DynCodecType>(ty: T) -> Self {
        Self { ty: Box::new(ty) }
    }

    /// Try to downcast into a concretely-typed codec type.
    ///
    /// # Errors
    ///
    /// Returns `self` if the type-erased codec type is not of the concrete
    /// type.
    pub fn downcast<T: DynCodecType>(self) -> Result<T, Self> {
        if self.ty.erased_as_any().is::<T>() {
            let raw = Box::into_raw(self.ty);
            #[expect(unsafe_code)]
            // SAFETY: we have checked that self.ty is of type T
            let ty = unsafe { Box::from_raw(raw.cast::<T>()) };
            Ok(*ty)
        } else {
            Err(self)
        }
    }

    /// Try to downcast to a concretely-typed codec type reference.
    #[must_use]
    pub fn downcast_ref<T: DynCodecType>(&self) -> Option<&T> {
        self.ty.erased_as_any().downcast_ref()
    }

    /// Try to downcast to a concretely-typed mutable codec type reference.
    #[must_use]
    pub fn downcast_mut<T: DynCodecType>(&mut self) -> Option<&mut T> {
        self.ty.erased_as_any_mut().downcast_mut()
    }
}

impl DynCodecType for ErasedDynCodecType {
    type Codec = ErasedDynCodec;

    fn codec_id(&self) -> &str {
        self.ty.erased_codec_id()
    }

    fn codec_config_schema(&self) -> Schema {
        self.ty.erased_codec_config_schema()
    }

    fn codec_from_config<'de, D: Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<Self::Codec, D::Error> {
        match self
            .ty
            .erased_codec_from_config(&mut <dyn erased_serde::Deserializer>::erase(config))
        {
            Ok(codec) => Ok(ErasedDynCodec { codec }),
            Err(err) => Err(serde::de::Error::custom(err)), // TODO: improve
        }
    }
}

trait ErasedDynCodecDispatch: 'static + Send + Sync {
    fn erased_encode(&self, data: AnyCowArray) -> Result<AnyArray, ErasedError>;
    fn erased_decode(&self, encoded: AnyCowArray) -> Result<AnyArray, ErasedError>;
    fn erased_decode_into(
        &self,
        encoded: AnyArrayView,
        decoded: AnyArrayViewMut,
    ) -> Result<(), ErasedError>;

    fn erased_clone(&self) -> Box<dyn ErasedDynCodecDispatch>;

    fn erased_ty(&self) -> Box<dyn ErasedDynCodecTypeDispatch>;

    fn erased_as_any(&self) -> &dyn Any;
    fn erased_as_any_mut(&mut self) -> &mut dyn Any;

    fn erased_as_serialize(&self) -> &dyn erased_serde::Serialize;
}

trait ErasedDynCodecTypeDispatch: 'static + Send + Sync {
    fn erased_codec_id(&self) -> &str;
    fn erased_codec_config_schema(&self) -> Schema;
    fn erased_codec_from_config(
        &self,
        config: &mut dyn erased_serde::Deserializer,
    ) -> Result<Box<dyn ErasedDynCodecDispatch>, erased_serde::Error>;

    fn erased_as_any(&self) -> &dyn Any;
    fn erased_as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: DynCodec> ErasedDynCodecDispatch for T {
    fn erased_encode(&self, data: AnyCowArray) -> Result<AnyArray, ErasedError> {
        Codec::encode(self, data).map_err(ErasedError::new)
    }

    fn erased_decode(&self, encoded: AnyCowArray) -> Result<AnyArray, ErasedError> {
        Codec::decode(self, encoded).map_err(ErasedError::new)
    }

    fn erased_decode_into(
        &self,
        encoded: AnyArrayView,
        decoded: AnyArrayViewMut,
    ) -> Result<(), ErasedError> {
        Codec::decode_into(self, encoded, decoded).map_err(ErasedError::new)
    }

    fn erased_clone(&self) -> Box<dyn ErasedDynCodecDispatch> {
        Box::new(Clone::clone(self))
    }

    fn erased_ty(&self) -> Box<dyn ErasedDynCodecTypeDispatch> {
        Box::new(DynCodec::ty(self))
    }

    fn erased_as_any(&self) -> &dyn Any {
        self
    }

    fn erased_as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn erased_as_serialize(&self) -> &dyn erased_serde::Serialize {
        #[repr(transparent)]
        struct SerializeDynCodec<T: DynCodec>(T);

        impl<T: DynCodec> Serialize for SerializeDynCodec<T> {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                DynCodec::get_config(&self.0, serializer)
            }
        }

        #[expect(unsafe_code)]
        // SAFETY: SerializeDynCodec is a transparent newtype around Self
        unsafe {
            &*std::ptr::from_ref(self).cast::<SerializeDynCodec<Self>>()
        }
    }
}

impl<T: DynCodecType> ErasedDynCodecTypeDispatch for T {
    fn erased_codec_id(&self) -> &str {
        DynCodecType::codec_id(self)
    }

    fn erased_codec_config_schema(&self) -> Schema {
        DynCodecType::codec_config_schema(self)
    }

    fn erased_codec_from_config(
        &self,
        config: &mut dyn erased_serde::Deserializer,
    ) -> Result<Box<dyn ErasedDynCodecDispatch>, erased_serde::Error> {
        match DynCodecType::codec_from_config(self, config) {
            Ok(codec) => Ok(Box::new(codec)),
            Err(err) => Err(err),
        }
    }

    fn erased_as_any(&self) -> &dyn Any {
        self
    }

    fn erased_as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
