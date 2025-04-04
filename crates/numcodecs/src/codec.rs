use std::{borrow::Cow, error::Error, fmt, marker::PhantomData};

use schemars::{generate::SchemaSettings, json_schema, JsonSchema, Schema, SchemaGenerator};
use semver::{Version, VersionReq};
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
}

/// Statically typed compression codec.
pub trait StaticCodec: Codec {
    /// Codec identifier.
    const CODEC_ID: &'static str;

    /// Configuration type, from which the codec can be created infallibly.
    ///
    /// The `config` must *not* contain an `id` field.
    ///
    /// The config *must* be compatible with JSON encoding and have a schema.
    type Config<'de>: Serialize + Deserialize<'de> + JsonSchema;

    /// Instantiate a codec from its `config`uration.
    fn from_config(config: Self::Config<'_>) -> Self;

    /// Get the configuration for this codec.
    ///
    /// The [`StaticCodecConfig`] ensures that the returned config includes an
    /// `id` field with the codec's [`StaticCodec::CODEC_ID`].
    fn get_config(&self) -> StaticCodecConfig<Self>;
}

/// Dynamically typed compression codec.
///
/// Every codec that implements [`StaticCodec`] also implements [`DynCodec`].
pub trait DynCodec: Codec {
    /// Type object type for this codec.
    type Type: DynCodecType;

    /// Returns the type object for this codec.
    fn ty(&self) -> Self::Type;

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

/// Type object for dynamically typed compression codecs.
pub trait DynCodecType: 'static + Send + Sync {
    /// Type of the instances of this codec type object.
    type Codec: DynCodec<Type = Self>;

    /// Codec identifier.
    fn codec_id(&self) -> &str;

    /// JSON schema for the codec's configuration.
    fn codec_config_schema(&self) -> Schema;

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

    fn get_config<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        <T as StaticCodec>::get_config(self).serialize(serializer)
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

    fn codec_config_schema(&self) -> Schema {
        let mut settings = SchemaSettings::draft2020_12();
        // TODO: perhaps this could be done as a more generally applicable
        //       transformation instead
        settings.inline_subschemas = true;
        settings
            .into_generator()
            .into_root_schema_for::<T::Config<'static>>()
    }

    fn codec_from_config<'de, D: Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<Self::Codec, D::Error> {
        let config = T::Config::deserialize(config)?;
        Ok(T::from_config(config))
    }
}

/// Utility struct to serialize a [`StaticCodec`]'s [`StaticCodec::Config`]
/// together with its [`StaticCodec::CODEC_ID`]
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct StaticCodecConfig<'a, T: StaticCodec> {
    #[serde(default)]
    id: StaticCodecId<T>,
    /// The configuration parameters
    #[serde(flatten)]
    #[serde(borrow)]
    pub config: T::Config<'a>,
}

impl<'a, T: StaticCodec> StaticCodecConfig<'a, T> {
    /// Wraps the `config` so that it can be serialized together with its
    /// [`StaticCodec::CODEC_ID`]
    #[must_use]
    pub const fn new(config: T::Config<'a>) -> Self {
        Self {
            id: StaticCodecId::of(),
            config,
        }
    }
}

impl<'a, T: StaticCodec> From<&T::Config<'a>> for StaticCodecConfig<'a, T>
where
    T::Config<'a>: Clone,
{
    fn from(config: &T::Config<'a>) -> Self {
        Self::new(config.clone())
    }
}

struct StaticCodecId<T: StaticCodec>(PhantomData<T>);

impl<T: StaticCodec> StaticCodecId<T> {
    #[must_use]
    pub const fn of() -> Self {
        Self(PhantomData::<T>)
    }
}

impl<T: StaticCodec> Default for StaticCodecId<T> {
    fn default() -> Self {
        Self::of()
    }
}

impl<T: StaticCodec> Serialize for StaticCodecId<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        T::CODEC_ID.serialize(serializer)
    }
}

impl<'de, T: StaticCodec> Deserialize<'de> for StaticCodecId<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let id = Cow::<str>::deserialize(deserializer)?;
        let id = &*id;

        if id != T::CODEC_ID {
            return Err(serde::de::Error::custom(format!(
                "expected codec id {:?} but found {id:?}",
                T::CODEC_ID,
            )));
        }

        Ok(Self::of())
    }
}

/// Utility function to serialize a codec's config together with its
/// [`DynCodecType::codec_id`].
///
/// This function may be useful when implementing the [`DynCodec::get_config`]
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
    struct DynCodecConfigWithId<'a, T> {
        id: &'a str,
        #[serde(flatten)]
        config: &'a T,
    }

    DynCodecConfigWithId {
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

/// Marker type that represents the semantic version of a codec.
///
/// The codec's version can be decoupled from its implementation version to
/// allow implementation changes that have no effect on the codec's semantics
/// or encoded representation.
///
/// `StaticCodecVersion`s serialize transparently to their equivalent
/// [`Version`]s. On deserialization, the deserialized [`Version`] is checked
/// to be compatible (`^`) with the `StaticCodecVersion`, i.e. the
/// `StaticCodecVersion` must be of a the same or a newer compatible version.
pub struct StaticCodecVersion<const MAJOR: u64, const MINOR: u64, const PATCH: u64>;

impl<const MAJOR: u64, const MINOR: u64, const PATCH: u64> StaticCodecVersion<MAJOR, MINOR, PATCH> {
    /// Extract the semantic version.
    #[must_use]
    pub const fn version() -> Version {
        Version::new(MAJOR, MINOR, PATCH)
    }
}

#[expect(clippy::expl_impl_clone_on_copy)]
impl<const MAJOR: u64, const MINOR: u64, const PATCH: u64> Clone
    for StaticCodecVersion<MAJOR, MINOR, PATCH>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<const MAJOR: u64, const MINOR: u64, const PATCH: u64> Copy
    for StaticCodecVersion<MAJOR, MINOR, PATCH>
{
}

impl<const MAJOR: u64, const MINOR: u64, const PATCH: u64> fmt::Debug
    for StaticCodecVersion<MAJOR, MINOR, PATCH>
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        <semver::Version as fmt::Debug>::fmt(&Self::version(), fmt)
    }
}

impl<const MAJOR: u64, const MINOR: u64, const PATCH: u64> fmt::Display
    for StaticCodecVersion<MAJOR, MINOR, PATCH>
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        <semver::Version as fmt::Display>::fmt(&Self::version(), fmt)
    }
}

impl<const MAJOR: u64, const MINOR: u64, const PATCH: u64> Default
    for StaticCodecVersion<MAJOR, MINOR, PATCH>
{
    fn default() -> Self {
        Self
    }
}

impl<const MAJOR: u64, const MINOR: u64, const PATCH: u64> Serialize
    for StaticCodecVersion<MAJOR, MINOR, PATCH>
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Self::version().serialize(serializer)
    }
}

impl<'de, const MAJOR: u64, const MINOR: u64, const PATCH: u64> Deserialize<'de>
    for StaticCodecVersion<MAJOR, MINOR, PATCH>
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let version = Version::deserialize(deserializer)?;

        let requirement = VersionReq {
            comparators: vec![semver::Comparator {
                op: semver::Op::Caret,
                major: version.major,
                minor: Some(version.minor),
                patch: Some(version.patch),
                pre: version.pre,
            }],
        };

        if !requirement.matches(&Self::version()) {
            return Err(serde::de::Error::custom(format!(
                "{Self} does not fulfil {requirement}"
            )));
        }

        Ok(Self)
    }
}

impl<const MAJOR: u64, const MINOR: u64, const PATCH: u64> JsonSchema
    for StaticCodecVersion<MAJOR, MINOR, PATCH>
{
    fn schema_name() -> Cow<'static, str> {
        Cow::Borrowed("StaticCodecVersion")
    }

    fn schema_id() -> Cow<'static, str> {
        Cow::Borrowed(concat!(module_path!(), "::", "StaticCodecVersion"))
    }

    fn json_schema(_gen: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "type": "string",
            "pattern": r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$",
            "description": "A semver.org compliant semantic version number.",
        })
    }
}
