use std::sync::{Arc, OnceLock};

use ndarray::{ArrayBase, ArrayView, Data, Dimension};
use numcodecs::{AnyArray, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray};
use schemars::Schema;
use serde::Serializer;
use wasm_component_layer::{
    AsContextMut, Enum, EnumType, Func, Instance, List, ListType, Record, RecordType, ResourceOwn,
    Value, ValueType, Variant, VariantCase, VariantType,
};

use crate::{
    component::WasmCodecComponent,
    error::{CodecError, RuntimeError},
    wit::guest_error_from_wasm,
};

/// Codec instantiated inside a WebAssembly component.
///
/// `WasmCodec` does not implement the [`Codec`][numcodecs::Codec],
/// [`DynCodec`][numcodecs::DynCodec], [`Clone`], or [`Drop`] traits itself so
/// that it can expose un-opinionated bindings. However, it provides methods
/// that can be used to implement these traits on a wrapper.
pub struct WasmCodec {
    // codec
    pub(crate) resource: ResourceOwn,
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

/// Methods for implementing the [`Codec`][numcodecs::Codec] trait
impl WasmCodec {
    #[expect(clippy::needless_pass_by_value)]
    /// Encodes the `data` and returns the result.
    ///
    /// The `ctx` must refer to the same store in which the component was
    /// instantiated.
    ///
    /// # Errors
    ///
    /// Errors with a
    /// - [`CodecError`] if encoding the buffer fails.
    /// - [`RuntimeError`] if interacting with the component fails.
    pub fn encode(
        &self,
        ctx: impl AsContextMut,
        data: AnyCowArray,
    ) -> Result<Result<AnyArray, CodecError>, RuntimeError> {
        self.process(
            ctx,
            data.view(),
            None,
            |ctx, arguments, results| self.encode.call(ctx, arguments, results),
            |encoded| Ok(encoded.into_owned()),
        )
    }

    #[expect(clippy::needless_pass_by_value)]
    /// Decodes the `encoded` data and returns the result.
    ///
    /// The `ctx` must refer to the same store in which the component was
    /// instantiated.
    ///
    /// # Errors
    ///
    /// Errors with a
    /// - [`CodecError`] if decoding the buffer fails.
    /// - [`RuntimeError`] if interacting with the component fails.
    pub fn decode(
        &self,
        ctx: impl AsContextMut,
        encoded: AnyCowArray,
    ) -> Result<Result<AnyArray, CodecError>, RuntimeError> {
        self.process(
            ctx,
            encoded.view(),
            None,
            |ctx, arguments, results| self.decode.call(ctx, arguments, results),
            |decoded| Ok(decoded.into_owned()),
        )
    }

    /// Decodes the `encoded` data and writes the result into the provided
    /// `decoded` output.
    ///
    /// The output must have the correct type and shape.
    ///
    /// The `ctx` must refer to the same store in which the component was
    /// instantiated.
    ///
    /// # Errors
    ///
    /// Errors with a
    /// - [`CodecError`] if decoding the buffer fails.
    /// - [`RuntimeError`] if interacting with the component fails.
    pub fn decode_into(
        &self,
        ctx: impl AsContextMut,
        encoded: AnyArrayView,
        mut decoded: AnyArrayViewMut,
    ) -> Result<Result<(), CodecError>, RuntimeError> {
        self.process(
            ctx,
            encoded,
            #[expect(clippy::unnecessary_to_owned)] // we need the lifetime extension
            Some((decoded.dtype(), &decoded.shape().to_vec())),
            |ctx, arguments, results| self.decode_into.call(ctx, arguments, results),
            |decoded_in| {
                decoded
                    .assign(&decoded_in)
                    .map_err(anyhow::Error::new)
                    .map_err(RuntimeError::from)
            },
        )
    }
}

/// Methods for implementing the [`DynCodec`][numcodecs::DynCodec] trait
impl WasmCodec {
    /// Returns the component object for this codec.
    #[must_use]
    pub fn ty(&self) -> WasmCodecComponent {
        WasmCodecComponent {
            codec_id: self.codec_id.clone(),
            codec_config_schema: self.codec_config_schema.clone(),
            from_config: self.from_config.clone(),
            encode: self.encode.clone(),
            decode: self.decode.clone(),
            decode_into: self.decode_into.clone(),
            get_config: self.get_config.clone(),
            instance: self.instance.clone(),
        }
    }

    /// Serializes the configuration parameters for this codec.
    ///
    /// The `ctx` must refer to the same store in which the component was
    /// instantiated.
    ///
    /// # Errors
    ///
    /// Errors if serializing the codec configuration or interacting with the
    /// component fails.
    pub fn get_config<S: Serializer>(
        &self,
        mut ctx: impl AsContextMut,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let resource = self
            .resource
            .borrow(&mut ctx)
            .map_err(serde::ser::Error::custom)?;

        let arg = Value::Borrow(resource);
        let mut result = Value::U8(0);

        self.get_config
            .call(
                &mut ctx,
                std::slice::from_ref(&arg),
                std::slice::from_mut(&mut result),
            )
            .map_err(serde::ser::Error::custom)?;

        let config = match result {
            Value::Result(result) => match &*result {
                Ok(Some(Value::String(config))) => config.clone(),
                Err(err) => match guest_error_from_wasm(err.as_ref()) {
                    Ok(err) => return Err(serde::ser::Error::custom(err)),
                    Err(err) => return Err(serde::ser::Error::custom(err)),
                },
                result => {
                    return Err(serde::ser::Error::custom(format!(
                        "unexpected get-config result value {result:?}"
                    )))
                }
            },
            value => {
                return Err(serde::ser::Error::custom(format!(
                    "unexpected get-config result value {value:?}"
                )))
            }
        };

        serde_transcode::transcode(&mut serde_json::Deserializer::from_str(&config), serializer)
    }
}

/// Methods for implementing the [`Clone`] trait
impl WasmCodec {
    /// Try cloning the codec by recreating it from its configuration.
    ///
    /// The `ctx` must refer to the same store in which the component was
    /// instantiated.
    ///
    /// # Errors
    ///
    /// Errors if serializing the codec configuration, constructing the new
    /// codec, or interacting with the component fails.
    pub fn try_clone(&self, mut ctx: impl AsContextMut) -> Result<Self, serde_json::Error> {
        let mut config = self.get_config(&mut ctx, serde_json::value::Serializer)?;

        if let Some(config) = config.as_object_mut() {
            config.remove("id");
        }

        let codec: Self = self.ty().codec_from_config(ctx, config)?;

        Ok(codec)
    }

    /// Try cloning the codec into a different context by recreating it from
    /// its configuration.
    ///
    /// The `ctx_from` must refer to the same store in which the component was
    /// instantiated.
    ///
    /// # Errors
    ///
    /// Errors if serializing the codec configuration, constructing the new
    /// codec, or interacting with the component fails.
    pub fn try_clone_into(
        &self,
        ctx_from: impl AsContextMut,
        ctx_into: impl AsContextMut,
    ) -> Result<Self, serde_json::Error> {
        let mut config = self.get_config(ctx_from, serde_json::value::Serializer)?;

        if let Some(config) = config.as_object_mut() {
            config.remove("id");
        }

        let codec: Self = self.ty().codec_from_config(ctx_into, config)?;

        Ok(codec)
    }
}

/// Methods for implementing the [`Drop`] trait
impl WasmCodec {
    /// Try dropping the codec.
    ///
    /// The `ctx` must refer to the same store in which the component was
    /// instantiated.
    ///
    /// # Errors
    ///
    /// Errors if the codec's resource is borrowed or has already been dropped.
    pub fn try_drop(&self, ctx: impl AsContextMut) -> Result<(), RuntimeError> {
        self.resource.drop(ctx).map_err(RuntimeError::from)
    }
}

impl WasmCodec {
    fn process<O, C: AsContextMut>(
        &self,
        mut ctx: C,
        data: AnyArrayView,
        output_prototype: Option<(AnyArrayDType, &[usize])>,
        process: impl FnOnce(&mut C, &[Value], &mut [Value]) -> anyhow::Result<()>,
        with_result: impl for<'a> FnOnce(AnyArrayView<'a>) -> Result<O, RuntimeError>,
    ) -> Result<Result<O, CodecError>, RuntimeError> {
        let resource = self.resource.borrow(&mut ctx)?;

        let array = Self::array_into_wasm(data)?;

        let output_prototype = output_prototype
            .map(|(dtype, shape)| Self::array_prototype_into_wasm(dtype, shape))
            .transpose()?;

        let mut result = Value::U8(0);

        process(
            &mut ctx,
            &match output_prototype {
                None => vec![Value::Borrow(resource), Value::Record(array)],
                Some(output) => vec![
                    Value::Borrow(resource),
                    Value::Record(array),
                    Value::Record(output),
                ],
            },
            std::slice::from_mut(&mut result),
        )?;

        match result {
            Value::Result(result) => match &*result {
                Ok(Some(Value::Record(record))) if &record.ty() == Self::any_array_ty() => {
                    Self::with_array_view_from_wasm_record(record, |array| {
                        Ok(Ok(with_result(array)?))
                    })
                }
                Err(err) => guest_error_from_wasm(err.as_ref()).map(Err),
                result => Err(RuntimeError::from(anyhow::Error::msg(format!(
                    "unexpected process result value {result:?}"
                )))),
            },
            value => Err(RuntimeError::from(anyhow::Error::msg(format!(
                "unexpected process result value {value:?}"
            )))),
        }
    }

    fn any_array_data_ty() -> &'static VariantType {
        static ANY_ARRAY_DATA_TY: OnceLock<VariantType> = OnceLock::new();

        #[expect(clippy::expect_used)]
        // FIXME: use OnceLock::get_or_try_init,
        //        blocked on https://github.com/rust-lang/rust/issues/109737
        ANY_ARRAY_DATA_TY.get_or_init(|| {
            VariantType::new(
                None,
                [
                    VariantCase::new("u8", Some(ValueType::List(ListType::new(ValueType::U8)))),
                    VariantCase::new("u16", Some(ValueType::List(ListType::new(ValueType::U16)))),
                    VariantCase::new("u32", Some(ValueType::List(ListType::new(ValueType::U32)))),
                    VariantCase::new("u64", Some(ValueType::List(ListType::new(ValueType::U64)))),
                    VariantCase::new("i8", Some(ValueType::List(ListType::new(ValueType::S8)))),
                    VariantCase::new("i16", Some(ValueType::List(ListType::new(ValueType::S16)))),
                    VariantCase::new("i32", Some(ValueType::List(ListType::new(ValueType::S32)))),
                    VariantCase::new("i64", Some(ValueType::List(ListType::new(ValueType::S64)))),
                    VariantCase::new("f32", Some(ValueType::List(ListType::new(ValueType::F32)))),
                    VariantCase::new("f64", Some(ValueType::List(ListType::new(ValueType::F64)))),
                ],
            )
            .expect("constructing the any-array-data variant type must not fail")
        })
    }

    fn any_array_ty() -> &'static RecordType {
        static ANY_ARRAY_TY: OnceLock<RecordType> = OnceLock::new();

        #[expect(clippy::expect_used)]
        // FIXME: use OnceLock::get_or_try_init,
        //        blocked on https://github.com/rust-lang/rust/issues/109737
        ANY_ARRAY_TY.get_or_init(|| {
            RecordType::new(
                None,
                [
                    (
                        "data",
                        ValueType::Variant(Self::any_array_data_ty().clone()),
                    ),
                    ("shape", ValueType::List(ListType::new(ValueType::U32))),
                ],
            )
            .expect("constructing the any-array record type must not fail")
        })
    }

    #[expect(clippy::needless_pass_by_value)]
    fn array_into_wasm(array: AnyArrayView) -> Result<Record, RuntimeError> {
        fn list_from_standard_layout<'a, T: 'static + Copy, S: Data<Elem = T>, D: Dimension>(
            array: &'a ArrayBase<S, D>,
        ) -> List
        where
            List: From<&'a [T]> + From<Arc<[T]>>,
        {
            #[expect(clippy::option_if_let_else)]
            if let Some(slice) = array.as_slice() {
                List::from(slice)
            } else {
                List::from(Arc::from(array.iter().copied().collect::<Vec<T>>()))
            }
        }

        let any_array_data_ty = Self::any_array_data_ty().clone();

        let data = match &array {
            AnyArrayView::U8(array) => Variant::new(
                any_array_data_ty,
                0,
                Some(Value::List(list_from_standard_layout(array))),
            ),
            AnyArrayView::U16(array) => Variant::new(
                any_array_data_ty,
                1,
                Some(Value::List(list_from_standard_layout(array))),
            ),
            AnyArrayView::U32(array) => Variant::new(
                any_array_data_ty,
                2,
                Some(Value::List(list_from_standard_layout(array))),
            ),
            AnyArrayView::U64(array) => Variant::new(
                any_array_data_ty,
                3,
                Some(Value::List(list_from_standard_layout(array))),
            ),
            AnyArrayView::I8(array) => Variant::new(
                any_array_data_ty,
                4,
                Some(Value::List(list_from_standard_layout(array))),
            ),
            AnyArrayView::I16(array) => Variant::new(
                any_array_data_ty,
                5,
                Some(Value::List(list_from_standard_layout(array))),
            ),
            AnyArrayView::I32(array) => Variant::new(
                any_array_data_ty,
                6,
                Some(Value::List(list_from_standard_layout(array))),
            ),
            AnyArrayView::I64(array) => Variant::new(
                any_array_data_ty,
                7,
                Some(Value::List(list_from_standard_layout(array))),
            ),
            AnyArrayView::F32(array) => Variant::new(
                any_array_data_ty,
                8,
                Some(Value::List(list_from_standard_layout(array))),
            ),
            AnyArrayView::F64(array) => Variant::new(
                any_array_data_ty,
                9,
                Some(Value::List(list_from_standard_layout(array))),
            ),
            array => Err(anyhow::Error::msg(format!(
                "unknown array dtype type {}",
                array.dtype()
            ))),
        }?;

        let shape = array
            .shape()
            .iter()
            .map(|s| u32::try_from(*s))
            .collect::<Result<Vec<_>, _>>()
            .map_err(anyhow::Error::new)?;
        let shape = List::from(Arc::from(shape));

        Record::new(
            Self::any_array_ty().clone(),
            [
                ("data", Value::Variant(data)),
                ("shape", Value::List(shape)),
            ],
        )
        .map_err(RuntimeError::from)
    }

    fn any_array_dtype_ty() -> &'static EnumType {
        static ANY_ARRAY_DTYPE_TY: OnceLock<EnumType> = OnceLock::new();

        #[expect(clippy::expect_used)]
        // FIXME: use OnceLock::get_or_try_init,
        //        blocked on https://github.com/rust-lang/rust/issues/109737
        ANY_ARRAY_DTYPE_TY.get_or_init(|| {
            EnumType::new(
                None,
                [
                    "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f32", "f64",
                ],
            )
            .expect("constructing the any-array-dtype enum type must not fail")
        })
    }

    fn any_array_prototype_ty() -> &'static RecordType {
        static ANY_ARRAY_PROTOTYPE_TY: OnceLock<RecordType> = OnceLock::new();

        #[expect(clippy::expect_used)]
        // FIXME: use OnceLock::get_or_try_init,
        //        blocked on https://github.com/rust-lang/rust/issues/109737
        ANY_ARRAY_PROTOTYPE_TY.get_or_init(|| {
            RecordType::new(
                None,
                [
                    ("dtype", ValueType::Enum(Self::any_array_dtype_ty().clone())),
                    ("shape", ValueType::List(ListType::new(ValueType::U32))),
                ],
            )
            .expect("constructing the any-array-prototype record type must not fail")
        })
    }

    fn array_prototype_into_wasm(
        dtype: AnyArrayDType,
        shape: &[usize],
    ) -> Result<Record, RuntimeError> {
        let any_array_dtype_ty = Self::any_array_dtype_ty().clone();

        let dtype = match dtype {
            AnyArrayDType::U8 => Enum::new(any_array_dtype_ty, 0),
            AnyArrayDType::U16 => Enum::new(any_array_dtype_ty, 1),
            AnyArrayDType::U32 => Enum::new(any_array_dtype_ty, 2),
            AnyArrayDType::U64 => Enum::new(any_array_dtype_ty, 3),
            AnyArrayDType::I8 => Enum::new(any_array_dtype_ty, 4),
            AnyArrayDType::I16 => Enum::new(any_array_dtype_ty, 5),
            AnyArrayDType::I32 => Enum::new(any_array_dtype_ty, 6),
            AnyArrayDType::I64 => Enum::new(any_array_dtype_ty, 7),
            AnyArrayDType::F32 => Enum::new(any_array_dtype_ty, 8),
            AnyArrayDType::F64 => Enum::new(any_array_dtype_ty, 9),
            dtype => Err(anyhow::Error::msg(format!(
                "unknown array dtype type {dtype}"
            ))),
        }?;

        let shape = shape
            .iter()
            .map(|s| u32::try_from(*s))
            .collect::<Result<Vec<_>, _>>()
            .map_err(anyhow::Error::new)?;
        let shape = List::from(Arc::from(shape));

        Record::new(
            Self::any_array_prototype_ty().clone(),
            [("dtype", Value::Enum(dtype)), ("shape", Value::List(shape))],
        )
        .map_err(RuntimeError::from)
    }

    fn with_array_view_from_wasm_record<O>(
        record: &Record,
        with: impl for<'a> FnOnce(AnyArrayView<'a>) -> Result<O, RuntimeError>,
    ) -> Result<O, RuntimeError> {
        let Some(Value::List(shape)) = record.field("shape") else {
            return Err(RuntimeError::from(anyhow::Error::msg(format!(
                "process result record {record:?} is missing shape field"
            ))));
        };
        let shape = shape
            .typed::<u32>()?
            .iter()
            .copied()
            .map(usize::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(anyhow::Error::new)?;

        let Some(Value::Variant(data)) = record.field("data") else {
            return Err(RuntimeError::from(anyhow::Error::msg(format!(
                "process result record {record:?} is missing data field"
            ))));
        };
        let Some(Value::List(values)) = data.value() else {
            return Err(RuntimeError::from(anyhow::Error::msg(format!(
                "process result buffer has an invalid variant type {:?}",
                data.value().map(|v| v.ty())
            ))));
        };

        let array = match data.discriminant() {
            0 => AnyArrayView::U8(
                ArrayView::from_shape(shape.as_slice(), values.typed()?)
                    .map_err(anyhow::Error::new)?,
            ),
            1 => AnyArrayView::U16(
                ArrayView::from_shape(shape.as_slice(), values.typed()?)
                    .map_err(anyhow::Error::new)?,
            ),
            2 => AnyArrayView::U32(
                ArrayView::from_shape(shape.as_slice(), values.typed()?)
                    .map_err(anyhow::Error::new)?,
            ),
            3 => AnyArrayView::U64(
                ArrayView::from_shape(shape.as_slice(), values.typed()?)
                    .map_err(anyhow::Error::new)?,
            ),
            4 => AnyArrayView::I8(
                ArrayView::from_shape(shape.as_slice(), values.typed()?)
                    .map_err(anyhow::Error::new)?,
            ),
            5 => AnyArrayView::I16(
                ArrayView::from_shape(shape.as_slice(), values.typed()?)
                    .map_err(anyhow::Error::new)?,
            ),
            6 => AnyArrayView::I32(
                ArrayView::from_shape(shape.as_slice(), values.typed()?)
                    .map_err(anyhow::Error::new)?,
            ),
            7 => AnyArrayView::I64(
                ArrayView::from_shape(shape.as_slice(), values.typed()?)
                    .map_err(anyhow::Error::new)?,
            ),
            8 => AnyArrayView::F32(
                ArrayView::from_shape(shape.as_slice(), values.typed()?)
                    .map_err(anyhow::Error::new)?,
            ),
            9 => AnyArrayView::F64(
                ArrayView::from_shape(shape.as_slice(), values.typed()?)
                    .map_err(anyhow::Error::new)?,
            ),
            discriminant => {
                return Err(RuntimeError::from(anyhow::Error::msg(format!(
                    "process result buffer has an invalid variant [{discriminant}]:{:?}",
                    data.value().map(|v| v.ty())
                ))))
            }
        };

        with(array)
    }
}
