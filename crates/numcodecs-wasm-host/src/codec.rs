use std::{
    borrow::BorrowMut,
    sync::{Arc, OnceLock},
};

use ndarray::{ArrayBase, ArrayView, Data, Dimension};
use numcodecs::{AnyArray, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray};
use serde::Serializer;
use wasm_component_layer::{
    Enum, EnumType, List, ListType, Record, RecordType, ResourceOwn, Value, ValueType, Variant,
    VariantCase, VariantType,
};

use crate::{
    error::{GuestError, RuntimeError},
    plugin::CodecPlugin,
    wit::guest_error_from_wasm,
};

#[expect(clippy::module_name_repetitions)]
pub struct WasmCodec<P: BorrowMut<CodecPlugin> = CodecPlugin> {
    pub(crate) resource: ResourceOwn,
    pub(crate) plugin: P,
    pub(crate) instruction_counter: u64,
}

impl<P: BorrowMut<CodecPlugin>> WasmCodec<P> {
    #[must_use]
    pub const fn instruction_counter(&self) -> u64 {
        self.instruction_counter
    }

    #[expect(clippy::needless_pass_by_value)]
    pub fn encode(
        &mut self,
        data: AnyCowArray,
    ) -> Result<Result<AnyArray, GuestError>, RuntimeError> {
        self.process(
            data.view(),
            None,
            |plugin, arguments, results| plugin.ctx.call_func(&plugin.encode, arguments, results),
            |encoded| Ok(encoded.into_owned()),
        )
    }

    #[expect(clippy::needless_pass_by_value)]
    pub fn encode_into(
        &mut self,
        data: AnyCowArray,
        mut encoded: AnyArrayViewMut,
    ) -> Result<Result<(), GuestError>, RuntimeError> {
        self.process(
            data.view(),
            None,
            |plugin, arguments, results| plugin.ctx.call_func(&plugin.encode, arguments, results),
            |encoded_in| {
                encoded
                    .assign(&encoded_in)
                    .map_err(anyhow::Error::new)
                    .map_err(RuntimeError::from)
            },
        )
    }

    #[expect(clippy::needless_pass_by_value)]
    pub fn decode(
        &mut self,
        encoded: AnyCowArray,
    ) -> Result<Result<AnyArray, GuestError>, RuntimeError> {
        self.process(
            encoded.view(),
            None,
            |plugin, arguments, results| plugin.ctx.call_func(&plugin.decode, arguments, results),
            |decoded| Ok(decoded.into_owned()),
        )
    }

    pub fn decode_into(
        &mut self,
        encoded: AnyArrayView,
        mut decoded: AnyArrayViewMut,
    ) -> Result<Result<(), GuestError>, RuntimeError> {
        self.process(
            encoded,
            #[expect(clippy::unnecessary_to_owned)] // we need the lifetime extension
            Some((decoded.dtype(), &decoded.shape().to_vec())),
            |plugin, arguments, results| {
                plugin
                    .ctx
                    .call_func(&plugin.decode_into, arguments, results)
            },
            |decoded_in| {
                decoded
                    .assign(&decoded_in)
                    .map_err(anyhow::Error::new)
                    .map_err(RuntimeError::from)
            },
        )
    }

    pub fn get_config<S: Serializer>(&mut self, serializer: S) -> Result<S::Ok, S::Error> {
        let plugin: &mut CodecPlugin = self.plugin.borrow_mut();

        let resource = plugin
            .ctx
            .borrow_resource(&self.resource)
            .map_err(serde::ser::Error::custom)?;

        let arg = Value::Borrow(resource);
        let mut result = Value::U8(0);

        plugin
            .ctx
            .call_func(
                &plugin.get_config,
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
                },
            },
            value => {
                return Err(serde::ser::Error::custom(format!(
                    "unexpected get-config result value {value:?}"
                )))
            },
        };

        serde_transcode::transcode(&mut serde_json::Deserializer::from_str(&config), serializer)
    }

    pub fn drop(mut self) -> Result<(), RuntimeError> {
        let plugin: &mut CodecPlugin = self.plugin.borrow_mut();

        let result = plugin
            .ctx
            .drop_resource(&self.resource)
            .map_err(RuntimeError::from);

        // We need to forget here instead of using ManuallyDrop since we need
        //  both a mutable borrow to self.plugin and an immutable borrow to
        //  self.resource at the same time
        std::mem::forget(self);

        result
    }
}

impl<P: BorrowMut<CodecPlugin>> WasmCodec<P> {
    fn process<O>(
        &mut self,
        data: AnyArrayView,
        output_prototype: Option<(AnyArrayDType, &[usize])>,
        process: impl FnOnce(&mut CodecPlugin, &[Value], &mut [Value]) -> anyhow::Result<()>,
        with_result: impl for<'a> FnOnce(AnyArrayView<'a>) -> Result<O, RuntimeError>,
    ) -> Result<Result<O, GuestError>, RuntimeError> {
        let plugin: &mut CodecPlugin = self.plugin.borrow_mut();

        let resource = plugin
            .ctx
            .borrow_resource(&self.resource)?;

        let array = Self::array_into_wasm(data)?;

        let output_prototype = output_prototype
            .map(|(dtype, shape)| Self::array_prototype_into_wasm(dtype, shape))
            .transpose()?;

        let instruction_counter_pre = plugin
            .ctx
            .call_typed_u64_func(&plugin.instruction_counter)?;

        let mut result = Value::U8(0);

        process(
            plugin,
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

        let instruction_counter_post = plugin
            .ctx
            .call_typed_u64_func(&plugin.instruction_counter)?;
        self.instruction_counter += instruction_counter_post - instruction_counter_pre;

        match result {
            Value::Result(result) => match &*result {
                Ok(Some(Value::Record(record))) if &record.ty() == Self::any_array_ty() => {
                    Self::with_array_view_from_wasm_record(record, |array| {
                        Ok(Ok(with_result(array)?))
                    })
                },
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

    #[expect(clippy::too_many_lines)] // FIXME
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
                ArrayView::from_shape(
                    shape.as_slice(),
                    values.typed()?,
                )
                .map_err(anyhow::Error::new)?,
            ),
            1 => AnyArrayView::U16(
                ArrayView::from_shape(
                    shape.as_slice(),
                    values.typed()?,
                )
                .map_err(anyhow::Error::new)?,
            ),
            2 => AnyArrayView::U32(
                ArrayView::from_shape(
                    shape.as_slice(),
                    values.typed()?,
                )
                .map_err(anyhow::Error::new)?,
            ),
            3 => AnyArrayView::U64(
                ArrayView::from_shape(
                    shape.as_slice(),
                    values.typed()?,
                )
                .map_err(anyhow::Error::new)?,
            ),
            4 => AnyArrayView::I8(
                ArrayView::from_shape(
                    shape.as_slice(),
                    values.typed()?,
                )
                .map_err(anyhow::Error::new)?,
            ),
            5 => AnyArrayView::I16(
                ArrayView::from_shape(
                    shape.as_slice(),
                    values.typed()?,
                )
                .map_err(anyhow::Error::new)?,
            ),
            6 => AnyArrayView::I32(
                ArrayView::from_shape(
                    shape.as_slice(),
                    values.typed()?,
                )
                .map_err(anyhow::Error::new)?,
            ),
            7 => AnyArrayView::I64(
                ArrayView::from_shape(
                    shape.as_slice(),
                    values.typed()?,
                )
                .map_err(anyhow::Error::new)?,
            ),
            8 => AnyArrayView::F32(
                ArrayView::from_shape(
                    shape.as_slice(),
                    values.typed()?,
                )
                .map_err(anyhow::Error::new)?,
            ),
            9 => AnyArrayView::F64(
                ArrayView::from_shape(
                    shape.as_slice(),
                    values.typed()?,
                )
                .map_err(anyhow::Error::new)?,
            ),
            discriminant => {
                return Err(RuntimeError::from(anyhow::Error::msg(format!(
                    "process result buffer has an invalid variant [{discriminant}]:{:?}",
                    data.value().map(|v| v.ty())
                ))))
            },
        };

        with(array)
    }
}

impl<P: BorrowMut<CodecPlugin>> Drop for WasmCodec<P> {
    fn drop(&mut self) {
        let plugin: &mut CodecPlugin = self.plugin.borrow_mut();

        std::mem::drop(plugin.ctx.drop_resource(&self.resource));
    }
}
