use std::{error::Error, sync::Arc};

use numcodecs::{Codec, DynCodec, DynCodecType, ErasedDynCodec, ErasedDynCodecType};
use numcodecs_registry::Registry;
use wasm_component_layer::{
    AsContext, AsContextMut, Func, FuncType, Linker, List, ListType, Record, RecordType,
    ResourceOwn, ResourceType, ResultType, ResultValue, TypeIdentifier, Value, ValueType,
};

use crate::{WasmCodec, wit::NumcodecsWitInterfaces};

/// Adds the `registry` to the `linker` to define the `numcodecs:abc/registry`
/// interface.
///
/// # Errors
///
/// Errors if adding the `registry` to the `linker` fails.
#[expect(clippy::too_many_lines)] // FIXME
pub fn add_registry_to_linker(
    linker: &mut Linker,
    mut ctx: impl AsContextMut,
    registry: impl Registry,
) -> Result<(), anyhow::Error> {
    let NumcodecsWitInterfaces {
        registry: numcodecs_registry_interface,
        ..
    } = NumcodecsWitInterfaces::get();

    let registry = Arc::new(registry);

    let numcodecs_types_error_record = RecordType::new(
        None, // skip name to keep plain data types flexible
        [
            ("message", ValueType::String),
            ("chain", ValueType::List(ListType::new(ValueType::String))),
        ],
    )?;

    let numcodecs_registry_instance =
        linker.define_instance(numcodecs_registry_interface.clone())?;

    let numcodecs_registry_external_codec_resource = ResourceType::with_destructor(
        ctx.as_context_mut(),
        Some(TypeIdentifier::new(
            "external-codec",
            Some(numcodecs_registry_interface.clone()),
        )),
        |_ctx, codec: ErasedDynCodec| {
            std::mem::drop(codec);
            Ok(())
        },
    )?;

    numcodecs_registry_instance.define_resource(
        "external-codec",
        numcodecs_registry_external_codec_resource.clone(),
    )?;

    let numcodecs_registry_external_codec_type_resource = ResourceType::with_destructor(
        ctx.as_context_mut(),
        Some(TypeIdentifier::new(
            "external-codec-type",
            Some(numcodecs_registry_interface.clone()),
        )),
        |_ctx, codec_ty: ErasedDynCodecType| {
            std::mem::drop(codec_ty);
            Ok(())
        },
    )?;

    numcodecs_registry_instance.define_resource(
        "external-codec-type",
        numcodecs_registry_external_codec_type_resource.clone(),
    )?;

    let any_array_record = WasmCodec::any_array_ty().clone();

    let any_array_result = ResultType::new(
        Some(ValueType::Record(any_array_record.clone())),
        Some(ValueType::Record(numcodecs_types_error_record.clone())),
    );

    let my_any_array_result = any_array_result.clone();
    let my_numcodecs_types_error_record = numcodecs_types_error_record.clone();
    let external_codec_encode = Func::new(
        ctx.as_context_mut(),
        FuncType::new(
            [
                ValueType::Borrow(numcodecs_registry_external_codec_resource.clone()),
                ValueType::Record(any_array_record.clone()),
            ],
            [ValueType::Result(any_array_result.clone())],
        ),
        move |ctx, args, results| {
            let [Value::Borrow(codec), Value::Record(data)] = args else {
                anyhow::bail!(
                    "invalid numcodecs:abc/registry#[method]external-codec.encode arguments"
                );
            };

            let [result] = results else {
                anyhow::bail!(
                    "invalid numcodecs:abc/registry#[method]external-codec.encode results"
                );
            };

            let encoded = WasmCodec::with_array_view_from_wasm_record(data, |data| {
                let ctx = ctx.as_context();
                let codec: &ErasedDynCodec = codec.rep(&ctx)?;

                let encoded = codec.encode(data.cow()).map_err(anyhow::Error::new)?;
                Ok(encoded)
            });

            let encoded = match encoded {
                Ok(encoded) => Ok(WasmCodec::array_into_wasm(encoded.view())?),
                Err(err) => Err(into_wit_error(err, &my_numcodecs_types_error_record)?),
            };

            let res = match encoded {
                Ok(encoded) => Ok(Some(Value::Record(encoded))),
                Err(err) => Err(Some(Value::Record(err))),
            };

            *result = Value::Result(ResultValue::new(my_any_array_result.clone(), res)?);

            Ok(())
        },
    );
    numcodecs_registry_instance.define_func("[method]external-codec.encode", external_codec_encode)?;

    let my_any_array_result = any_array_result.clone();
    let my_numcodecs_types_error_record = numcodecs_types_error_record.clone();
    let external_codec_decode = Func::new(
        ctx.as_context_mut(),
        FuncType::new(
            [
                ValueType::Borrow(numcodecs_registry_external_codec_resource.clone()),
                ValueType::Record(any_array_record.clone()),
            ],
            [ValueType::Result(any_array_result.clone())],
        ),
        move |ctx, args, results| {
            let [Value::Borrow(codec), Value::Record(encoded)] = args else {
                anyhow::bail!(
                    "invalid numcodecs:abc/registry#[method]external-codec.decode arguments"
                );
            };

            let [result] = results else {
                anyhow::bail!(
                    "invalid numcodecs:abc/registry#[method]external-codec.decode results"
                );
            };

            let decoded = WasmCodec::with_array_view_from_wasm_record(encoded, |encoded| {
                let ctx = ctx.as_context();
                let codec: &ErasedDynCodec = codec.rep(&ctx)?;

                let decoded = codec.decode(encoded.cow()).map_err(anyhow::Error::new)?;
                Ok(decoded)
            });

            let decoded = match decoded {
                Ok(decoded) => Ok(WasmCodec::array_into_wasm(decoded.view())?),
                Err(err) => Err(into_wit_error(err, &my_numcodecs_types_error_record)?),
            };

            let res = match decoded {
                Ok(decoded) => Ok(Some(Value::Record(decoded))),
                Err(err) => Err(Some(Value::Record(err))),
            };

            *result = Value::Result(ResultValue::new(my_any_array_result.clone(), res)?);

            Ok(())
        },
    );
    numcodecs_registry_instance.define_func("[method]external-codec.decode", external_codec_decode)?;

    let any_array_prototype_record = WasmCodec::any_array_prototype_ty().clone();

    let my_any_array_result = any_array_result.clone();
    let my_numcodecs_types_error_record = numcodecs_types_error_record.clone();
    let external_codec_decode_into = Func::new(
        ctx.as_context_mut(),
        FuncType::new(
            [
                ValueType::Borrow(numcodecs_registry_external_codec_resource.clone()),
                ValueType::Record(any_array_record),
                ValueType::Record(any_array_prototype_record),
            ],
            [ValueType::Result(any_array_result)],
        ),
        move |ctx, args, results| {
            let [
                Value::Borrow(codec),
                Value::Record(encoded),
                Value::Record(decoded),
            ] = args
            else {
                anyhow::bail!(
                    "invalid numcodecs:abc/registry#[method]external-codec.decode-into arguments"
                );
            };

            let [result] = results else {
                anyhow::bail!(
                    "invalid numcodecs:abc/registry#[method]external-codec.decode-into results"
                );
            };

            let mut decoded = WasmCodec::array_prototype_from_wasm_record(decoded)?;

            let res = WasmCodec::with_array_view_from_wasm_record(encoded, |encoded| {
                let ctx = ctx.as_context();
                let codec: &ErasedDynCodec = codec.rep(&ctx)?;

                codec
                    .decode_into(encoded, decoded.view_mut())
                    .map_err(anyhow::Error::new)?;
                Ok(())
            });

            let decoded = match res {
                Ok(()) => Ok(WasmCodec::array_into_wasm(decoded.view())?),
                Err(err) => Err(into_wit_error(err, &my_numcodecs_types_error_record)?),
            };

            let res = match decoded {
                Ok(decoded) => Ok(Some(Value::Record(decoded))),
                Err(err) => Err(Some(Value::Record(err))),
            };

            *result = Value::Result(ResultValue::new(my_any_array_result.clone(), res)?);

            Ok(())
        },
    );
    numcodecs_registry_instance
        .define_func("[method]external-codec.decode-into", external_codec_decode_into)?;

    let my_numcodecs_registry_codec_resource = numcodecs_registry_external_codec_resource.clone();
    let external_codec_clone = Func::new(
        ctx.as_context_mut(),
        FuncType::new(
            [ValueType::Borrow(numcodecs_registry_external_codec_resource.clone())],
            [ValueType::Own(numcodecs_registry_external_codec_resource.clone())],
        ),
        move |ctx, args, results| {
            let [Value::Borrow(codec)] = args else {
                anyhow::bail!(
                    "invalid numcodecs:abc/registry#[method]external-codec.clone arguments"
                );
            };

            let [result] = results else {
                anyhow::bail!(
                    "invalid numcodecs:abc/registry#[method]external-codec.clone results"
                );
            };

            let codec = {
                let ctx = ctx.as_context();
                let codec: &ErasedDynCodec = codec.rep(&ctx)?;
                codec.clone()
            };

            *result = Value::Own(ResourceOwn::new(
                ctx,
                codec,
                my_numcodecs_registry_codec_resource.clone(),
            )?);

            Ok(())
        },
    );
    numcodecs_registry_instance.define_func("[method]external-codec.clone", external_codec_clone)?;

    let string_result = ResultType::new(
        Some(ValueType::String),
        Some(ValueType::Record(numcodecs_types_error_record.clone())),
    );

    let my_numcodecs_types_error_record = numcodecs_types_error_record.clone();
    let external_codec_get_config = Func::new(
        ctx.as_context_mut(),
        FuncType::new(
            [ValueType::Borrow(numcodecs_registry_external_codec_resource.clone())],
            [ValueType::Result(string_result.clone())],
        ),
        move |ctx, args, results| {
            let [Value::Borrow(codec)] = args else {
                anyhow::bail!(
                    "invalid numcodecs:abc/registry#[method]external-codec.get-config arguments"
                );
            };

            let [result] = results else {
                anyhow::bail!(
                    "invalid numcodecs:abc/registry#[method]external-codec.get-config results"
                );
            };

            let config = {
                let ctx = ctx.as_context();
                let codec: &ErasedDynCodec = codec.rep(&ctx)?;

                let mut config_bytes = Vec::new();
                match codec.get_config(&mut serde_json::Serializer::new(&mut config_bytes)) {
                    Ok(()) => match String::from_utf8(config_bytes) {
                        Ok(config) => Ok(config),
                        Err(err) => Err(into_wit_error(err, &my_numcodecs_types_error_record)?),
                    },
                    Err(err) => Err(into_wit_error(err, &my_numcodecs_types_error_record)?),
                }
            };

            let res = match config {
                Ok(config) => Ok(Some(Value::String(Arc::from(config)))),
                Err(err) => Err(Some(Value::Record(err))),
            };

            *result = Value::Result(ResultValue::new(string_result.clone(), res)?);

            Ok(())
        },
    );
    numcodecs_registry_instance
        .define_func("[method]external-codec.get-config", external_codec_get_config)?;

    let my_numcodecs_registry_codec_type_resource = numcodecs_registry_external_codec_type_resource.clone();
    let external_codec_ty = Func::new(
        ctx.as_context_mut(),
        FuncType::new(
            [ValueType::Borrow(numcodecs_registry_external_codec_resource.clone())],
            [ValueType::Own(
                numcodecs_registry_external_codec_type_resource.clone(),
            )],
        ),
        move |ctx, args, results| {
            let [Value::Borrow(codec)] = args else {
                anyhow::bail!(
                    "invalid numcodecs:abc/registry#[method]external-codec.ty arguments"
                );
            };

            let [result] = results else {
                anyhow::bail!("invalid numcodecs:abc/registry#[method]external-codec.ty results");
            };

            let ty = {
                let ctx = ctx.as_context();
                let codec: &ErasedDynCodec = codec.rep(&ctx)?;
                codec.ty()
            };

            *result = Value::Own(ResourceOwn::new(
                ctx,
                ty,
                my_numcodecs_registry_codec_type_resource.clone(),
            )?);

            Ok(())
        },
    );
    numcodecs_registry_instance.define_func("[method]external-codec.ty", external_codec_ty)?;

    let external_codec_type_id = Func::new(
        ctx.as_context_mut(),
        FuncType::new(
            [ValueType::Borrow(
                numcodecs_registry_external_codec_type_resource.clone(),
            )],
            [ValueType::String],
        ),
        move |ctx, args, results| {
            let [Value::Borrow(ty)] = args else {
                anyhow::bail!(
                    "invalid numcodecs:abc/registry#[method]external-codec-type.codec-id arguments"
                );
            };

            let [result] = results else {
                anyhow::bail!(
                    "invalid numcodecs:abc/registry#[method]external-codectype.codec-id results"
                );
            };

            let ctx = ctx.as_context();
            let ty: &ErasedDynCodecType = ty.rep(&ctx)?;

            *result = Value::String(Arc::from(ty.codec_id()));

            Ok(())
        },
    );
    numcodecs_registry_instance
        .define_func("[method]external-codec-type.codec-id", external_codec_type_id)?;

    let external_codec_type_schema = Func::new(
        ctx.as_context_mut(),
        FuncType::new(
            [ValueType::Borrow(
                numcodecs_registry_external_codec_type_resource.clone(),
            )],
            [ValueType::String],
        ),
        move |ctx, args, results| {
            let [Value::Borrow(ty)] = args else {
                anyhow::bail!(
                    "invalid numcodecs:abc/registry#[method]external-codec-type.codec-config-schema arguments"
                );
            };

            let [result] = results else {
                anyhow::bail!(
                    "invalid numcodecs:abc/registry#[method]external-codectype.codec-config-schema results"
                );
            };

            let ctx = ctx.as_context();
            let ty: &ErasedDynCodecType = ty.rep(&ctx)?;

            *result = Value::String(Arc::from(ty.codec_config_schema().to_value().to_string()));

            Ok(())
        },
    );
    numcodecs_registry_instance.define_func(
        "[method]external-codec-type.codec-config-schema",
        external_codec_type_schema,
    )?;

    let codec_result = ResultType::new(
        Some(ValueType::Own(numcodecs_registry_external_codec_resource.clone())),
        Some(ValueType::Record(numcodecs_types_error_record.clone())),
    );

    let my_numcodecs_registry_codec_resource = numcodecs_registry_external_codec_resource.clone();
    let my_numcodecs_types_error_record = numcodecs_types_error_record.clone();
    let my_codec_result = codec_result.clone();
    let external_codec_from_config = Func::new(
        ctx.as_context_mut(),
        FuncType::new(
            [
                ValueType::Borrow(numcodecs_registry_external_codec_type_resource),
                ValueType::String,
            ],
            [ValueType::Result(my_codec_result.clone())],
        ),
        move |ctx, args, results| {
            let [Value::Borrow(ty), Value::String(config)] = args else {
                anyhow::bail!(
                    "invalid numcodecs:abc/registry#[method]external-codec-type.codec-from-config arguments"
                );
            };

            let [result] = results else {
                anyhow::bail!(
                    "invalid numcodecs:abc/registry#[method]external-codectype.codec-from-config results"
                );
            };

            let res = {
                let ctx = ctx.as_context();
                let ty: &ErasedDynCodecType = ty.rep(&ctx)?;
                ty.codec_from_config(&mut serde_json::Deserializer::from_str(config))
            };

            let res = match res {
                Ok(codec) => Ok(Some(Value::Own(ResourceOwn::new(
                    ctx,
                    codec,
                    my_numcodecs_registry_codec_resource.clone(),
                )?))),
                Err(err) => Err(Some(Value::Record(into_wit_error(
                    err,
                    &my_numcodecs_types_error_record,
                )?))),
            };

            *result = Value::Result(ResultValue::new(my_codec_result.clone(), res)?);

            Ok(())
        },
    );
    numcodecs_registry_instance.define_func(
        "[method]external-codec-type.codec-from-config",
        external_codec_from_config,
    )?;

    let my_numcodecs_registry_codec_resource = numcodecs_registry_external_codec_resource;
    let my_numcodecs_types_error_record = numcodecs_types_error_record;
    let my_codec_result = codec_result;
    let get_external_codec = Func::new(
        ctx,
        FuncType::new(
            [ValueType::String],
            [ValueType::Result(my_codec_result.clone())],
        ),
        move |ctx, args, results| {
            let [Value::String(config)] = args else {
                anyhow::bail!("invalid numcodecs:abc/registry#get-external-codec arguments");
            };

            let [result] = results else {
                anyhow::bail!("invalid numcodecs:abc/registry#get-external-codec results");
            };

            let res = match registry.get_codec(&mut serde_json::Deserializer::from_str(config)) {
                Ok(codec) => Ok(Some(Value::Own(ResourceOwn::new(
                    ctx,
                    codec,
                    my_numcodecs_registry_codec_resource.clone(),
                )?))),
                Err(err) => Err(Some(Value::Record(into_wit_error(
                    err,
                    &my_numcodecs_types_error_record,
                )?))),
            };

            *result = Value::Result(ResultValue::new(my_codec_result.clone(), res)?);

            Ok(())
        },
    );
    numcodecs_registry_instance.define_func("get-external-codec", get_external_codec)?;

    Ok(())
}

fn into_wit_error<T: Error>(err: T, ty: &RecordType) -> Result<Record, anyhow::Error> {
    let mut source: Option<&dyn Error> = err.source();

    let message = Value::String(Arc::from(format!("{err}")));
    let mut chain = if source.is_some() {
        Vec::with_capacity(4)
    } else {
        Vec::new()
    };

    while let Some(err) = source.take() {
        chain.push(Value::String(Arc::from(format!("{err}"))));
        source = err.source();
    }

    Record::new(
        ty.clone(),
        [
            ("message", message),
            (
                "chain",
                Value::List(List::new(ListType::new(ValueType::String), chain)?),
            ),
        ],
    )
}
