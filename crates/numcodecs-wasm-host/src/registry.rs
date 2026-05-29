use std::{error::Error, sync::Arc};

use numcodecs::{ErasedDynCodec, ErasedDynCodecType};
use numcodecs_registry::Registry;
use wasm_component_layer::{
    AsContext, AsContextMut, Func, FuncType, Linker, List, ListType, Record, RecordType,
    ResourceOwn, ResourceType, ResultType, ResultValue, TypeIdentifier, Value, ValueType,
};

use crate::wit::NumcodecsWitInterfaces;

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
        types: numcodecs_types_interface,
        ..
    } = NumcodecsWitInterfaces::get();

    let registry = Arc::new(registry);

    let numcodecs_types_error_record = RecordType::new(
        Some(TypeIdentifier::new(
            "error",
            Some(numcodecs_types_interface.clone()),
        )),
        [
            ("message", ValueType::String),
            ("chain", ValueType::List(ListType::new(ValueType::String))),
        ],
    )?;

    let numcodecs_registry_instance =
        linker.define_instance(numcodecs_registry_interface.clone())?;

    let numcodecs_registry_codec_resource = ResourceType::with_destructor(
        ctx.as_context_mut(),
        Some(TypeIdentifier::new(
            "erased-dyn-codec",
            Some(numcodecs_registry_interface.clone()),
        )),
        |_ctx, codec: ErasedDynCodec| {
            std::mem::drop(codec);
            Ok(())
        },
    )?;

    numcodecs_registry_instance.define_resource(
        "erased-dyn-codec",
        numcodecs_registry_codec_resource.clone(),
    )?;

    let numcodecs_registry_codec_type_resource = ResourceType::with_destructor(
        ctx.as_context_mut(),
        Some(TypeIdentifier::new(
            "erased-dyn-codec-type",
            Some(numcodecs_registry_interface.clone()),
        )),
        |_ctx, codec_ty: ErasedDynCodecType| {
            std::mem::drop(codec_ty);
            Ok(())
        },
    )?;

    numcodecs_registry_instance.define_resource(
        "erased-dyn-codec-type",
        numcodecs_registry_codec_type_resource,
    )?;

    let my_numcodecs_registry_codec_resource = numcodecs_registry_codec_resource.clone();
    let codec_clone = Func::new(
        ctx.as_context_mut(),
        FuncType::new(
            [ValueType::Borrow(numcodecs_registry_codec_resource.clone())],
            [ValueType::Own(numcodecs_registry_codec_resource.clone())],
        ),
        move |ctx, args, results| {
            let [Value::Borrow(codec)] = args else {
                anyhow::bail!(
                    "invalid numcodecs:abc/registry#[method]erased-dyn-codec.clone arguments"
                );
            };

            let [result] = results else {
                anyhow::bail!(
                    "invalid numcodecs:abc/registry#[method]erased-dyn-codec.clone results"
                );
            };

            let codec = {
                let codec_ctx = ctx.as_context();
                let codec: &ErasedDynCodec = codec.rep(&codec_ctx)?;
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
    numcodecs_registry_instance.define_func("[method]erased-dyn-codec.clone", codec_clone)?;

    let get_codec_result = ResultType::new(
        Some(ValueType::Own(numcodecs_registry_codec_resource.clone())),
        Some(ValueType::Record(numcodecs_types_error_record.clone())),
    );

    let my_numcodecs_registry_codec_resource = numcodecs_registry_codec_resource;
    let get_codec = Func::new(
        ctx,
        FuncType::new(
            [ValueType::String],
            [ValueType::Result(get_codec_result.clone())],
        ),
        move |ctx, args, results| {
            let [Value::String(config)] = args else {
                anyhow::bail!("invalid numcodecs:abc/registry#get-codec arguments");
            };

            let [result] = results else {
                anyhow::bail!("invalid numcodecs:abc/registry#get-codec results");
            };

            let res = match registry.get_codec(&mut serde_json::Deserializer::from_str(config)) {
                Ok(codec) => Ok(Some(Value::Own(ResourceOwn::new(
                    ctx,
                    codec,
                    my_numcodecs_registry_codec_resource.clone(),
                )?))),
                Err(err) => Err(Some(Value::Record(into_wit_error(
                    err,
                    &numcodecs_types_error_record,
                )?))),
            };

            *result = Value::Result(ResultValue::new(get_codec_result.clone(), res)?);

            Ok(())
        },
    );
    numcodecs_registry_instance.define_func("get-codec", get_codec)?;

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
