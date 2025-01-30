use std::sync::OnceLock;

use log::Level;
use wasm_component_layer::{
    AsContextMut, EnumType, Func, FuncType, InterfaceIdentifier, Linker, PackageIdentifier,
    PackageName, TypeIdentifier, Value, ValueType,
};

pub fn add_to_linker(linker: &mut Linker, ctx: impl AsContextMut) -> Result<(), anyhow::Error> {
    const LEVEL_CASES: [&str; 6] = ["trace", "debug", "info", "warn", "error", "critical"];
    const LOG_LEVELS: [Level; 6] = [
        Level::Trace,
        Level::Debug,
        Level::Info,
        Level::Warn,
        Level::Error,
        Level::Error,
    ];

    let WasiLoggingInterface {
        logging: wasi_logging_interface,
    } = WasiLoggingInterface::get();

    let wasi_logging_instance = linker.define_instance(wasi_logging_interface.clone())?;

    let level_ty = EnumType::new(
        Some(TypeIdentifier::new(
            "level",
            Some(wasi_logging_interface.clone()),
        )),
        LEVEL_CASES,
    )?;

    let log = Func::new(
        ctx,
        FuncType::new(
            [
                ValueType::Enum(level_ty.clone()),
                ValueType::String,
                ValueType::String,
            ],
            [],
        ),
        move |_ctx, args, _results| {
            let [Value::Enum(level), Value::String(context), Value::String(message)] = args else {
                anyhow::bail!("invalid wasi:logging/logging#log arguments");
            };

            anyhow::ensure!(
                level.ty() == level_ty,
                "invalid wasi:logging/logging#log level type"
            );

            let Some(level) = LOG_LEVELS.get(level.discriminant()) else {
                anyhow::bail!("invalid wasi:logging/logging#log level kind");
            };

            log!(target: context, *level, "{message}");

            Ok(())
        },
    );

    wasi_logging_instance.define_func("log", log)?;

    Ok(())
}

#[non_exhaustive]
pub struct WasiLoggingInterface {
    pub logging: InterfaceIdentifier,
}

impl WasiLoggingInterface {
    #[must_use]
    pub fn get() -> &'static Self {
        static WASI_LOGGING_INTERFACE: OnceLock<WasiLoggingInterface> = OnceLock::new();

        WASI_LOGGING_INTERFACE.get_or_init(|| Self {
            logging: InterfaceIdentifier::new(
                PackageIdentifier::new(PackageName::new("wasi", "logging"), None),
                "logging",
            ),
        })
    }
}
