use std::{fmt, io::Write, sync::OnceLock};

use wasm_component_layer::{
    AsContextMut, Func, FuncType, InterfaceIdentifier, Linker, ListType, PackageIdentifier,
    PackageName, Value, ValueType,
};

pub fn add_to_linker(linker: &mut Linker, mut ctx: impl AsContextMut) -> Result<(), anyhow::Error> {
    let WasiSandboxedStdioInterface {
        stdio: simple_stdio_interface,
    } = WasiSandboxedStdioInterface::get();

    let simple_stdio_instance = linker.define_instance(simple_stdio_interface.clone())?;

    simple_stdio_instance.define_func(
        "write-stdout",
        OutputStream::Stdout.create_write_func(ctx.as_context_mut()),
    )?;
    simple_stdio_instance.define_func(
        "flush-stdout",
        OutputStream::Stdout.create_flush_func(ctx.as_context_mut()),
    )?;
    simple_stdio_instance.define_func(
        "write-stderr",
        OutputStream::Stderr.create_write_func(ctx.as_context_mut()),
    )?;
    simple_stdio_instance.define_func(
        "flush-stderr",
        OutputStream::Stderr.create_flush_func(ctx.as_context_mut()),
    )?;

    Ok(())
}

#[derive(Copy, Clone)]
enum OutputStream {
    Stdout,
    Stderr,
}

impl OutputStream {
    fn create_write_func(self, ctx: impl AsContextMut) -> Func {
        Func::new(
            ctx,
            FuncType::new([ValueType::List(ListType::new(ValueType::U8))], []),
            move |_ctx, args, results| {
                let [Value::List(contents)] = args else {
                    anyhow::bail!("invalid wasi-sandboxed:io/stdio#write-{self} arguments");
                };
                let Ok(contents) = contents.typed::<u8>() else {
                    anyhow::bail!("invalid wasi-sandboxed:io/stdio#write-{self} argument type");
                };

                anyhow::ensure!(
                    results.is_empty(),
                    "invalid wasi-sandboxed:io/stdio#write-{self} results"
                );

                if let Err(err) = match self {
                    Self::Stdout => std::io::stdout().write_all(contents),
                    Self::Stderr => std::io::stderr().write_all(contents),
                } {
                    error!(
                        "Failed to write {} byte{} to {self}: {err}",
                        contents.len(),
                        if contents.len() == 1 { "" } else { "s" }
                    );
                }

                Ok(())
            },
        )
    }

    fn create_flush_func(self, ctx: impl AsContextMut) -> Func {
        Func::new(ctx, FuncType::new([], []), move |_ctx, args, results| {
            anyhow::ensure!(
                args.is_empty(),
                "invalid wasi-sandboxed:io/stdio#flush-{self} arguments"
            );

            anyhow::ensure!(
                results.is_empty(),
                "invalid wasi-sandboxed:io/stdio#flush-{self} results"
            );

            if let Err(err) = match self {
                Self::Stdout => std::io::stdout().flush(),
                Self::Stderr => std::io::stderr().flush(),
            } {
                error!("Failed to flush {self}: {err}");
            }

            Ok(())
        })
    }
}

impl fmt::Display for OutputStream {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str(match self {
            Self::Stdout => "stdout",
            Self::Stderr => "stderr",
        })
    }
}

#[non_exhaustive]
pub struct WasiSandboxedStdioInterface {
    pub stdio: InterfaceIdentifier,
}

impl WasiSandboxedStdioInterface {
    #[must_use]
    pub fn get() -> &'static Self {
        static WASI_SANDBOXED_STDIO_INTERFACE: OnceLock<WasiSandboxedStdioInterface> =
            OnceLock::new();

        WASI_SANDBOXED_STDIO_INTERFACE.get_or_init(|| Self {
            stdio: InterfaceIdentifier::new(
                PackageIdentifier::new(
                    PackageName::new("wasi-sandboxed", "io"),
                    Some(semver::Version::new(0, 2, 3)),
                ),
                "stdio",
            ),
        })
    }
}
