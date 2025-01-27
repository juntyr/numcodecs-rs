#![allow(missing_docs)] // FIXME

use std::{
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

use clap::Parser;
use semver::Version;

// ensure the crate is rebuilt if any of these files change
const _: &'static str = include_str!("../flake.lock");
const _: &'static str = include_str!("../flake.nix");
const _: &'static str = include_str!("../include.hpp");

#[derive(Parser, Debug)]
#[command()]
struct Args {
    /// Name of the numcodecs codec crate to compile
    #[arg(name = "crate", long)]
    crate_: String,

    /// Version of the numcodecs codec crate to compile
    #[arg(long)]
    version: Version,

    /// Path to the codec type to export, without the leading crate name
    #[arg(long)]
    codec: String,

    /// Path to which the wasm file is output
    #[arg(long, short)]
    out: PathBuf,
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    let scratch_dir = scratch::path(concat!(
        env!("CARGO_PKG_NAME"),
        "-",
        env!("CARGO_PKG_VERSION"),
    ));
    let target_dir = scratch_dir.join("target");
    let crate_dir = scratch_dir.join(format!("{}-wasm-{}", args.crate_, args.version));
    let flake_path = get_nix_flake_path()?;
    let cpp_include_path = write_or_get_cpp_include_path(&scratch_dir)?;

    let nix_env = NixEnv::new(&flake_path)?;

    eprintln!("scratch_dir={scratch_dir:?}");
    eprintln!("target_dir={target_dir:?}");
    eprintln!("flake_path={flake_path:?}");
    eprintln!("cpp_include_path={cpp_include_path:?}");

    eprintln!("creating {target_dir:?}");
    fs::create_dir_all(&target_dir)?;

    eprintln!("creating {crate_dir:?}");
    if crate_dir.exists() {
        fs::remove_dir_all(&crate_dir)?;
    }
    fs::create_dir_all(&crate_dir)?;

    fs::write(crate_dir.join("Cargo.toml"), format!(r#"
[package]
name = "{crate_}-wasm"
version = "{version}"
edition = "2021"

# See more keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html

[dependencies]
numcodecs-wasm-logging = {{ version = "0.1", default-features = false }}
numcodecs-wasm-guest = {{ version = "0.2", default-features = false }}
numcodecs-my-codec = {{ package = "{crate_}", version = {version}, default-features = false }}
    "#, crate_ = args.crate_, version = args.version))?;

    fs::create_dir_all(crate_dir.join("src"))?;

    fs::write(crate_dir.join("src").join("lib.rs"), format!(r#"
#![cfg_attr(not(test), no_main)]

numcodecs_wasm_guest::export_codec!(
    codecs_wasm_logging::LoggingCodec<numcodecs_my_codec::{codec}>
);
    "#, codec = args.codec))?;

    let wasm = build_wasm_codec(
        &flake_path,
        &nix_env,
        &cpp_include_path,
        &target_dir,
        &crate_dir,
        &format!("{}-wasm", args.crate_),
    )?;
    let wasm = optimize_wasm_codec(&wasm, &nix_env)?;
    let wasm = adapt_wasi_snapshot_to_preview2(&wasm)?;

    fs::copy(wasm, args.out)?;

    Ok(())
}

fn write_or_get_cpp_include_path(_scratch_dir: &Path) -> io::Result<PathBuf> {
    Path::new("include.hpp").canonicalize()
}

fn get_nix_flake_path() -> io::Result<PathBuf> {
    Path::new(".").canonicalize()
}

struct NixEnv {
    ar: PathBuf,
    clang: PathBuf,
    libclang: PathBuf,
    lld: PathBuf,
    nm: PathBuf,
    wasi_sysroot: PathBuf,
    wasm_opt: PathBuf,
}

impl NixEnv {
    pub fn new(flake_path: &Path) -> io::Result<Self> {
        fn try_read_env(env: &HashMap<&str, &str>, key: &str) -> Result<PathBuf, io::Error> {
            env.get(key).copied().map(PathBuf::from).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("missing flake env key: {key}"),
                )
            })
        }

        let mut env = Command::new("nix");
        env.arg("develop");
        // env.arg("--store");
        // env.arg(nix_store_path);
        env.arg("--no-update-lock-file");
        env.arg("--ignore-environment");
        env.arg(flake_path);
        env.arg("--command");
        env.arg("env");
        eprintln!("executing {env:?}");
        let env = env.output()?;
        eprintln!(
            "{}\n{}",
            String::from_utf8_lossy(&env.stdout),
            String::from_utf8_lossy(&env.stderr)
        );
        let env = std::str::from_utf8(&env.stdout).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid flake env output: {err}"),
            )
        })?;
        let env = env
            .lines()
            .filter_map(|line| line.split_once('='))
            .collect::<HashMap<_, _>>();

        Ok(Self {
            ar: try_read_env(&env, "MY_AR")?,
            clang: try_read_env(&env, "MY_CLANG")?,
            libclang: try_read_env(&env, "MY_LIBCLANG")?,
            lld: try_read_env(&env, "MY_LLD")?,
            nm: try_read_env(&env, "MY_NM")?,
            wasi_sysroot: try_read_env(&env, "MY_WASI_SYSROOT")?,
            wasm_opt: try_read_env(&env, "MY_WASM_OPT")?,
        })
    }
}

#[expect(clippy::too_many_lines)]
fn configure_cargo_cmd(
    flake_path: &Path,
    nix_env: &NixEnv,
    cpp_include_path: &Path,
    target_dir: &Path,
) -> Command {
    let NixEnv {
        ar,
        clang,
        libclang,
        lld,
        nm,
        wasi_sysroot,
        ..
    } = nix_env;

    let mut cmd = Command::new("nix");
    cmd.env_remove("CARGO_ENCODED_RUSTFLAGS");
    cmd.arg("develop");
    // cmd.arg("--store");
    // cmd.arg(nix_store_path);
    cmd.arg("--no-update-lock-file");
    cmd.arg("--ignore-environment");
    cmd.arg(flake_path);
    cmd.arg("--command");
    cmd.arg("env");
    cmd.arg(format!("CC={clang}", clang = clang.join("clang").display()));
    cmd.arg(format!(
        "CXX={clang}",
        clang = clang.join("clang++").display()
    ));
    cmd.arg(format!("LD={lld}", lld = lld.join("lld").display()));
    cmd.arg(format!("LLD={lld}", lld = lld.join("lld").display()));
    cmd.arg(format!("AR={ar}", ar = ar.display()));
    cmd.arg(format!("NM={nm}", nm = nm.display()));
    cmd.arg(format!(
        "LIBCLANG_PATH={libclang}",
        libclang = libclang.display()
    ));
    cmd.arg(format!(
        "CFLAGS=--target=wasm32-wasip1 -nodefaultlibs -resource-dir {resource_dir} \
         --sysroot={wasi_sysroot} -isystem {clang_include} -isystem {wasi32_wasi_include} \
         -isystem {include} -B {lld} -D_WASI_EMULATED_PROCESS_CLOCKS -O3 --target=wasm32-wasip1",
        resource_dir = libclang.join("clang").join("18").display(),
        wasi_sysroot = wasi_sysroot.display(),
        clang_include = libclang.join("clang").join("18").join("include").display(),
        wasi32_wasi_include = wasi_sysroot.join("include").join("wasm32-wasip1").display(),
        include = wasi_sysroot.join("include").display(),
        lld = lld.display(),
    ));
    cmd.arg(format!(
        "CXXFLAGS=--target=wasm32-wasip1 -nodefaultlibs -resource-dir {resource_dir} \
         --sysroot={wasi_sysroot} -isystem {wasm32_wasi_cxx_include} -isystem {cxx_include} \
         -isystem {clang_include} -isystem {wasi32_wasi_include} -isystem {include} -B {lld} \
         -D_WASI_EMULATED_PROCESS_CLOCKS -include {cpp_include_path} -O3 --target=wasm32-wasip1",
        resource_dir = libclang.join("clang").join("18").display(),
        wasi_sysroot = wasi_sysroot.display(),
        wasm32_wasi_cxx_include = wasi_sysroot
            .join("include")
            .join("wasm32-wasip1")
            .join("c++")
            .join("v1")
            .display(),
        cxx_include = wasi_sysroot
            .join("include")
            .join("c++")
            .join("v1")
            .display(),
        clang_include = libclang.join("clang").join("18").join("include").display(),
        wasi32_wasi_include = wasi_sysroot.join("include").join("wasm32-wasip1").display(),
        include = wasi_sysroot.join("include").display(),
        lld = lld.display(),
        cpp_include_path = cpp_include_path.display(),
    ));
    cmd.arg(format!(
        "BINDGEN_EXTRA_CLANG_ARGS=--target=wasm32-wasip1 -nodefaultlibs -resource-dir \
         {resource_dir} --sysroot={wasi_sysroot} -isystem {wasm32_wasi_cxx_include} -isystem \
         {cxx_include} -isystem {clang_include} -isystem {wasi32_wasi_include} -isystem {include} \
         -B {lld} -D_WASI_EMULATED_PROCESS_CLOCKS -fvisibility=default",
        resource_dir = libclang.join("clang").join("18").display(),
        wasi_sysroot = wasi_sysroot.display(),
        wasm32_wasi_cxx_include = wasi_sysroot
            .join("include")
            .join("wasm32-wasip1")
            .join("c++")
            .join("v1")
            .display(),
        cxx_include = wasi_sysroot
            .join("include")
            .join("c++")
            .join("v1")
            .display(),
        clang_include = libclang.join("clang").join("18").join("include").display(),
        wasi32_wasi_include = wasi_sysroot.join("include").join("wasm32-wasip1").display(),
        include = wasi_sysroot.join("include").display(),
        lld = lld.display(),
    ));
    cmd.arg("CXXSTDLIB=c++");
    // disable default flags from cc
    cmd.arg("CRATE_CC_NO_DEFAULTS=1");
    cmd.arg("LDFLAGS=-lc -lwasi-emulated-process-clocks");
    cmd.arg(format!(
        "RUSTFLAGS=-C panic=abort -C strip=symbols -C link-arg=-L{wasm32_wasi_lib}",
        wasm32_wasi_lib = wasi_sysroot.join("lib").join("wasm32-wasip1").display(),
    ));
    cmd.arg(format!(
        "CARGO_TARGET_DIR={target_dir}",
        target_dir = target_dir.display()
    ));

    // we don't need nightly Rust features but need to compile std with immediate
    // panic abort instead of compiling with nightly, we fake it and forbid the
    // unstable_features lint
    cmd.arg("RUSTC_BOOTSTRAP=1");

    cmd.arg("cargo");

    cmd
}

fn build_wasm_codec(
    flake_path: &Path,
    nix_env: &NixEnv,
    cpp_include_path: &Path,
    target_dir: &Path,
    crate_dir: &Path,
    crate_name: &str,
) -> io::Result<PathBuf> {
    let mut cmd = configure_cargo_cmd(flake_path, nix_env, cpp_include_path, target_dir);
    cmd.arg("rustc")
        .arg("--crate-type=cdylib")
        .arg("-Z")
        .arg("build-std=std,panic_abort")
        .arg("-Z")
        .arg("build-std-features=panic_immediate_abort")
        .arg("--release")
        .arg("--target=wasm32-wasip1")
        .current_dir(crate_dir);

    eprintln!("executing {cmd:?}");

    let status = cmd.status()?;
    if !status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("cargo exited with code {status}"),
        ));
    }

    Ok(target_dir
        .join("wasm32-wasip1")
        .join("release")
        .join(crate_name.replace('-', "_"))
        .with_extension("wasm"))
}

fn optimize_wasm_codec(wasm: &Path, nix_env: &NixEnv) -> io::Result<PathBuf> {
    let NixEnv { wasm_opt, .. } = nix_env;

    let opt_out = wasm.with_extension("opt.wasm");

    let mut cmd = Command::new(wasm_opt);

    cmd.arg("--enable-sign-ext")
        .arg("--disable-threads")
        .arg("--enable-mutable-globals")
        .arg("--enable-nontrapping-float-to-int")
        .arg("--enable-simd")
        .arg("--enable-bulk-memory")
        .arg("--disable-exception-handling")
        .arg("--disable-tail-call")
        .arg("--disable-reference-types")
        .arg("--enable-multivalue")
        .arg("--disable-gc")
        .arg("--disable-memory64")
        .arg("--disable-relaxed-simd")
        .arg("--disable-extended-const")
        .arg("--disable-strings")
        .arg("--disable-multimemory");

    cmd.arg("-O4").arg("-o").arg(&opt_out).arg(wasm);

    eprintln!("executing {cmd:?}");

    let status = cmd.status()?;
    if !status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("wasm-opt exited with code {status}"),
        ));
    }

    Ok(opt_out)
}

fn adapt_wasi_snapshot_to_preview2(wasm: &Path) -> io::Result<PathBuf> {
    let wasm_preview2 = wasm.with_extension("preview2.wasm");

    eprintln!("reading from {wasm:?}");
    let wasm = fs::read(wasm)?;

    let mut encoder = wit_component::ComponentEncoder::default()
        .module(&wasm)
        .map_err(|err| {
            io::Error::new(
                io::ErrorKind::Other,
                // FIXME: better error reporting in the build script
                format!("wit_component::ComponentEncoder::module failed: {err:#}"),
            )
        })?
        .adapter(
            wasi_preview1_component_adapter_provider::WASI_SNAPSHOT_PREVIEW1_ADAPTER_NAME,
            wasi_preview1_component_adapter_provider::WASI_SNAPSHOT_PREVIEW1_REACTOR_ADAPTER,
        )
        .map_err(|err| {
            io::Error::new(
                io::ErrorKind::Other,
                // FIXME: better error reporting in the build script
                format!("wit_component::ComponentEncoder::adapter failed: {err:#}"),
            )
        })?;

    let wasm = encoder.encode().map_err(|err| {
        io::Error::new(
            io::ErrorKind::Other,
            // FIXME: better error reporting in the build script
            format!("wit_component::ComponentEncoder::encode failed: {err:#}"),
        )
    })?;

    eprintln!("writing to {wasm_preview2:?}");
    fs::write(&wasm_preview2, wasm)?;

    Ok(wasm_preview2)
}
