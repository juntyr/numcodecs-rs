#[cfg(feature = "bindgen")]
use std::env;
#[cfg(feature = "bindgen")]
use std::path::PathBuf;

fn main() {
    let src_dir = "vendor/src";
    
    // Build the static library using CMake from src/ directory
    let dst = cmake::Config::new(src_dir)
        .build();
    
    // Tell cargo to look for libraries in the CMake build directory
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    
    // Link against the static EBCC library and its dependencies
    println!("cargo:rustc-link-lib=static=ebcc");
    println!("cargo:rustc-link-lib=static=openjp2");
    println!("cargo:rustc-link-lib=static=zstd");
    
    // Try explicitly adding the static libraries as link args for tests
    println!("cargo:rustc-link-arg=-lebcc");
    println!("cargo:rustc-link-arg=-lopenjp2");
    println!("cargo:rustc-link-arg=-lzstd");
    
    // Link against required system libraries
    println!("cargo:rustc-link-lib=dylib=m");
    
    // Tell cargo to invalidate the built crate whenever these files change
    println!("cargo:rerun-if-changed={}/ebcc_codec.h", src_dir);
    println!("cargo:rerun-if-changed={}/ebcc_codec.c", src_dir);
    println!("cargo:rerun-if-changed={}/log/log.h", src_dir);
    println!("cargo:rerun-if-changed={}/log/log.c", src_dir);
    println!("cargo:rerun-if-changed={}/spiht/spiht_re.c", src_dir);
    println!("cargo:rerun-if-changed={}/spiht/spiht_re.h", src_dir);
    println!("cargo:rerun-if-changed={}/CMakeLists.txt", src_dir);
    println!("cargo:rerun-if-changed=build.rs");

    // Generate bindings only if the bindgen feature is enabled
    #[cfg(feature = "bindgen")]
    {
        // Generate bindings for the EBCC header
        let bindings = bindgen::Builder::default()
            .header(&format!("{}/ebcc_codec.h", src_dir))
            .clang_arg(&format!("-I{}/", src_dir))
            .clang_arg(&format!("-I{}/log/", src_dir))
            .clang_arg(&format!("-I{}/spiht/", src_dir))
            // Tell bindgen to generate bindings for these types and functions
            .allowlist_type("codec_config_t")
            .allowlist_type("residual_t")
            .allowlist_function("encode_climate_variable")
            .allowlist_function("decode_climate_variable")
            .allowlist_function("free_buffer")
            .allowlist_function("print_config")
            .allowlist_function("log_set_level_from_env")
            // Generate constants
            .allowlist_var("NDIMS")
            // Use constified enum module for better enum handling
            .constified_enum_module("residual_t")
            // Generate comments from C headers
            .generate_comments(true)
            // Use core instead of std for no_std compatibility
            .use_core()
            // Generate layout tests
            .layout_tests(true)
            // Don't generate recursively for system headers
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate bindings");

        // Write the bindings to the $OUT_DIR/bindings.rs file
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("bindings.rs"))
            .expect("Couldn't write bindings!");
    }
}