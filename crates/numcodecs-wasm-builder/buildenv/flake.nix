{
  inputs = {
    nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/*.tar.gz";
    rust-overlay.url = "https://flakehub.com/f/oxalica/rust-overlay/*.tar.gz";
  };

  outputs = { nixpkgs, rust-overlay, ... }:
    let
      allSystems = [ "x86_64-linux" "aarch64-darwin" "x86_64-darwin" ];
      # keep in sync with rust-toolchain and wasi-sysroot
      llvmVersion = "21";

      forEachSystem = f:
        nixpkgs.lib.genAttrs allSystems (system:
          f {
            inherit system;
            pkgs = import nixpkgs {
              inherit system;
              overlays = [ rust-overlay.overlays.default ];
            };
          });
    in {
      devShells = forEachSystem ({ pkgs, system, }:
        let
          wasi-sysroot = pkgs.stdenv.mkDerivation {
            pname = "wasi-sysroot";
            version = "30.0";
            src = pkgs.fetchurl {
              url =
                "https://github.com/alexcrichton/wasi-sdk/releases/download/wasi-sdk-30.0-cpp-exn/wasi-sysroot-30.1g-exns+m.tar.gz";
              sha256 =
                "8eae8433e403c4d7062348ba615f770089b468a4c21f117e003e7d07e0d2a27e";
            };

            phases = "installPhase";

            installPhase = ''
              mkdir -p $out
              tar -xf $src --strip-components=1 -C $out
            '';
          };
          libclang_rt = pkgs.stdenv.mkDerivation {
            pname = "libclang_rt";
            version = "30.0";
            src = pkgs.fetchurl {
              url =
                "https://github.com/alexcrichton/wasi-sdk/releases/download/wasi-sdk-30.0-cpp-exn/libclang_rt-30.1g-exns+m.tar.gz";
              sha256 =
                "33a908cbd169e629a314a3ff5a03bb259ca2e1b4e7627a8aa73e2080e4dbcfaa";
            };

            phases = "installPhase";

            installPhase = ''
              mkdir -p $out
              tar -xf $src --strip-components=1 -C $out
            '';
          };
        in {
          default = pkgs.mkShellNoCC {
            packages = [
              (pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain)
              pkgs."llvmPackages_${llvmVersion}".libclang
              wasi-sysroot
              libclang_rt
              pkgs.cmake
              pkgs.binaryen
            ];
            env = {
              MY_LLVM_VERSION = "${llvmVersion}";
              MY_AR = "${pkgs."llvmPackages_${llvmVersion}".bintools}/bin/ar";
              MY_CLANG = "${pkgs."llvmPackages_${llvmVersion}".clang.cc}/bin";
              MY_LIBCLANG = "${pkgs."llvmPackages_${llvmVersion}".libclang.lib}/lib";
              MY_LLD = "${pkgs."llvmPackages_${llvmVersion}".lld}/bin";
              MY_NM = "${pkgs."llvmPackages_${llvmVersion}".bintools}/bin/nm";
              MY_RANLIB = "${pkgs."llvmPackages_${llvmVersion}".bintools}/bin/ranlib";
              MY_STRIP = "${pkgs."llvmPackages_${llvmVersion}".bintools}/bin/strip";
              MY_OBJDUMP = "${pkgs."llvmPackages_${llvmVersion}".bintools}/bin/objdump";
              MY_DLLTOOL = "${pkgs."llvmPackages_${llvmVersion}".bintools}/bin/dlltool";
              MY_WASI_SYSROOT = "${wasi-sysroot}";
              MY_LIBCLANG_RT = "${libclang_rt}";
              MY_WASM_OPT = "${pkgs.binaryen}/bin/wasm-opt";
            };
          };
        });
    };
}
