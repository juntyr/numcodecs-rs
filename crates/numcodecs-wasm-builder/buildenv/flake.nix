{
  inputs = {
    nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/*.tar.gz";
    rust-overlay.url = "https://flakehub.com/f/oxalica/rust-overlay/*.tar.gz";
  };

  outputs = { nixpkgs, rust-overlay, ... }:
    let
      allSystems = [ "x86_64-linux" "aarch64-darwin" "x86_64-darwin" ];
      # keep in sync with rust-toolchain and wasi-sysroot
      llvmVersion = "20";

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
            version = "27.0";
            src = pkgs.fetchurl {
              url =
                "https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-27/wasi-sysroot-27.0.tar.gz";
              sha256 =
                "7110ac48f5d0b1f6ab67d57aecf52450540dddd790cafdc45f0fdfb429bdab84";
            };

            phases = "installPhase";

            installPhase = ''
              mkdir -p $out
              tar -xf $src --strip-components=1 -C $out
            '';
          };
          libclang_rt = pkgs.stdenv.mkDerivation {
            pname = "libclang_rt";
            version = "27.0";
            src = pkgs.fetchurl {
              url =
                "https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-27/libclang_rt-27.0.tar.gz";
              sha256 =
                "9e0f382110a3cf9196f02432c8f2e54d151515de36f9311c8c16073f6e6b16d3";
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
