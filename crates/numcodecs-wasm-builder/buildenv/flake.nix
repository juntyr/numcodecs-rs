{
  inputs = {
    nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/*.tar.gz";
    rust-overlay.url = "https://flakehub.com/f/oxalica/rust-overlay/*.tar.gz";
  };

  outputs = { nixpkgs, rust-overlay, ... }:
    let
      allSystems = [ "x86_64-linux" "aarch64-darwin" "x86_64-darwin" ];

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
            version = "22.0";
            src = pkgs.fetchurl {
              url =
                "https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-22/wasi-sysroot-22.0.tar.gz";
              sha256 =
                "23881870d5a9c94df0529bc3e9b13682b7bbb07e5167555132fdc14e1faf1bb8";
            };

            phases = "installPhase";

            installPhase = ''
              mkdir -p $out
              tar -xf $src --strip-components=1 -C $out
            '';
          };
        in {
          default = pkgs.mkShellNoCC {
            packages = with pkgs; [
              (rust-bin.fromRustupToolchainFile ./rust-toolchain)
              llvmPackages_19.libclang
              wasi-sysroot
              cmake
              binaryen
            ];
            env = {
              MY_AR = "${pkgs.llvmPackages_19.bintools}/bin/ar";
              MY_CLANG = "${pkgs.llvmPackages_19.clang.cc}/bin";
              MY_LIBCLANG = "${pkgs.llvmPackages_19.libclang.lib}/lib";
              MY_LLD = "${pkgs.llvmPackages_19.lld}/bin";
              MY_NM = "${pkgs.llvmPackages_19.bintools}/bin/nm";
              MY_WASI_SYSROOT = "${wasi-sysroot}";
              MY_WASM_OPT = "${pkgs.binaryen}/bin/wasm-opt";
            };
          };
        });
    };
}
