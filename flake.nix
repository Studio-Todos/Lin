{
  description = "Lin Programming Language - Multi-paradigm, Interaction Net-based, statically typed language";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    let
      overlay = final: prev:
        let
          llvm = final.llvmPackages_18;
          vulkan-loader = final.vulkan-loader;
          vulkan-validation-layers = final.vulkan-validation-layers;
        in
        {
          lin = final.stdenv.mkDerivation {
            pname = "lin";
            version = "0.1.0";

            src = ./.;

            nativeBuildInputs = [
              llvm.llvm
              llvm.bintools
              llvm.clang
              llvm.mlir
              llvm.mlir.dev
              final.cmake
              final.ninja
              final.python3
              final.lit
              final.gtest
              final.spirv-tools
              final.vulkan-headers
              vulkan-loader
              vulkan-validation-layers
            ];

            configurePhase = ''
              export LLVM_DIR=${llvm.llvm.dev}
              export MLIR_DIR=${llvm.mlir.dev}
              export LD_LIBRARY_PATH="${vulkan-loader}/lib:${vulkan-validation-layers}/lib:$LD_LIBRARY_PATH"
              export VK_LAYER_PATH="${vulkan-validation-layers}/share/vulkan/explicit_layer.d"

              rm -rf build
              cmake -G Ninja -B build \
                -DLLVM_DIR=$out/lib/cmake/llvm \
                -DMLIR_DIR=$out/lib/cmake/mlir \
                -DCMAKE_BUILD_TYPE=Release \
                .
            '';

            buildPhase = ''
              ninja -C build
            '';

            installPhase = ''
              mkdir -p $out/bin
              cp build/src/linc $out/bin/
            '';
          };
        };
      overlayTests = final: prev:
        let
          llvm = final.llvmPackages_18;
          vulkan-loader = final.vulkan-loader;
          vulkan-validation-layers = final.vulkan-validation-layers;
        in
        {
          lin-tests = final.stdenv.mkDerivation {
            pname = "lin-tests";
            version = "0.1.0";

            src = ./.;
            allowUnpack = true;

            outputs = [ "out" ];
            doCheck = true;

            nativeBuildInputs = [
              llvm.llvm
              llvm.bintools
              llvm.clang
              llvm.mlir
              llvm.mlir.dev
              final.cmake
              final.ninja
              final.python3
              final.lit
              final.gtest
              final.spirv-tools
              final.vulkan-headers
              vulkan-loader
              vulkan-validation-layers
            ];

            configurePhase = ''
              export LLVM_DIR=${llvm.llvm.dev}
              export MLIR_DIR=${llvm.mlir.dev}
              export LD_LIBRARY_PATH="${vulkan-loader}/lib:${vulkan-validation-layers}/lib:$LD_LIBRARY_PATH"
              export VK_LAYER_PATH="${vulkan-validation-layers}/share/vulkan/explicit_layer.d"

              cmake -G Ninja -B build \
                -DLLVM_DIR=$out/lib/cmake/llvm \
                -DMLIR_DIR=$out/lib/cmake/mlir \
                -DCMAKE_BUILD_TYPE=Release \
                .
            '';

            buildPhase = ''
              ninja -C build
            '';

            checkPhase = ''
              ninja -C build check-linc
            '';

            installPhase = ''
              mkdir -p $out
              touch $out
            '';
          };
        };
    in
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; overlays = [ overlay overlayTests ]; };
        llvm = pkgs.llvmPackages_18;
        vulkan-loader = pkgs.vulkan-loader;
        vulkan-validation-layers = pkgs.vulkan-validation-layers;
      in
      {
        packages.default = pkgs.lin;
        packages.lin-tests = pkgs.lin-tests;
        packages.lin = pkgs.lin;

        checks.lin-tests = pkgs.lin-tests;

        devShells.default = pkgs.mkShell {
          buildInputs = [
            llvm.llvm
            llvm.bintools
            llvm.clang
            llvm.mlir
            llvm.mlir.dev
            pkgs.cmake
            pkgs.ninja
            pkgs.python3
            pkgs.lit
            pkgs.gtest
            pkgs.spirv-tools
            pkgs.vulkan-headers
            vulkan-loader
            vulkan-validation-layers
            pkgs.gcc
          ];

          shellHook = ''
            export LLVM_DIR=${llvm.llvm.dev}
            export MLIR_DIR=${llvm.mlir.dev}
            export LD_LIBRARY_PATH="${vulkan-loader}/lib:${vulkan-validation-layers}/lib:$LD_LIBRARY_PATH"
            export VK_LAYER_PATH="${vulkan-validation-layers}/share/vulkan/explicit_layer.d"
            export PATH="${llvm.llvm}/bin:$PATH"
            echo "Lin Development Environment"
            echo "LLVM/MLIR Version: ${llvm.llvm.version}"
          '';
        };
      }
    );
}