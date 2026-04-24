{
  description = "Lin Programming Language - Multi-paradigm, Interaction Net-based, statically typed language";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        llvm = pkgs.llvmPackages_18;
      in
      {
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
            pkgs.vulkan-loader
            pkgs.vulkan-validation-layers
          ];

          shellHook = ''
            export LLVM_DIR=${llvm.llvm.dev}
            export MLIR_DIR=${llvm.mlir.dev}
            export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib:${pkgs.vulkan-validation-layers}/lib:$LD_LIBRARY_PATH"
            export VK_LAYER_PATH="${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d"
            echo "Lin Development Environment"
            echo "LLVM/MLIR Version: ${llvm.llvm.version}"
          '';
        };
      }
    );
}
