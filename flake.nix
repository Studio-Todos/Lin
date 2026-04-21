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
          ];

          shellHook = ''
            export LLVM_DIR=${llvm.llvm.dev}
            export MLIR_DIR=${llvm.mlir.dev}
            echo "Lin Development Environment"
            echo "LLVM/MLIR Version: ${llvm.llvm.version}"
          '';
        };
      }
    );
}
