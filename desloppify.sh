#!/bin/bash
nix develop -c bash -c "mkdir -p build && cd build && cmake -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .. && cd .. && desloppify \"\$@\"" bash "$@"
