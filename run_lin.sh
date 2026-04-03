#!/bin/bash
nix develop -c bash -c "mkdir -p build && cd build && cmake -GNinja .. && ninja check-linc"
