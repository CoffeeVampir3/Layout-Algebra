#!/usr/bin/env fish

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
