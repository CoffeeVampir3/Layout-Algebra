# Layout Algebra

A compile-time layout transformation library for C++23. This implements the CuTe layout algebra specification, enabling zero-cost abstractions for memory access patterns, tiling, and data format transformations.

## What it does

Layout algebra provides operations on multi-dimensional index mappings:

- **Composition**: Select elements from one layout using another as an index pattern
- **Division**: Partition a layout into tiles and remainders
- **Product**: Replicate a layout according to a pattern
- **Complement**: Find disjoint tiling offsets that cover remaining elements
- **Coalesce**: Simplify layout representations by merging contiguous modes

## Modules

- `layoutalg.cppm`: Core layout algebra operations (composition, division, product, complement)
- `layoututils.cppm`: Iterator utilities for layout offset sequences
- `tensor.cppm`: Tensor types using layouts as access patterns
- `matmul.cppm`: AMX-accelerated matrix multiplication example

## Building

Requires C++23 compiler with modules support (GCC 15+ or Clang 19+) and Ninja

```bash
chmod +x build.sh
./build.sh
./build/algebrademo
```

Or with fish:

```fish
chmod +x build.fish
./build.fish
./build/algebrademo
```

The demo showcases basic layout operations, composition, tiling, and tensor access patterns.
