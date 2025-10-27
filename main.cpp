#include <print>
#include <ranges>

import layoutalg;
import layoututils;
import tensor;

template<LayoutType L>
void print_layout_offsets(L, const char* label) {
    std::print("{}: ", label);
    size_t count = 0;
    for (auto offset : offsets(L{})) {
        if (count >= 8) break;
        std::print("{} ", offset);
        ++count;
    }
    if (L::size_v() > 8) {
        std::print("... (size: {})", L::size_v());
    }
    std::println("");
}

void basic_layouts() {
    std::println("=== Basic Layouts ===");

    using Simple = Layout<Int<16>, Int<1>>;
    print_layout_offsets(Simple{}, "16:1 (contiguous)");

    using Strided = Layout<Int<8>, Int<2>>;
    print_layout_offsets(Strided{}, "8:2 (stride-2)");

    using RowMajor = Layout<Tuple<Int<4>, Int<4>>, Tuple<Int<1>, Int<4>>>;
    print_layout_offsets(RowMajor{}, "4x4 row-major");

    using ColMajor = Layout<Tuple<Int<4>, Int<4>>, Tuple<Int<1>, Int<4>>>;
    print_layout_offsets(ColMajor{}, "4x4 col-major");
    std::println("");
}

void composition_demo() {
    std::println("=== Composition: (A ∘ B)(c) = A(B(c)) ===");

    using A = Layout<Int<16>, Int<2>>;
    using B = Layout<Int<4>, Int<1>>;
    auto composed = compose(A{}, B{});

    print_layout_offsets(A{}, "A = 16:2 (every 2nd)");
    print_layout_offsets(B{}, "B = 4:1 (first 4)");
    print_layout_offsets(composed, "A ∘ B");
    std::println("");
}

void complement_demo() {
    std::println("=== Complement: A* covers [0, M) disjoint from A ===");

    using A = Layout<Int<4>, Int<2>>;
    auto a_star = complement(A{}, Int<24>{});

    print_layout_offsets(A{}, "A = 4:2");
    print_layout_offsets(a_star, "A* = complement(A, 24)");

    auto combined = concat_layouts(A{}, a_star);
    print_layout_offsets(combined, "(A, A*) covers [0, 24)");
    std::println("");
}

void division_demo() {
    std::println("=== Division: A ⊘ B partitions A into (tile, rest) ===");

    using A = Layout<Int<64>, Int<1>>;
    using B = Layout<Int<16>, Int<1>>;
    auto div = divide(A{}, B{});

    print_layout_offsets(A{}, "A = 64:1");
    print_layout_offsets(B{}, "B = 16:1 (tile pattern)");
    std::println("A ⊘ B creates rank-2: (tiles, rests)");

    auto flat = flat_divide(A{}, B{});
    print_layout_offsets(flat, "flat_divide flattens modes");
    std::println("");
}

void product_demo() {
    std::println("=== Product: A ⊗ B replicates A by pattern B ===");

    using A = Layout<Int<4>, Int<1>>;
    using B = Layout<Int<3>, Int<1>>;
    auto prod = product(A{}, B{});

    print_layout_offsets(A{}, "A = 4:1");
    print_layout_offsets(B{}, "B = 3:1");
    std::println("A ⊗ B creates rank-2: (original, replicas)");

    auto flat = flat_product(A{}, B{});
    print_layout_offsets(flat, "flat_product");
    std::println("");
}

void tiling_2d() {
    std::println("=== 2D Tiling: Matrix -> Blocks ===");

    using Matrix = Layout<Tuple<Int<16>, Int<16>>, Tuple<Int<1>, Int<16>>>;
    using TileSize = Tuple<Int<4>, Int<4>>;

    auto tiled = logical_divide(Matrix{}, make_layout(TileSize{}));

    std::println("16x16 matrix, tiled into 4x4 blocks");
    std::println("Result: ((tile_m, rest_m), (tile_n, rest_n))");
    std::println("Creates hierarchical iteration: blocks, then within-block");
    std::println("");
}

void coalesce_demo() {
    std::println("=== Coalesce: Simplify layout representation ===");

    using L1 = Layout<Tuple<Int<4>, Int<1>>, Tuple<Int<1>, Int<4>>>;
    auto c1 = coalesce(L1{});
    std::println("(4, 1):(1, 4) -> 4:1 (remove size-1 mode)");

    using L2 = Layout<Tuple<Int<4>, Int<2>>, Tuple<Int<1>, Int<4>>>;
    auto c2 = coalesce(L2{});
    std::println("(4, 2):(1, 4) -> 8:1 (merge contiguous)");
    std::println("");
}

void tensor_demo() {
    std::println("=== Tensor: Layout as access pattern ===");

    using L = Layout<Tuple<Int<4>, Int<4>>, Tuple<Int<1>, Int<4>>>;
    auto t = make_tensor<float, L>();

    for (size_t i = 0; i < t.size(); ++i) {
        t(i) = static_cast<float>(i);
    }

    std::print("First 8 elements: ");
    for (size_t i = 0; i < 8; ++i) {
        std::print("{} ", t(i));
    }
    std::println("");

    std::println("Same data, different layout interpretation");
    std::println("");
}

int main() {
    basic_layouts();
    composition_demo();
    complement_demo();
    division_demo();
    product_demo();
    tiling_2d();
    coalesce_demo();
    tensor_demo();

    std::println("=== Summary ===");
    std::println("Layout algebra enables:");
    std::println("  - Compile-time layout transformations (zero runtime cost)");
    std::println("  - Tiling and blocking via division/product");
    std::println("  - Memory access pattern optimization");
    std::println("  - Hardware-specific format generation");

    return 0;
}
