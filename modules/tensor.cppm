module;
#include <cstddef>
#include <array>
export module tensor;
import layoutalg;

// Tensor: Layout as access pattern
//
// The buffer is a flat array. The Layout describes how to MAP logical coordinates
// to physical buffer positions. The same buffer can be interpreted through different
// layouts without moving data.
//
// Example: A 4x4 row-major tensor can be reinterpreted as column-major by changing
// the layout - no data movement occurs, only the access pattern changes.
//
// Use this when:
// - Performing layout transformations without copying
// - Reinterpreting data through different coordinate systems
// - Working with strided or non-contiguous access patterns
export template<typename T, LayoutType L>
struct Tensor {
    std::array<T, L::cosize()> data;

    static constexpr auto layout = L{};

    constexpr size_t size() const { return L::size_v(); }

    constexpr T& operator()(size_t i) {
        return data[layout(i)];
    }
    constexpr const T& operator()(size_t i) const {
        return data[layout(i)];
    }

    T* ptr() { return data.data(); }
    const T* ptr() const { return data.data(); }
};

// FormattedTensor: Layout as physical format
//
// The buffer is PHYSICALLY ARRANGED in the layout's format. Access is always linear
// (stride-1). The Layout describes WHAT FORMAT the data represents, not HOW TO ACCESS it.
//
// This is essential for hardware instructions (e.g., AMX VNNI) that require data in
// specific physical formats. These instructions cannot interpret layouts - they expect
// raw memory in a precise arrangement.
//
// Key differences from Tensor:
// 1. Tensor: layout(i) computes buffer offset → data[layout(i)]
//    FormattedTensor: always linear access → data[i]
//
// 2. Tensor: Same buffer, multiple interpretations via layout changes
//    FormattedTensor: Layout change requires physical data reorganization
//
// 3. Tensor: For computation and transformation
//    FormattedTensor: For hardware instructions and cache optimization
//
// Example: VNNI format requires transposed 4-element packs. You cannot pass a regular
// Tensor to _tile_dpbf16ps - the instruction reads memory linearly. You MUST use
// FormattedTensor<bfloat16, VNNILayout> where the buffer is physically in VNNI format.
//
// Use this when:
// - Passing data to hardware instructions (AMX, VNNI, etc.)
// - Optimizing for sequential cache access
// - Physical data format matches required specification
export template<typename T, LayoutType L>
struct FormattedTensor {
    std::array<T, L::size_v()> data;

    static constexpr auto layout = L{};

    constexpr size_t size() const { return L::size_v(); }

    // Always linear - data is physically materialized in layout format
    constexpr T& operator()(size_t i) {
        return data[i];
    }
    constexpr const T& operator()(size_t i) const {
        return data[i];
    }

    T* ptr() { return data.data(); }
    const T* ptr() const { return data.data(); }
};

export template<typename T, LayoutType L>
auto make_tensor() {
    return Tensor<T, L>{};
}

export template<typename T, LayoutType L>
auto make_formatted_tensor() {
    return FormattedTensor<T, L>{};
}

// Transform Tensor into FormattedTensor by physically reorganizing data
// This performs an actual copy/gather operation using the source layout
export template<typename T, LayoutType L>
auto format(Tensor<T, L> const& src) {
    auto dst = make_formatted_tensor<T, L>();
    for (size_t i = 0; i < src.size(); ++i) {
        dst(i) = src(i);  // src uses layout mapping, dst is linear
    }
    return dst;
}

// Convert FormattedTensor back to regular Tensor
// The data is already in the right format, just copy it over
export template<typename T, LayoutType L>
auto unformat(FormattedTensor<T, L> const& src) {
    auto dst = make_tensor<T, L>();
    for (size_t i = 0; i < src.size(); ++i) {
        dst(i) = src(i);  // Both linear since src is formatted
    }
    return dst;
}
