module;
#include <cstddef>
#include <utility>
#include <iterator>
export module layoututils;
import layoutalg;

export template<LayoutType L>
struct LayoutOffsetIterator {
    size_t index;
    using value_type = size_t;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::input_iterator_tag;

    constexpr size_t operator*() const { return L{}(index); }
    constexpr LayoutOffsetIterator& operator++() { ++index; return *this; }
    constexpr LayoutOffsetIterator operator++(int) { auto tmp = *this; ++index; return tmp; }
    constexpr bool operator==(const LayoutOffsetIterator&) const = default;
};

export template<LayoutType L>
struct LayoutOffsetView {
    constexpr auto begin() const { return LayoutOffsetIterator<L>{0}; }
    constexpr auto end() const { return LayoutOffsetIterator<L>{L::size_v()}; }
    constexpr size_t size() const { return L::size_v(); }
};

export template<LayoutType L>
constexpr auto offsets(L) {
    return LayoutOffsetView<L>{};
}

export template<LayoutType SrcL, LayoutType DstL>
struct LayoutPairIterator {
    size_t index;
    using value_type = std::pair<size_t, size_t>;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::input_iterator_tag;

    constexpr auto operator*() const {
        return std::pair{SrcL{}(index), DstL{}(index)};
    }
    constexpr LayoutPairIterator& operator++() { ++index; return *this; }
    constexpr LayoutPairIterator operator++(int) { auto tmp = *this; ++index; return tmp; }
    constexpr bool operator==(const LayoutPairIterator&) const = default;
};

export template<LayoutType SrcL, LayoutType DstL>
struct LayoutPairView {
    static_assert(SrcL::size_v() == DstL::size_v(),
                  "Source and destination layouts must have same size");

    constexpr auto begin() const { return LayoutPairIterator<SrcL, DstL>{0}; }
    constexpr auto end() const { return LayoutPairIterator<SrcL, DstL>{SrcL::size_v()}; }
    constexpr size_t size() const { return SrcL::size_v(); }
};

export template<LayoutType SrcL, LayoutType DstL>
constexpr auto offset_pairs(SrcL, DstL) {
    return LayoutPairView<SrcL, DstL>{};
}
