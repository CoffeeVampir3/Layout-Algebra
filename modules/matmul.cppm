module;
#include <immintrin.h>
#include <cstdint>
#include <cstddef>
export module matmul;
import tensor;
import layoutalg;

struct TileConfig {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
};

// AMX INT8 Matrix Multiplication
//
// A: Tensor<int8_t, LayoutA> - input matrix (MxK), accessed via layout
// B: FormattedTensor<int8_t, LayoutB> - VNNI formatted matrix (KxN), physically in VNNI format
// C: Tensor<int32_t, LayoutC> - output matrix (MxN), accessed via layout
//
// VNNI format requirement: B must be physically arranged as (K/4)x(N*4) with transposed 4-element packs
// This is why B must be FormattedTensor - AMX instructions read memory linearly
export template<typename LayoutA, typename LayoutB, typename LayoutC>
void matmul_amx_int8(
    Tensor<int8_t, LayoutA> const& A,
    FormattedTensor<int8_t, LayoutB> const& B,
    Tensor<int32_t, LayoutC>& C)
{
    // Extract dimensions from layouts
    // Assuming rank-2 layouts with shape (M, K) for A, (K, N) for B, (M, N) for C
    constexpr size_t M = size<get<0, typename LayoutA::Shape>>();
    constexpr size_t K = size<get<1, typename LayoutA::Shape>>();
    constexpr size_t N = size<get<1, typename LayoutC::Shape>>();

    // Tile dimensions (from AMX spec and old code)
    constexpr size_t TILE_M = 16;
    constexpr size_t TILE_K = 64;
    constexpr size_t TILE_N = 16;
    constexpr size_t M_STEP = 32;  // Process 2 tiles vertically
    constexpr size_t N_STEP = 32;  // Process 2 tiles horizontally

    // Configure AMX tiles
    TileConfig cfg{};
    cfg.palette_id = 1;
    // tmm0, tmm1: A tiles (TILE_M x TILE_K)
    cfg.rows[0] = TILE_M;
    cfg.colsb[0] = TILE_K;
    cfg.rows[1] = TILE_M;
    cfg.colsb[1] = TILE_K;
    // tmm2, tmm3: B tiles (TILE_K/4 x TILE_N*4) in VNNI format
    cfg.rows[2] = TILE_K / 4;
    cfg.colsb[2] = TILE_N * 4;
    cfg.rows[3] = TILE_K / 4;
    cfg.colsb[3] = TILE_N * 4;
    // tmm4-7: C accumulator tiles (TILE_M x TILE_N*4)
    cfg.rows[4] = TILE_M;
    cfg.colsb[4] = TILE_N * 4;
    cfg.rows[5] = TILE_M;
    cfg.colsb[5] = TILE_N * 4;
    cfg.rows[6] = TILE_M;
    cfg.colsb[6] = TILE_N * 4;
    cfg.rows[7] = TILE_M;
    cfg.colsb[7] = TILE_N * 4;

    _tile_loadconfig(&cfg);

    // Main computation loop
    for (size_t m = 0; m < M; m += M_STEP) {
        for (size_t n = 0; n < N; n += N_STEP) {
            // Zero accumulator tiles
            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            // Accumulate over K dimension
            for (size_t k = 0; k < K; k += TILE_K) {
                // Compute offsets for A tiles (m, k) and (m+16, k)
                // A is accessed via layout - layout maps logical coords to buffer offsets
                size_t a0_base_idx = m * K + k;  // Logical 1D index
                size_t a1_base_idx = (m + TILE_M) * K + k;

                // For layout access, we need to map through the layout
                // For now, assume row-major contiguous and compute base pointer
                // In general case, would need to handle arbitrary layouts
                const int8_t* a0_ptr = A.ptr() + a0_base_idx;
                const int8_t* a1_ptr = A.ptr() + a1_base_idx;

                // B is FormattedTensor - physically in VNNI format: (K/4) × (N*4)
                // For tile at logical (k, n):
                //   Physical offset = (k/4) * (N*4) + n*4
                //   Stride = N*4 bytes per VNNI row
                size_t vnni_row = k / 4;
                size_t b0_offset = vnni_row * (N * 4) + n * 4;
                size_t b1_offset = vnni_row * (N * 4) + (n + TILE_N) * 4;

                const int8_t* b0_ptr = B.ptr() + b0_offset;
                const int8_t* b1_ptr = B.ptr() + b1_offset;

                // Load A tiles (row-major, stride = K bytes)
                _tile_loadd(0, a0_ptr, K);
                _tile_loadd(1, a1_ptr, K);

                // Load B tiles (VNNI format, stride = N*4 bytes per VNNI row)
                _tile_loadd(2, b0_ptr, N * 4);
                _tile_loadd(3, b1_ptr, N * 4);

                // Multiply-accumulate: C += A * B
                _tile_dpbssd(4, 0, 2);  // C[m:m+16, n:n+16] += A[m:m+16, k:k+64] * B[k:k+64, n:n+16]
                _tile_dpbssd(5, 0, 3);  // C[m:m+16, n+16:n+32] += A[m:m+16, k:k+64] * B[k:k+64, n+16:n+32]
                _tile_dpbssd(6, 1, 2);  // C[m+16:m+32, n:n+16] += A[m+16:m+32, k:k+64] * B[k:k+64, n:n+16]
                _tile_dpbssd(7, 1, 3);  // C[m+16:m+32, n+16:n+32] += A[m+16:m+32, k:k+64] * B[k:k+64, n+16:n+32]
            }

            // Store results to C
            size_t c0_base_idx = m * N + n;
            size_t c1_base_idx = m * N + (n + TILE_N);
            size_t c2_base_idx = (m + TILE_M) * N + n;
            size_t c3_base_idx = (m + TILE_M) * N + (n + TILE_N);

            int32_t* c0_ptr = C.ptr() + c0_base_idx;
            int32_t* c1_ptr = C.ptr() + c1_base_idx;
            int32_t* c2_ptr = C.ptr() + c2_base_idx;
            int32_t* c3_ptr = C.ptr() + c3_base_idx;

            _tile_stored(4, c0_ptr, N * sizeof(int32_t));
            _tile_stored(5, c1_ptr, N * sizeof(int32_t));
            _tile_stored(6, c2_ptr, N * sizeof(int32_t));
            _tile_stored(7, c3_ptr, N * sizeof(int32_t));
        }
    }

    _tile_release();
}

// ========================================
// OLD IMPLEMENTATIONS (kept for reference)
// ========================================

// export template<typename TA, typename TB, typename TC>
//     requires IsRowMajor<TA> && IsVNNI<TB> && IsRowMajor<TC> &&
//              std::same_as<typename TA::value_type, int8_t> &&
//              std::same_as<typename TC::value_type, int32_t>
// void matmul_amx_int8_blocked(TA A, const TB& B, TC C, int thread_id = 0, int num_threads = 1)
// {
//     // Types already checked by requires clause above
//     // Could add runtime optimization hints here with is_contiguous(), etc.
//
//     constexpr size_t TILE_M = 16;
//     constexpr size_t TILE_K = 64;
//     constexpr size_t TILE_N = 16;
//     constexpr size_t M_STEP = 32;
//     constexpr size_t N_STEP = 32;
//     size_t M = A.extent(0);
//     size_t K = A.extent(1);
//     size_t N = C.extent(1);
//     size_t n_per_thread = (N + num_threads - 1) / num_threads;
//     size_t n_start = thread_id * n_per_thread;
//     size_t n_end = std::min(N, n_start + n_per_thread);
//     TileConfig cfg{};
//     cfg.palette_id = 1;
//     cfg.rows[0] = TILE_M;
//     cfg.colsb[0] = TILE_K;
//     cfg.rows[1] = TILE_M;
//     cfg.colsb[1] = TILE_K;
//     cfg.rows[2] = TILE_K / 4;
//     cfg.colsb[2] = TILE_N * 4;
//     cfg.rows[3] = TILE_K / 4;
//     cfg.colsb[3] = TILE_N * 4;
//     cfg.rows[4] = TILE_M;
//     cfg.colsb[4] = TILE_N * 4;
//     cfg.rows[5] = TILE_M;
//     cfg.colsb[5] = TILE_N * 4;
//     cfg.rows[6] = TILE_M;
//     cfg.colsb[6] = TILE_N * 4;
//     cfg.rows[7] = TILE_M;
//     cfg.colsb[7] = TILE_N * 4;
//     _tile_loadconfig(&cfg);
//     for (size_t m = 0; m < M; m += M_STEP) {
//         for (size_t n = n_start; n < n_end; n += N_STEP) {
//             _tile_zero(4);
//             _tile_zero(5);
//             _tile_zero(6);
//             _tile_zero(7);
//             for (size_t k = 0; k < K; k += TILE_K) {
//                 auto a0_ptr = A.row(m) + k;
//                 auto a1_ptr = A.row(m + TILE_M) + k;
//                 auto b0_ptr = B.data() + B.layout(k, n);
//                 auto b1_ptr = B.data() + B.layout(k, n + TILE_N);
//                 _tile_loadd(0, a0_ptr, A.stride_bytes());
//                 _tile_loadd(1, a1_ptr, A.stride_bytes());
//                 _tile_loadd(2, b0_ptr, TILE_N * 4);
//                 _tile_loadd(3, b1_ptr, TILE_N * 4);
//                 _tile_dpbssd(4, 0, 2);
//                 _tile_dpbssd(5, 0, 3);
//                 _tile_dpbssd(6, 1, 2);
//                 _tile_dpbssd(7, 1, 3);
//             }
//             _tile_stored(4, C.row(m) + n, C.stride_bytes());
//             _tile_stored(5, C.row(m) + n + TILE_N, C.stride_bytes());
//             _tile_stored(6, C.row(m + TILE_M) + n, C.stride_bytes());
//             _tile_stored(7, C.row(m + TILE_M) + n + TILE_N, C.stride_bytes());
//         }
//     }
//     _tile_release();
// }

// export template<typename TA, typename TB, typename TC>
//     requires IsRowMajor<TA> && IsVNNI<TB> && IsRowMajor<TC> &&
//              std::same_as<typename TA::value_type, int8_t> &&
//              std::same_as<typename TC::value_type, int32_t>
// void matmul_amx_int8_blocked_mt(TA A, const TB& B, TC C, int num_threads = 0)
// {
//     // Types already checked by requires clause above
//     // Could add runtime optimization hints here with is_contiguous(), etc.
//
//     constexpr size_t TILE_M = 16;
//     constexpr size_t TILE_K = 64;
//     constexpr size_t TILE_N = 16;
//     constexpr size_t M_STEP = 32;
//     constexpr size_t N_STEP = 32;
//     size_t M = A.extent(0);
//     size_t K = A.extent(1);
//     size_t N = C.extent(1);
//     if (num_threads == 0) {
//         num_threads = std::thread::hardware_concurrency();
//     }
//     auto worker = [&](int tid) {
//         int cpu_id = tid;
//         cpu_set_t cpuset;
//         CPU_ZERO(&cpuset);
//         CPU_SET(cpu_id, &cpuset);
//         pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
//         size_t M_blocks = (M + M_STEP - 1) / M_STEP;
//         size_t blocks_per_thread = (M_blocks + num_threads - 1) / num_threads;
//         size_t block_start = tid * blocks_per_thread;
//         size_t block_end = std::min(M_blocks, block_start + blocks_per_thread);
//         if (block_start >= M_blocks) return;
//         TileConfig cfg{};
//         cfg.palette_id = 1;
//         cfg.rows[0] = TILE_M;
//         cfg.colsb[0] = TILE_K;
//         cfg.rows[1] = TILE_M;
//         cfg.colsb[1] = TILE_K;
//         cfg.rows[2] = TILE_K / 4;
//         cfg.colsb[2] = TILE_N * 4;
//         cfg.rows[3] = TILE_K / 4;
//         cfg.colsb[3] = TILE_N * 4;
//         cfg.rows[4] = TILE_M;
//         cfg.colsb[4] = TILE_N * 4;
//         cfg.rows[5] = TILE_M;
//         cfg.colsb[5] = TILE_N * 4;
//         cfg.rows[6] = TILE_M;
//         cfg.colsb[6] = TILE_N * 4;
//         cfg.rows[7] = TILE_M;
//         cfg.colsb[7] = TILE_N * 4;
//         _tile_loadconfig(&cfg);
//         for (size_t mb = block_start; mb < block_end; ++mb) {
//             size_t m = mb * M_STEP;
//             if (m >= M) break;
//             for (size_t n = 0; n < N; n += N_STEP) {
//                 _tile_zero(4);
//                 _tile_zero(5);
//                 _tile_zero(6);
//                 _tile_zero(7);
//                 for (size_t k = 0; k < K; k += TILE_K) {
//                     auto a0_ptr = A.row(m) + k;
//                     auto a1_ptr = A.row(m + TILE_M) + k;
//                     auto b0_ptr = B.data() + B.layout(k, n);
//                     auto b1_ptr = B.data() + B.layout(k, n + TILE_N);
//                     _tile_loadd(0, a0_ptr, A.stride_bytes());
//                     _tile_loadd(1, a1_ptr, A.stride_bytes());
//                     _tile_loadd(2, b0_ptr, TILE_N * 4);
//                     _tile_loadd(3, b1_ptr, TILE_N * 4);
//                     _tile_dpbssd(4, 0, 2);
//                     _tile_dpbssd(5, 0, 3);
//                     _tile_dpbssd(6, 1, 2);
//                     _tile_dpbssd(7, 1, 3);
//                 }
//                 _tile_stored(4, C.row(m) + n, C.stride_bytes());
//                 _tile_stored(5, C.row(m) + n + TILE_N, C.stride_bytes());
//                 _tile_stored(6, C.row(m + TILE_M) + n, C.stride_bytes());
//                 _tile_stored(7, C.row(m + TILE_M) + n + TILE_N, C.stride_bytes());
//             }
//         }
//         _tile_release();
//     };
//     std::vector<std::jthread> threads;
//     threads.reserve(num_threads);
//     for (int t = 0; t < num_threads; ++t) {
//         threads.emplace_back(worker, t);
//     }
// }
