#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstring>
#include <cfloat>
#include <cstdlib>
#include <filesystem>

#if !defined(__CUDACC__) && (defined(__AVX512F__) || defined(__AVX2__) || defined(__SSE2__))
#include <immintrin.h>
#endif

using namespace std;
typedef uint32_t vidType;

#define QG_BQUERY 6

inline int* read_ivecs(const char* filename, size_t& n_out, size_t& d_out) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open ivecs file");

    int d;
    input.read((char*)&d, sizeof(int));
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    size_t num = file_size / ((d + 1) * sizeof(int));
    int* data = new int[num * d];

    for (size_t i = 0; i < num; ++i) {
        int dim;
        input.read((char*)&dim, sizeof(int));
        assert(dim == d);
        input.read((char*)(data + i * d), sizeof(int) * d);
    }

    d_out = d;
    n_out = num;
    return data;
}

inline float* read_fvecs(const char* filename, size_t& n_out, size_t& d_out) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open fvecs file");

    int d;
    input.read((char*)&d, sizeof(int));
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    size_t num = file_size / (d * sizeof(float) + sizeof(int));
    float* data = new float[num * d];

    for (size_t i = 0; i < num; ++i) {
        int dim;
        input.read((char*)&dim, sizeof(int));
        assert(dim == d);
        input.read((char*)(data + i * d), sizeof(float) * d);
    }

    d_out = d;
    n_out = num;
    return data;
}

struct RunStats {
    double runtime = 0, latency = 0, throughput = 0, recall = 0;
    int64_t total_dc = 0, total_ds = 0, total_extra_dc = 0, total_full_dc = 0, total_iter = 0;
};

struct SearchResult {
    vector<int> neighbors;
    size_t dist_cmps = 0, dist_est = 0, extra_dist_cmps = 0, iters = 0;
};

// ─── Scalar fallback distance ───────────────────────────────────────
inline float compute_distance_scalar(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

#ifdef __AVX2__
// from DiskANN
static inline float _mm256_reduce_add_ps(__m256 x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}

inline float compute_distance_vec256(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    // assume size is divisible by 8
    uint16_t niters = (uint16_t)(dim / 8);
    __m256 sum = _mm256_setzero_ps();
    for (uint16_t j = 0; j < niters; j++) {
        // scope is a[8j:8j+7], b[8j:8j+7]
        if (j+1 < niters) {
        _mm_prefetch((char *)(a + 8 * (j + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(b + 8 * (j + 1)), _MM_HINT_T0);
        }
        __m256 a_vec = _mm256_loadu_ps(a + 8 * j);
        __m256 b_vec = _mm256_loadu_ps(b + 8 * j);
        __m256 tmp_vec = _mm256_sub_ps(a_vec, b_vec);
        sum = _mm256_fmadd_ps(tmp_vec, tmp_vec, sum);
    }
    // horizontal add sum
    float dist = _mm256_reduce_add_ps(sum);
    for (int i = static_cast<int>(niters) * 8; i < dim; ++i) {
        const float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

#endif // __AVX2__

#ifdef __AVX512F__
// adapted from DiskANN
inline float compute_distance_vec512(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    uint16_t niters = (uint16_t)(dim / 16);
    __m512 sum = _mm512_setzero_ps();
    for (uint16_t j = 0; j < niters; j++) {
        // scope is a[16j:16j+15], b[16j:16j+15]
        if (j+1 < niters) {
        _mm_prefetch((char *)(a + 16 * (j + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(b + 16 * (j + 1)), _MM_HINT_T0);
        }
        __m512 a_vec = _mm512_loadu_ps(a + 16 * j);
        __m512 b_vec = _mm512_loadu_ps(b + 16 * j);
        __m512 tmp_vec = _mm512_sub_ps(a_vec, b_vec);
        sum = _mm512_fmadd_ps(tmp_vec, tmp_vec, sum);
    }
    // horizontal add sum
    float dist = _mm512_reduce_add_ps(sum);
    for (int i = static_cast<int>(niters) * 16; i < dim; ++i) {
        const float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

#endif // __AVX512F__

// ─── Best-available distance function ───────────────────────────────
inline float compute_distance(int dim, const float* __restrict__ a, const float* __restrict__ b) {
#ifdef __AVX512F__
    return compute_distance_vec512(dim, a, b);
#elif defined(__AVX2__)
    return compute_distance_vec256(dim, a, b);
#else
    return compute_distance_scalar(dim, a, b);
#endif
}

inline float recall_at_k(const vector<int>& ann_result, const int* gt_ptr, int K, int gt_k) {
    unordered_set<int> ann_set(ann_result.begin(), ann_result.begin() + K);

    int hit = 0;
    for (int i = 0; i < std::min(K, gt_k); i++) {
        if (ann_set.count(gt_ptr[i])) hit++;
    }
    return float(hit) / K;
}

// compute average recall for all queries
inline float compute_avg_recall(const vector<std::vector<int>>& all_results,
                                const int* all_groundtruth, size_t n_query, int K, int gt_k) {
    float total_recall = 0.0f;
    for (size_t q = 0; q < n_query; q++) {
        const int* gt_ptr = all_groundtruth + q * gt_k;
        total_recall += recall_at_k(all_results[q], gt_ptr, K, gt_k);
    }
    return total_recall / n_query * 100.0f;
}

// Fast Walsh-Hadamard Transform (in-place, power-of-2 B, B >= 8)
inline void fwht_inplace(float* data, size_t B) {
#ifdef __AVX512F__
    // ---- AVX-512: 16 floats per register ----

    // len=1: butterfly on adjacent pairs
    for (size_t i = 0; i < B; i += 16) {
        __m512 v = _mm512_load_ps(data + i);
        __m512 s = _mm512_permute_ps(v, 0xB1); // [1,0,3,2,...,15,14]
        __m512 add = _mm512_add_ps(v, s);
        __m512 sub = _mm512_sub_ps(s, v);
        __m512 r = _mm512_mask_blend_ps(0xAAAA, add, sub);
        _mm512_store_ps(data + i, r);
    }

    // len=2: butterfly on pairs 2 apart
    for (size_t i = 0; i < B; i += 16) {
        __m512 v = _mm512_load_ps(data + i);
        __m512 s = _mm512_permute_ps(v, 0x4E); // [2,3,0,1,6,7,4,5,...]
        __m512 add = _mm512_add_ps(v, s);
        __m512 sub = _mm512_sub_ps(s, v);
        __m512 r = _mm512_mask_blend_ps(0xCCCC, add, sub);
        _mm512_store_ps(data + i, r);
    }

    // len=4: butterfly on pairs 4 apart (swap 128-bit lanes pairwise)
    for (size_t i = 0; i < B; i += 16) {
        __m512 v = _mm512_load_ps(data + i);
        __m512 s = _mm512_shuffle_f32x4(v, v, 0xB1); // swap 128-bit lane pairs
        __m512 add = _mm512_add_ps(v, s);
        __m512 sub = _mm512_sub_ps(s, v);
        __m512 r = _mm512_mask_blend_ps(0xF0F0, add, sub);
        _mm512_store_ps(data + i, r);
    }

    // len=8: butterfly on pairs 8 apart (swap 256-bit halves)
    for (size_t i = 0; i < B; i += 16) {
        __m512 v = _mm512_load_ps(data + i);
        __m512 s = _mm512_shuffle_f32x4(v, v, 0x4E); // swap 256-bit halves
        __m512 add = _mm512_add_ps(v, s);
        __m512 sub = _mm512_sub_ps(s, v);
        __m512 r = _mm512_mask_blend_ps(0xFF00, add, sub);
        _mm512_store_ps(data + i, r);
    }

    // len >= 16: pairs in separate registers
    for (size_t len = 16; len < B; len <<= 1) {
        for (size_t i = 0; i < B; i += len << 1) {
            for (size_t j = 0; j < len; j += 16) {
                __m512 u = _mm512_load_ps(data + i + j);
                __m512 v = _mm512_load_ps(data + i + j + len);
                _mm512_store_ps(data + i + j,       _mm512_add_ps(u, v));
                _mm512_store_ps(data + i + j + len,  _mm512_sub_ps(u, v));
            }
        }
    }

    // // Scale by 1/sqrt(B)
    // __m512 sc = _mm512_set1_ps(1.0f / std::sqrt(static_cast<float>(B)));
    // for (size_t i = 0; i < B; i += 16)
    //     _mm512_store_ps(data + i, _mm512_mul_ps(_mm512_load_ps(data + i), sc));

#elif defined(__AVX2__)
    // ---- AVX2: 8 floats per register ----

    // len=1: butterfly on adjacent pairs
    for (size_t i = 0; i < B; i += 8) {
        __m256 v = _mm256_load_ps(data + i);
        __m256 s = _mm256_permute_ps(v, 0xB1); // [1,0,3,2,5,4,7,6]
        __m256 r = _mm256_blend_ps(_mm256_add_ps(v, s), _mm256_sub_ps(s, v), 0xAA);
        _mm256_store_ps(data + i, r);
    }

    // len=2: butterfly on pairs 2 apart
    for (size_t i = 0; i < B; i += 8) {
        __m256 v = _mm256_load_ps(data + i);
        __m256 s = _mm256_permute_ps(v, 0x4E); // [2,3,0,1,6,7,4,5]
        __m256 r = _mm256_blend_ps(_mm256_add_ps(v, s), _mm256_sub_ps(s, v), 0xCC);
        _mm256_store_ps(data + i, r);
    }

    // len=4: butterfly on pairs 4 apart (swap 128-bit halves)
    for (size_t i = 0; i < B; i += 8) {
        __m256 v = _mm256_load_ps(data + i);
        __m256 s = _mm256_permute2f128_ps(v, v, 0x01);
        __m256 r = _mm256_blend_ps(_mm256_add_ps(v, s), _mm256_sub_ps(s, v), 0xF0);
        _mm256_store_ps(data + i, r);
    }

    // len >= 8: pairs in separate registers
    for (size_t len = 8; len < B; len <<= 1) {
        for (size_t i = 0; i < B; i += len << 1) {
            for (size_t j = 0; j < len; j += 8) {
                __m256 u = _mm256_load_ps(data + i + j);
                __m256 v = _mm256_load_ps(data + i + j + len);
                _mm256_store_ps(data + i + j,       _mm256_add_ps(u, v));
                _mm256_store_ps(data + i + j + len,  _mm256_sub_ps(u, v));
            }
        }
    }


#else
    // ---- Scalar fallback ----
    for (size_t len = 1; len < B; len <<= 1) {
        for (size_t i = 0; i < B; i += len << 1) {
            for (size_t j = 0; j < len; j++) {
                float u = data[i + j];
                float v = data[i + j + len];
                data[i + j]       = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
#endif
}

inline void rotate_vector(const float* src, float* dst, const float* signs,
                         size_t dim, size_t padded_dim) {
    size_t i = 0;
#ifdef __AVX512F__
    for (; i + 16 <= dim; i += 16) {
        __m512 v_src   = _mm512_loadu_ps(src + i);
        __m512 v_signs = _mm512_loadu_ps(signs + i);
        _mm512_store_ps(dst + i, _mm512_mul_ps(v_src, v_signs));
    }
#elif defined(__AVX2__)
    for (; i + 8 <= dim; i += 8) {
        __m256 v_src   = _mm256_loadu_ps(src + i);
        __m256 v_signs = _mm256_loadu_ps(signs + i);
        _mm256_store_ps(dst + i, _mm256_mul_ps(v_src, v_signs));
    }
#endif
    for (; i < dim; i++) {
        dst[i] = src[i] * signs[i];
    }
    if (padded_dim > dim) {
        std::fill(dst + dim, dst + padded_dim, 0.0f);
    }
    fwht_inplace(dst, padded_dim);
}

inline void append_stats_to_csv(const std::string& filename, int k, int beam, const RunStats& stats) {
    bool has_header = std::filesystem::exists(filename) && std::filesystem::file_size(filename) != 0;
    std::ofstream out(filename, std::ios::app);
    if (!out) throw std::runtime_error("Cannot open CSV file: " + filename);
    if (!has_header) {
        out << "TopK,Beam_size,Runtime,Latency,Throughput,Recall,Total_dist_comp,Total_dist_est,"
               "Total_extra_dist_comp,Total_full_dist_comp,Total_iter\n";
    }
    out << k << ',' << beam << ',' << stats.runtime << ',' << stats.latency << ','
        << stats.throughput << ',' << stats.recall << ',' << stats.total_dc << ',' << stats.total_ds << ','
        << stats.total_extra_dc << ',' << stats.total_full_dc << ',' << stats.total_iter << '\n';
}
