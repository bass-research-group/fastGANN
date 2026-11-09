// FastScan kernels for RaBitQ batch distance estimation.
// Ported from SymphonyQG (symqglib/quantization/fastscan_impl.hpp and
// symqglib/qg/qg_scanner.hpp), which is in turn based on the Faiss FastScan
// implementation: https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)

#pragma once

#include <immintrin.h>

#include <array>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#define LOWBIT(x) ((x) & (-(x)))

constexpr static size_t kBatchSize = 32;

constexpr static std::array<int, 16> kPos = {
    3 /*0000*/, 3 /*0001*/, 2 /*0010*/, 3 /*0011*/,
    1 /*0100*/, 3 /*0101*/, 2 /*0110*/, 3 /*0111*/,
    0 /*1000*/, 3 /*1001*/, 2 /*1010*/, 3 /*1011*/,
    1 /*1100*/, 3 /*1101*/, 2 /*1110*/, 3 /*1111*/,
};

inline void pack_lut_impl(
    size_t dim,
    const uint8_t* __restrict__ byte_query,
    uint8_t* __restrict__ LUT
) {
    size_t num_codebook = dim >> 2;
    for (size_t i = 0; i < num_codebook; ++i) {
        LUT[0] = 0;
        for (int j = 1; j < 16; ++j) {
            LUT[j] = LUT[j - LOWBIT(j)] + byte_query[kPos[j]];
        }
        LUT += 16;
        byte_query += 4;
    }
}

inline void accumulate_impl(
    size_t dim,
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ LUT,
    uint16_t* __restrict__ result
) {
    size_t code_length = dim << 2;

#if defined(__AVX512F__)
    __m512i c, lo, hi, lut, res_lo, res_hi;
    const __m512i lo_mask = _mm512_set1_epi8(0x0f);
    __m512i accu0 = _mm512_setzero_si512();
    __m512i accu1 = _mm512_setzero_si512();
    __m512i accu2 = _mm512_setzero_si512();
    __m512i accu3 = _mm512_setzero_si512();

    for (size_t i = 0; i < code_length; i += 64) {
        c   = _mm512_loadu_si512(&codes[i]);
        lut = _mm512_loadu_si512(&LUT[i]);
        lo  = _mm512_and_si512(c, lo_mask);
        hi  = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask);

        res_lo = _mm512_shuffle_epi8(lut, lo);
        res_hi = _mm512_shuffle_epi8(lut, hi);

        accu0 = _mm512_add_epi16(accu0, res_lo);
        accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
        accu2 = _mm512_add_epi16(accu2, res_hi);
        accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));

    }

    accu0 = _mm512_sub_epi16(accu0, _mm512_slli_epi16(accu1, 8));
    accu2 = _mm512_sub_epi16(accu2, _mm512_slli_epi16(accu3, 8));

    __m512i ret1 = _mm512_add_epi16(
        _mm512_mask_blend_epi64(0b11110000, accu0, accu1),
        _mm512_shuffle_i64x2(accu0, accu1, 0b01001110)
    );
    __m512i ret2 = _mm512_add_epi16(
        _mm512_mask_blend_epi64(0b11110000, accu2, accu3),
        _mm512_shuffle_i64x2(accu2, accu3, 0b01001110)
    );
    __m512i ret = _mm512_setzero_si512();
    ret = _mm512_add_epi16(ret, _mm512_shuffle_i64x2(ret1, ret2, 0b10001000));
    ret = _mm512_add_epi16(ret, _mm512_shuffle_i64x2(ret1, ret2, 0b11011101));
    _mm512_storeu_si512(result, ret);

#elif defined(__AVX2__)
    __m256i c, lo, hi, lut, res_lo, res_hi;
    const __m256i low_mask = _mm256_set1_epi8(0xf);
    __m256i accu0 = _mm256_setzero_si256();
    __m256i accu1 = _mm256_setzero_si256();
    __m256i accu2 = _mm256_setzero_si256();
    __m256i accu3 = _mm256_setzero_si256();

    for (size_t i = 0; i < code_length; i += 64) {
        c   = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&codes[i]));
        lut = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&LUT[i]));
        lo  = _mm256_and_si256(c, low_mask);
        hi  = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);
        res_lo = _mm256_shuffle_epi8(lut, lo);
        res_hi = _mm256_shuffle_epi8(lut, hi);
        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));

        c   = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&codes[i + 32]));
        lut = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&LUT[i + 32]));
        lo  = _mm256_and_si256(c, low_mask);
        hi  = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);
        res_lo = _mm256_shuffle_epi8(lut, lo);
        res_hi = _mm256_shuffle_epi8(lut, hi);
        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));
    }

    accu0 = _mm256_sub_epi16(accu0, _mm256_slli_epi16(accu1, 8));
    __m256i dis0 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu0, accu1, 0x21),
        _mm256_blend_epi32(accu0, accu1, 0xF0)
    );
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(result), dis0);

    accu2 = _mm256_sub_epi16(accu2, _mm256_slli_epi16(accu3, 8));
    __m256i dis1 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu2, accu3, 0x21),
        _mm256_blend_epi32(accu2, accu3, 0xF0)
    );
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result[16]), dis1);

#else
    std::cerr << "fastscan requires AVX512F or AVX2\n";
    abort();
#endif
}

inline void accumulate_impl_32(
    size_t dim,
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ LUT,
    uint32_t* __restrict__ result
) {
    constexpr size_t kSafeChunkDim = 2048;
    alignas(64) uint16_t partial[kBatchSize];

    for (size_t i = 0; i < kBatchSize; ++i) result[i] = 0;

    for (size_t d0 = 0; d0 < dim; d0 += kSafeChunkDim) {
        size_t chunk_dim = std::min(kSafeChunkDim, dim - d0);
        std::fill(partial, partial + kBatchSize, 0);

        accumulate_impl(
            chunk_dim,
            codes + (d0 << 2),
            LUT + (d0 << 2),
            partial
        );

        for (size_t i = 0; i < kBatchSize; ++i) {
            result[i] += static_cast<uint32_t>(partial[i]);
        }
    }
}

inline void appro_dist_impl(
    size_t num_points,
    float sqr_y,
    float width,
    float vl,
    const float* __restrict__ result,
    const float* __restrict__ triple_x,
    const float* __restrict__ fac_dq,
    const float* __restrict__ fac_vq,
    float* __restrict__ appro_dist
) {
#if defined(__AVX512F__)
    const __m512 sqr_y_v = _mm512_set1_ps(sqr_y);
    const __m512 width_v = _mm512_set1_ps(width);
    const __m512 vl_v    = _mm512_set1_ps(vl);

    for (size_t i = 0; i < num_points; i += 16) {
        __m512 res_v    = _mm512_loadu_ps(&result[i]);
        __m512 tx_v     = _mm512_loadu_ps(&triple_x[i]);
        __m512 fac_dq_v = _mm512_loadu_ps(&fac_dq[i]);
        __m512 fac_vq_v = _mm512_loadu_ps(&fac_vq[i]);

        // SymQG marks zero residuals with FLT_MAX; their distance is the parent's.
        const __mmask16 zero_residual = _mm512_cmp_ps_mask(tx_v, _mm512_set1_ps(FLT_MAX), _CMP_EQ_OQ);
        tx_v     = _mm512_add_ps(tx_v, sqr_y_v);
        fac_dq_v = _mm512_mul_ps(fac_dq_v, width_v);
        fac_dq_v = _mm512_mul_ps(fac_dq_v, res_v);
        fac_vq_v = _mm512_fmadd_ps(fac_vq_v, vl_v, tx_v);

        _mm512_storeu_ps(&appro_dist[i], _mm512_mask_mov_ps(_mm512_add_ps(fac_dq_v, fac_vq_v), zero_residual, sqr_y_v));
    }

#elif defined(__AVX2__)
    const __m256 sqr_y_v = _mm256_set1_ps(sqr_y);
    const __m256 width_v = _mm256_set1_ps(width);
    const __m256 vl_v    = _mm256_set1_ps(vl);

    for (size_t i = 0; i < num_points; i += 8) {
        __m256 res_v    = _mm256_loadu_ps(&result[i]);
        __m256 tx_v     = _mm256_loadu_ps(&triple_x[i]);
        __m256 fac_dq_v = _mm256_loadu_ps(&fac_dq[i]);
        __m256 fac_vq_v = _mm256_loadu_ps(&fac_vq[i]);

        const __m256 zero_residual = _mm256_cmp_ps(tx_v, _mm256_set1_ps(FLT_MAX), _CMP_EQ_OQ);
        tx_v     = _mm256_add_ps(tx_v, sqr_y_v);
        fac_dq_v = _mm256_mul_ps(fac_dq_v, width_v);
        fac_dq_v = _mm256_mul_ps(fac_dq_v, res_v);
        fac_vq_v = _mm256_mul_ps(fac_vq_v, vl_v);

        _mm256_storeu_ps(
            &appro_dist[i],
            _mm256_blendv_ps(_mm256_add_ps(_mm256_add_ps(tx_v, fac_dq_v), fac_vq_v), sqr_y_v, zero_residual)
        );
    }

#else
    std::cerr << "appro_dist_impl requires AVX512F or AVX2\n";
    abort();
#endif
}
