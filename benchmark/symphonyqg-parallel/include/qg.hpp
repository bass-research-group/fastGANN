#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "utils.h"
#include "fastscan_impl.hpp"

class QuantizationGraph {
public:
    struct QueryBuffers {
        float* rotated_query = nullptr;
        uint8_t* quantized_query = nullptr;
        uint8_t* lut = nullptr;
    };

    struct QueryFactors {
        float* rotated_query = nullptr;
        uint8_t* quantized_query = nullptr;
        uint8_t* lut = nullptr;
        float low_val = 0.0f, high_val = 0.0f, width = 0.0f;
        int32_t sum_q = 0;
    };

    // Layout: entry point, rows of vectors/codes/factors/neighbors, then rotation signs.
    QuantizationGraph(size_t dim, int degree, const std::string& path)
        : degree_(degree) {
        if (dim == 0 || degree <= 0 || degree % 32 != 0)
            throw std::runtime_error("Invalid codebook dimension or degree");
        padded_dim_ = 1ULL << static_cast<size_t>(std::ceil(std::log2(dim)));
        if (padded_dim_ < 64)
            throw std::runtime_error("SymphonyQG requires a padded dimension of at least 64");
        code_offset_ = dim;
        factor_offset_ = code_offset_ + padded_dim_ / 64 * 2 * degree_;
        neighbor_offset_ = factor_offset_ + 3 * degree_;
        row_offset_ = neighbor_offset_ + degree_;

        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in) throw std::runtime_error("Cannot open codebook: " + path);
        const auto file_bytes = in.tellg();
        const size_t signs_bytes = padded_dim_ * sizeof(float);
        const size_t header_bytes = sizeof(vidType) + signs_bytes;
        if (file_bytes < 0 || static_cast<size_t>(file_bytes) <= header_bytes)
            throw std::runtime_error("Invalid codebook size");
        const size_t data_bytes = static_cast<size_t>(file_bytes) - header_bytes;
        if (data_bytes % (row_offset_ * sizeof(float)) != 0)
            throw std::runtime_error("Codebook size does not match query dimension and degree");
        num_nodes_ = data_bytes / (row_offset_ * sizeof(float));
        data_.reset(static_cast<float*>(std::aligned_alloc(64, (data_bytes + 63) & ~size_t(63))));
        signs_.reset(static_cast<float*>(std::aligned_alloc(64, signs_bytes)));
        if (!data_ || !signs_) throw std::bad_alloc();
        in.seekg(0);
        in.read(reinterpret_cast<char*>(&entry_point_), sizeof(entry_point_));
        in.read(reinterpret_cast<char*>(data_.get()), data_bytes);
        in.read(reinterpret_cast<char*>(signs_.get()), signs_bytes);
        if (!in || entry_point_ >= num_nodes_) throw std::runtime_error("Invalid codebook contents");
        printf("Loaded codebook: %zu nodes, dim=%zu, degree=%d, padded_dim=%zu, row_offset=%zu floats (entry_point=%u)\n",
               num_nodes_, dim, degree_, padded_dim_, row_offset_, entry_point_);
    }

    QueryFactors query_prepare(const float* query_raw, const QueryBuffers& scratch) const {
        constexpr float kQueryLevelsInv = 1.0f / static_cast<float>((1 << QG_BQUERY) - 1);
        QueryFactors qf;
        qf.rotated_query = scratch.rotated_query;
        qf.quantized_query = scratch.quantized_query;
        qf.lut = scratch.lut;
        qf.low_val = FLT_MAX;
        qf.high_val = -FLT_MAX;

        assert(qf.rotated_query != nullptr);
        assert(qf.quantized_query != nullptr);
        assert(qf.lut != nullptr);

        // rotate the query
        rotate_vector(query_raw, qf.rotated_query, signs_.get(), code_offset_, padded_dim_);

        // data range: find the lowest and highest dimension in the rotated query
        for (size_t i = 0; i < padded_dim_; ++i) {
            float tmp = qf.rotated_query[i];
            qf.low_val = std::min(qf.low_val, tmp);
            qf.high_val = std::max(qf.high_val, tmp);
        }
        const float query_span = qf.high_val - qf.low_val;
        qf.width = query_span * kQueryLevelsInv;

        // quantize the query
        const float inv_width = 1.0f / qf.width;
        int32_t sum = 0;
        for (size_t i = 0; i < padded_dim_; ++i) {
            const uint8_t cur = static_cast<uint8_t>(lroundf(((qf.rotated_query[i] - qf.low_val) * inv_width) + 0.5f));
            qf.quantized_query[i] = cur;
            sum += cur;
        }
        qf.sum_q = sum;

        pack_lut_impl(padded_dim_, qf.quantized_query, qf.lut);

        return qf;
    }

    void scan_all_neighbors_32(
        uint32_t node_idx,
        float exact_dist,
        float* appro_dist,
        const QueryFactors& qf,
        uint32_t* result_raw,
        float* result_float
    ) const {
        size_t num_blocks   = static_cast<size_t>(degree_) / kBatchSize;

        // Read the packed neighbor codes and factors from the codebook row.
        const uint8_t* packed_code = reinterpret_cast<const uint8_t*>(get_binary_codes(node_idx));
        const float* factors  = get_factor(node_idx);
        const float* triple_x = factors;
        const float* fac_dq   = triple_x + degree_;
        const float* fac_vq   = fac_dq   + degree_;

        // Step 1: accumulate raw uint32 sums for each block of 32 neighbors
        for (size_t b = 0; b < num_blocks; ++b) {
            accumulate_impl_32(padded_dim_, packed_code + b * padded_dim_ * 4, qf.lut, &result_raw[b * kBatchSize]);
        }

        // Step 2: convert uint32 raw sums to float: result = (raw * 2) - sum_q
    #if defined(__AVX512F__)
        {
            const __m512i qq = _mm512_set1_epi32(qf.sum_q);
            for (size_t i = 0; i < static_cast<size_t>(degree_); i += 16) {
                __m512i i32 = _mm512_loadu_si512(
                    reinterpret_cast<const void*>(&result_raw[i]));
                i32 = _mm512_sub_epi32(_mm512_slli_epi32(i32, 1), qq);
                __m512 f32 = _mm512_cvtepi32_ps(i32);
                _mm512_storeu_ps(&result_float[i], f32);
            }
        }
    #else
        for (size_t i = 0; i < degree_; ++i)
            result_float[i] = static_cast<float>((static_cast<uint32_t>(result_raw[i]) << 1) - qf.sum_q);
    #endif

        // Step 3: compute final approximate distances
        appro_dist_impl(degree_, exact_dist, qf.width, qf.low_val,
            result_float, triple_x, fac_dq, fac_vq, appro_dist);
    }

    const float* get_row(vidType node) const { return data_.get() + static_cast<size_t>(node) * row_offset_; }
    const uint64_t* get_binary_codes(vidType node) const {
        return reinterpret_cast<const uint64_t*>(get_row(node) + code_offset_);
    }
    const float* get_factor(vidType node) const { return get_row(node) + factor_offset_; }
    const vidType* get_neighbors(vidType node) const {
        return reinterpret_cast<const vidType*>(get_row(node) + neighbor_offset_);
    }
    size_t get_num_nodes() const { return num_nodes_; }
    size_t get_padded_dim() const { return padded_dim_; }
    int get_degree() const { return degree_; }
    vidType get_entry_point() const { return entry_point_; }

private:
    size_t num_nodes_, padded_dim_;
    int degree_;
    vidType entry_point_;
    size_t code_offset_, factor_offset_, neighbor_offset_, row_offset_;
    std::unique_ptr<float, decltype(&std::free)> data_{nullptr, std::free};
    std::unique_ptr<float, decltype(&std::free)> signs_{nullptr, std::free};
};
