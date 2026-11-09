#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cfloat>
#include <omp.h>

#include "include/utils.h"
#include "include/pqueue.hpp"
#include "include/hash_filter.hpp"
#include "include/qg.hpp"
using namespace std;

int K = 100;
int BEAM_SIZE = 128;

SearchResult symqg_parallel(
    const QuantizationGraph& qg, int K, const float* d_query, size_t dim, vidType entry_point)
{
    int degree = qg.get_degree();

    // initialize the hash table and frontiers for the search
    hash_filter<vidType> is_expanded(BEAM_SIZE * BEAM_SIZE);
    new_pqueue_t<vidType> frontier(BEAM_SIZE);
    new_pqueue_t<vidType> result_pool(K);

    // initialize the buffers for fastscan distance estimation
    vector<float> appro_dist(degree);
    vector<uint32_t> raw_buf_32(degree);
    vector<float> result_f_buf(degree);

    // counters for various statistics
    size_t dist_cmps = 0, dist_est = 0, extra_dist_cmps = 0;

    // pre-allocate buffers for query preparation
    size_t padded_dim = static_cast<size_t>(qg.get_padded_dim());
    QuantizationGraph::QueryBuffers query_buffers;
    query_buffers.rotated_query = static_cast<float*>(aligned_alloc(64, padded_dim * sizeof(float)));
    query_buffers.quantized_query = static_cast<uint8_t*>(aligned_alloc(64, padded_dim * sizeof(uint8_t)));
    query_buffers.lut = static_cast<uint8_t*>(aligned_alloc(64, padded_dim << 2));

    // push the starting point to the frontier with max distance
    frontier.push(entry_point, FLT_MAX);

    // quantize the query once to avoid redundant computations
    auto qf = qg.query_prepare(d_query, query_buffers);

    int iter = 0;
    while (frontier.has_unexpanded()) {
        iter++;

        int idx = frontier.get_next_index();
        vidType cur_node = frontier[idx];
        frontier.set_front_expanded();
        if (is_expanded.contains(cur_node)) {
            continue;
        }

        float exact_dist = compute_distance(dim, d_query, qg.get_row(cur_node));
        dist_cmps++;

        result_pool.push(cur_node, exact_dist);
        is_expanded.add(cur_node);

        // expand the current node
        const vidType* neighbors = qg.get_neighbors(cur_node);

        qg.scan_all_neighbors_32(cur_node, exact_dist, appro_dist.data(), qf, raw_buf_32.data(), result_f_buf.data());
        dist_est += degree;
        for (int i = 0; i < degree; i++) {
            if (is_expanded.contains(neighbors[i])) continue;
            float dist = appro_dist[i];
            if (dist < 0.0f) dist = 0.0f;
            if (frontier.size() < BEAM_SIZE || dist < frontier.get_tail_dist()) {
                frontier.push(neighbors[i], dist);
            }
        }
    }
    // final fill up stage of the result pool if it's not full due to duplicates in the frontier
    // expand result pool by visiting neighbors of current results
    int pool_size = result_pool.size();
    if (pool_size != K) {
        for (int i = 0; i < pool_size; i++) {
            vidType data_id = result_pool[i];
            const vidType* neighbors = qg.get_neighbors(data_id);
            for (int j = 0; j < degree; j++) {
                vidType cur_neighbor = neighbors[j];
                if (!is_expanded.contains(cur_neighbor)) {
                    is_expanded.add(cur_neighbor);
                    float dist = compute_distance(dim, d_query, qg.get_row(cur_neighbor));
                    extra_dist_cmps++;
                    result_pool.push(cur_neighbor, dist);
                }
            }
            if (result_pool.size() == K) break;
        }
    }

    // collect the top-K results from candidates and return with distance computation count
    vector<int> result;
    for (int i = 0; i < K; i++)
        result.push_back(result_pool[i]);

    free(query_buffers.rotated_query);
    free(query_buffers.quantized_query);
    free(query_buffers.lut);

    SearchResult res;
    res.neighbors = move(result);
    res.dist_cmps = dist_cmps;
    res.dist_est = dist_est;
    res.extra_dist_cmps = extra_dist_cmps;
    res.iters = iter;
    return res;
}

RunStats run_symqg_parallel(
    const QuantizationGraph& qg, int K, size_t n_query, size_t dim,
    const float* query_vectors, const int* groundtruth, size_t dim_gt
) {
    int actual_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }
    cout << "OpenMP ANN search (" << actual_threads << " threads)\n";

    int64_t total_count_dc = 0, total_count_ds = 0, total_extra_count_dc = 0, total_hop = 0;
    const int query_count = static_cast<int>(n_query);
    vector<vector<int>> all_results(query_count, vector<int>(K));

    vidType entry_point = qg.get_entry_point();

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic, 1) \
        reduction(+:total_count_dc,total_count_ds,total_extra_count_dc,total_hop)
    for (int q = 0; q < query_count; q++) {
        const float* query_ptr = query_vectors + q * dim;

        SearchResult result = symqg_parallel(qg, K, query_ptr, dim, entry_point);

        all_results[q] = result.neighbors;
        total_count_dc += result.dist_cmps;
        total_count_ds += result.dist_est;
        total_extra_count_dc += result.extra_dist_cmps;
        total_hop += result.iters;
    }

    auto end = std::chrono::high_resolution_clock::now();

    RunStats stats;
    stats.runtime = std::chrono::duration<double>(end - start).count();
    stats.latency = stats.runtime / query_count * 1000.0;
    stats.throughput = double(query_count) / stats.runtime;
    stats.recall = compute_avg_recall(all_results, groundtruth, query_count, K, dim_gt);
    stats.total_dc = total_count_dc;
    stats.total_ds = total_count_ds;
    stats.total_extra_dc = total_extra_count_dc;
    stats.total_full_dc = total_count_dc + total_extra_count_dc;
    stats.total_iter = total_hop;
    printf("runtime: %.6f sec, recall@%d: %.6f\n", stats.runtime, K, stats.recall);
    printf("avg latency: %.6f ms/query, throughput: %.6f queries/sec\n",
           stats.latency, stats.throughput);
    cout << "total # distance computation: " << stats.total_full_dc << "\n";
    cout << "average # distance computation: " << stats.total_full_dc / query_count << "\n";
    cout << "total # iteration: " << stats.total_iter << "\n";
    cout << "average # iteration: " << stats.total_iter / query_count << "\n";
    return stats;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0]
             << " -qg <codebook.index> -query <query.fvecs> -gt <groundtruth.ivecs>"
             << " [-degree <int>] [-k <int>] [-beam <int>] [-threads <int>] [-csv <output.csv>]\n";
        return 1;
    }

    string query_file, gt_file, qg_codebook, csv_file;
    int degree = 32;
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (i + 1 >= argc) {
            cerr << "Missing value for " << arg << "\n";
            return 1;
        }
        if (arg == "-qg") qg_codebook = argv[++i];
        else if (arg == "-query") query_file = argv[++i];
        else if (arg == "-gt") gt_file = argv[++i];
        else if (arg == "-csv") csv_file = argv[++i];
        else if (arg == "-degree") degree = atoi(argv[++i]);
        else if (arg == "-k") K = atoi(argv[++i]);
        else if (arg == "-beam") BEAM_SIZE = atoi(argv[++i]);
        else if (arg == "-threads") {
            int threads = atoi(argv[++i]);
            if (threads <= 0) {
                cerr << "Error: -threads must be positive.\n";
                return 1;
            }
            omp_set_num_threads(threads);
        } else {
            cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }
    if (query_file.empty() || gt_file.empty() || qg_codebook.empty()) {
        cerr << "Error: -qg, -query, and -gt are required.\n";
        return 1;
    }
    if (K <= 0 || BEAM_SIZE < K || degree <= 0 || degree % 32 != 0) {
        cerr << "Error: require 0 < k <= beam and degree to be a positive multiple of 32.\n";
        return 1;
    }

    try {
        size_t n_query, dim_query, n_gt, dim_gt;
        unique_ptr<float[]> query_vectors(read_fvecs(query_file.c_str(), n_query, dim_query));
        unique_ptr<int[]> groundtruth(read_ivecs(gt_file.c_str(), n_gt, dim_gt));
        if (n_query == 0 || n_query != n_gt || dim_gt < static_cast<size_t>(K))
            throw runtime_error("Queries, ground truth, or k do not match");

        QuantizationGraph qg(dim_query, degree, qg_codebook);
        if (qg.get_num_nodes() < static_cast<size_t>(K))
            throw runtime_error("k exceeds the number of indexed vectors");
        RunStats stats = run_symqg_parallel(qg, K, n_query, dim_query,
                                                 query_vectors.get(), groundtruth.get(), dim_gt);
        if (!csv_file.empty()) append_stats_to_csv(csv_file, K, BEAM_SIZE, stats);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
