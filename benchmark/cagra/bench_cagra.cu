// cuVS CAGRA search on an external fixed-degree graph. The base vectors and the graph are copied to the device once
// and a cagra::index is built over them; the queries are uploaded once, outside the timing. For each beam it sets
// itopk_size = beam, search_width 1 and max_iterations 0 (auto), runs 2 untimed searches, then reports the median
// of 3 timed ones, each between device synchronizations.
//
// usage: bench_cagra <base.fvecs> <query.fvecs> <gt.ivecs> <graph.bin> <k> <beam>[,<beam>...] [algo=multi]
//   graph.bin: uint32 n, uint32 degree, then n x degree uint32 neighbor ids (the SymQG graph format)
//   algo: auto | single | multi | kernel
// Output: one CSV line per beam: TopK,Beam_size,Runtime,Latency,Throughput,Recall.
// Recall counts distinct returned ids found in the ground truth's first k.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>

constexpr int WARMUP = 2;
constexpr int REPEAT = 3;

/** @brief Read a .fvecs / .ivecs file into rows x dim 4-byte values. */
template <typename T>
static std::vector<T> readvecs(const std::string& path, int64_t& rows, int64_t& dim) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    int32_t d = 0;
    in.read(reinterpret_cast<char*>(&d), 4);
    dim = d;
    in.seekg(0, std::ios::end);
    rows = static_cast<int64_t>(in.tellg()) / (4 * (dim + 1));
    in.seekg(0);
    std::vector<T> out(rows * dim);
    for (int64_t i = 0; i < rows; ++i) {
        in.read(reinterpret_cast<char*>(&d), 4);
        in.read(reinterpret_cast<char*>(out.data() + i * dim), 4 * dim);
    }
    return out;
}

/** @brief Copy a host row-major matrix to a new device matrix and wait for it. */
template <typename V>
static raft::device_matrix<V, int64_t> upload(const raft::device_resources& res, const V* host, int64_t rows,
                                              int64_t cols) {
    auto m = raft::make_device_matrix<V, int64_t>(res, rows, cols);
    raft::copy(m.data_handle(), host, rows * cols, raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    return m;
}

/** @brief Map an algorithm name onto the cuVS CAGRA search algorithm. */
static cuvs::neighbors::cagra::search_algo algoof(const std::string& name) {
    using cuvs::neighbors::cagra::search_algo;
    if (name == "auto") return search_algo::AUTO;
    if (name == "single") return search_algo::SINGLE_CTA;
    if (name == "multi") return search_algo::MULTI_CTA;
    if (name == "kernel") return search_algo::MULTI_KERNEL;
    throw std::runtime_error("unknown CAGRA algorithm " + name);
}

int main(int argc, char** argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s <base.fvecs> <query.fvecs> <gt.ivecs> <graph.bin> <k> <beams> [algo=multi]\n", argv[0]);
        return 1;
    }
    const int64_t k = std::atol(argv[5]);
    std::vector<int64_t> beams;
    for (std::string s = argv[6]; !s.empty();) {
        const size_t c = s.find(',');
        beams.push_back(std::stol(s.substr(0, c)));
        s = c == std::string::npos ? "" : s.substr(c + 1);
    }
    const std::string algo = argc > 7 ? argv[7] : "multi";

    // [1] host inputs
    int64_t nb = 0, dim = 0, nq = 0, qdim = 0, ngt = 0, gdim = 0;
    const std::vector<float> base = readvecs<float>(argv[1], nb, dim);
    const std::vector<float> query = readvecs<float>(argv[2], nq, qdim);
    const std::vector<int32_t> gt = readvecs<int32_t>(argv[3], ngt, gdim);
    std::ifstream gin(argv[4], std::ios::binary);
    uint32_t head[2] = {0, 0};
    gin.read(reinterpret_cast<char*>(head), 8);
    std::vector<uint32_t> edges(static_cast<size_t>(head[0]) * head[1]);
    gin.read(reinterpret_cast<char*>(edges.data()), static_cast<std::streamsize>(4 * edges.size()));
    if (!gin || head[0] != nb || qdim != dim || ngt != nq || gdim < k) throw std::runtime_error("inputs do not match");

    // [2] device copies and the cuVS index over the external graph
    raft::device_resources res;
    auto stream = raft::resource::get_cuda_stream(res);
    auto dbase = upload(res, base.data(), nb, dim);
    auto dgraph = upload(res, edges.data(), nb, static_cast<int64_t>(head[1]));
    cuvs::neighbors::cagra::index<float, uint32_t> index(res, cuvs::distance::DistanceType::L2Expanded,
                                                         raft::make_const_mdspan(dbase.view()),
                                                         raft::make_const_mdspan(dgraph.view()));
    auto dquery = upload(res, query.data(), nq, dim);
    auto nbrs = raft::make_device_matrix<uint32_t, int64_t>(res, nq, k);
    auto dists = raft::make_device_matrix<float, int64_t>(res, nq, k);
    printf("n %ld, dim %ld, degree %u, %ld queries, k %ld, algo %s\n", nb, dim, head[1], nq, k, algo.c_str());
    printf("TopK,Beam_size,Runtime,Latency,Throughput,Recall\n");

    // [3] per beam: WARMUP untimed searches, then the median of REPEAT timed ones
    std::vector<uint32_t> ids(static_cast<size_t>(nq) * k);
    for (int64_t beam : beams) {
        cuvs::neighbors::cagra::search_params params;
        params.algo = algoof(algo);
        params.itopk_size = static_cast<size_t>(beam);
        params.search_width = 1;
        params.max_iterations = 0;
        double times[REPEAT];
        for (int run = -WARMUP; run < REPEAT; ++run) {
            cudaDeviceSynchronize();
            const auto start = std::chrono::steady_clock::now();
            cuvs::neighbors::cagra::search(res, params, index, raft::make_const_mdspan(dquery.view()), nbrs.view(),
                                           dists.view());
            cudaDeviceSynchronize();
            if (run >= 0) times[run] = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
        std::sort(times, times + REPEAT);
        const double secs = times[REPEAT / 2];
        raft::copy(ids.data(), nbrs.data_handle(), ids.size(), stream);
        raft::resource::sync_stream(res);
        size_t hits = 0;
        for (int64_t i = 0; i < nq; ++i) {
            std::unordered_set<int32_t> truth(gt.begin() + i * gdim, gt.begin() + i * gdim + k);
            std::unordered_set<uint32_t> seen;
            for (int64_t j = 0; j < k; ++j) {
                const uint32_t id = ids[i * k + j];
                if (seen.insert(id).second && truth.count(static_cast<int32_t>(id))) ++hits;
            }
        }
        printf("%ld,%ld,%g,%g,%g,%g\n", k, beam, secs, secs / nq * 1000.0, nq / secs, 100.0 * hits / (k * nq));
        fflush(stdout);
    }
    return 0;
}
