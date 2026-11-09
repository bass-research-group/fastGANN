// ParlayANN exact beam search on a fixed-degree graph: ParlayANN's own qsearchAll (the query path of its vamana
// `neighbors` driver with no quantization) over every query, one timed pass per trial, as check_nn_recall times it.
//
// usage: bench_parlayann <base.fvecs> <query.fvecs> <gt.ivecs> <graph.bin> <entry> <k> <beam>[,<beam>...] [trials=3]
//   graph.bin: uint32 n, uint32 degree, then n x degree uint32 neighbor ids (the SymQG graph format)
//   entry: start node; -1 picks the base vector nearest to the centroid
// The base, query and graph are converted once to ParlayANN's .fbin / graph files in $PARLAY_CACHE (default ./parlay_cache)
// and loaded with ParlayANN's own readers. Output: one CSV line per beam and trial:
//   beam,trial,recall,qps,avg_dist_cmps,avg_visited
// Recall counts distinct returned ids found in the ground truth's first k.

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "utils/beamSearch.h"
#include "utils/euclidian_point.h"
#include "utils/graph.h"
#include "utils/point_range.h"
#include "utils/stats.h"
#include "utils/types.h"

using namespace parlayANN;
using Points = PointRange<Euclidian_Point<float>>;

/** @brief Read a .fvecs / .ivecs file into rows x dim 4-byte values. */
template <typename T>
static std::vector<T> readvecs(const std::string& path, size_t& rows, size_t& dim) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    int32_t d = 0;
    in.read(reinterpret_cast<char*>(&d), 4);
    dim = static_cast<size_t>(d);
    in.seekg(0, std::ios::end);
    rows = static_cast<size_t>(in.tellg()) / (4 * (dim + 1));
    in.seekg(0);
    std::vector<T> out(rows * dim);
    for (size_t i = 0; i < rows; ++i) {
        in.read(reinterpret_cast<char*>(&d), 4);
        in.read(reinterpret_cast<char*>(out.data() + i * dim), 4 * dim);
    }
    return out;
}

/** @brief Write rows x dim floats as ParlayANN's .fbin (uint32 rows, uint32 dim, values). */
static void writefbin(const std::string& path, const std::vector<float>& v, size_t rows, size_t dim) {
    std::ofstream out(path, std::ios::binary);
    const uint32_t head[2] = {static_cast<uint32_t>(rows), static_cast<uint32_t>(dim)};
    out.write(reinterpret_cast<const char*>(head), 8);
    out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(4 * rows * dim));
}

/** @brief Convert a fixed-degree graph to ParlayANN's graph file (n, max degree, n degrees, edges). */
static void writegraph(const std::string& src, const std::string& dst) {
    std::ifstream in(src, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + src);
    uint32_t head[2];
    in.read(reinterpret_cast<char*>(head), 8);
    std::vector<uint32_t> ids(static_cast<size_t>(head[0]) * head[1]);
    in.read(reinterpret_cast<char*>(ids.data()), static_cast<std::streamsize>(4 * ids.size()));
    std::vector<uint32_t> degrees(head[0], head[1]);
    std::ofstream out(dst, std::ios::binary);
    out.write(reinterpret_cast<const char*>(head), 8);
    out.write(reinterpret_cast<const char*>(degrees.data()), static_cast<std::streamsize>(4 * degrees.size()));
    out.write(reinterpret_cast<const char*>(ids.data()), static_cast<std::streamsize>(4 * ids.size()));
}

/** @brief Return the base vector nearest to the centroid. */
static uint32_t centroidnn(const std::vector<float>& base, size_t rows, size_t dim) {
    std::vector<double> c(dim, 0.0);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < dim; ++j) c[j] += base[i * dim + j];
    for (double& x : c) x /= static_cast<double>(rows);
    auto dists = parlay::tabulate(rows, [&](size_t i) {
        double s = 0.0;
        for (size_t j = 0; j < dim; ++j) {
            const double t = base[i * dim + j] - c[j];
            s += t * t;
        }
        return s;
    });
    return static_cast<uint32_t>(parlay::min_element(dists) - dists.begin());
}

int main(int argc, char** argv) {
    if (argc < 8) {
        fprintf(stderr, "usage: %s <base.fvecs> <query.fvecs> <gt.ivecs> <graph.bin> <entry> <k> <beams> [trials=3]\n",
                argv[0]);
        return 1;
    }
    const std::string basef = argv[1], queryf = argv[2], gtf = argv[3], graphf = argv[4];
    long entry = std::atol(argv[5]);
    const long k = std::atol(argv[6]);
    std::vector<long> beams;
    for (std::string s = argv[7]; !s.empty();) {
        const size_t c = s.find(',');
        beams.push_back(std::stol(s.substr(0, c)));
        s = c == std::string::npos ? "" : s.substr(c + 1);
    }
    const int trials = argc > 8 ? std::atoi(argv[8]) : 3;

    // [1] ground truth, and ParlayANN's input files written once from the .fvecs and graph
    size_t nq = 0, gdim = 0, nb = 0, dim = 0;
    const std::vector<int32_t> gt = readvecs<int32_t>(gtf, nq, gdim);
    const char* cache = std::getenv("PARLAY_CACHE");
    const std::string dir = cache ? cache : "parlay_cache";
    std::filesystem::create_directories(dir);
    const std::string stem = dir + "/" + std::filesystem::path(graphf).stem().string();
    const std::string bbin = stem + ".base.fbin", qbin = stem + ".query.fbin", gbin = stem + ".graph";
    if (!std::filesystem::exists(bbin) || entry < 0) {
        const std::vector<float> base = readvecs<float>(basef, nb, dim);
        if (!std::filesystem::exists(bbin)) writefbin(bbin, base, nb, dim);
        if (entry < 0) entry = centroidnn(base, nb, dim);
    }
    if (!std::filesystem::exists(qbin)) {
        size_t rows = 0, qd = 0;
        const std::vector<float> query = readvecs<float>(queryf, rows, qd);
        writefbin(qbin, query, rows, qd);
    }
    if (!std::filesystem::exists(gbin)) writegraph(graphf, gbin);

    // [2] ParlayANN's readers, as its neighbors driver loads them
    Points Base(const_cast<char*>(bbin.c_str()));
    Points Query(const_cast<char*>(qbin.c_str()));
    Graph<uint32_t> G(const_cast<char*>(gbin.c_str()));
    printf("entry %ld, threads %zu, %zu queries, k %ld\n", entry, parlay::num_workers(), Query.size(), k);
    printf("beam,trial,recall,qps,avg_dist_cmps,avg_visited\n");

    // [3] per beam and trial: one timed qsearchAll over all queries (exact points stand in for the quantized ones)
    for (long beam : beams) {
        QueryParams QP(k, beam, static_cast<long>(G.size()), G.max_degree(), 10, .125);
        for (int t = 1; t <= trials; ++t) {
            stats<uint32_t> QS(Query.size());
            QS.clear();
            auto volatile flush = parlay::random_permutation<long>(5000000);
            parlay::internal::timer timer;
            timer.next_time();
            auto res = qsearchAll<Points, Points, Points, uint32_t>(Query, Query, Query, G, Base, Base, Base, QS,
                                                                  static_cast<uint32_t>(entry), QP);
            const double secs = timer.next_time();
            size_t hits = 0;
            for (size_t i = 0; i < Query.size(); ++i) {
                std::unordered_set<int32_t> truth(gt.begin() + i * gdim, gt.begin() + i * gdim + k);
                std::unordered_set<uint32_t> seen;
                for (long j = 0; j < k && j < static_cast<long>(res[i].size()); ++j)
                    if (seen.insert(res[i][j]).second && truth.count(static_cast<int32_t>(res[i][j]))) ++hits;
            }
            printf("%ld,%d,%.4f,%.1f,%.1f,%.1f\n", beam, t, 100.0 * hits / (k * Query.size()), Query.size() / secs,
                   static_cast<double>(QS.dist_stats()[0]), static_cast<double>(QS.visited_stats()[0]));
            fflush(stdout);
        }
    }
    return 0;
}
