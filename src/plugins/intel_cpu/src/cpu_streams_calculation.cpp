// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_streams_calculation.hpp"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <transformations/utils/utils.hpp>

#include "cpu_map_scheduling.hpp"
#include "graph.h"
#include "ie_system_conf.h"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "performance_heuristics.hpp"
#include "threading/ie_cpu_streams_info.hpp"

using namespace InferenceEngine;
using namespace ov;

namespace ov {
namespace intel_cpu {

std::vector<std::vector<int>> get_streams_info_table(const int input_streams,
                                                     const bool input_streams_changed,
                                                     const int input_threads,
                                                     const int input_infer_requests,
                                                     const int model_prefer_threads,
                                                     const std::string input_perf_hint,
                                                     const Config::LatencyThreadingMode scopeOflatencyCandidate,
                                                     const std::vector<std::vector<int>> proc_type_table) {
    std::vector<int> stream_info(CPU_STREAMS_TABLE_SIZE);
    std::vector<std::vector<int>> streams_info_table;

    auto UpdateMixStreamInfo = [&]() {
        stream_info[NUMBER_OF_STREAMS] = 0;
        int n_threads = stream_info[THREADS_PER_STREAM];
        for (int n = MAIN_CORE_PROC; n <= HYPER_THREADING_PROC; n++) {
            if (0 != proc_type_table[0][n]) {
                stream_info[PROC_TYPE] = n;
                if (n_threads <= proc_type_table[0][n]) {
                    stream_info[THREADS_PER_STREAM] = n_threads;
                    streams_info_table.push_back(stream_info);
                    break;
                } else {
                    stream_info[THREADS_PER_STREAM] = proc_type_table[0][n];
                    streams_info_table.push_back(stream_info);
                    n_threads -= proc_type_table[0][n];
                }
            }
        }
    };

    if (((input_streams_changed == false) && (input_perf_hint == CONFIG_VALUE(LATENCY)) &&
         ((scopeOflatencyCandidate == Config::LatencyThreadingMode::PER_PLATFORM) || (proc_type_table.size() == 1))) ||
        ((input_streams_changed == true) && (input_streams == 1))) {
        stream_info[NUMBER_OF_STREAMS] = 1;
        if (input_threads > 0) {
            stream_info[THREADS_PER_STREAM] = std::min(proc_type_table[0][ALL_PROC], input_threads);
            if ((stream_info[THREADS_PER_STREAM] > proc_type_table[0][MAIN_CORE_PROC]) &&
                (proc_type_table[0][MAIN_CORE_PROC] > 0) && (proc_type_table[0][EFFICIENT_CORE_PROC] > 0)) {
                stream_info[PROC_TYPE] = ALL_PROC;
                streams_info_table.push_back(stream_info);
                UpdateMixStreamInfo();
            } else if ((stream_info[THREADS_PER_STREAM] <= proc_type_table[0][MAIN_CORE_PROC]) ||
                       (proc_type_table[0][EFFICIENT_CORE_PROC] == 0)) {
                stream_info[PROC_TYPE] = MAIN_CORE_PROC;
                streams_info_table.push_back(stream_info);
            } else {
                stream_info[PROC_TYPE] = EFFICIENT_CORE_PROC;
                streams_info_table.push_back(stream_info);
            }
        } else {
            if (proc_type_table[0][ALL_PROC] == proc_type_table[0][EFFICIENT_CORE_PROC]) {
                stream_info[PROC_TYPE] = EFFICIENT_CORE_PROC;
                stream_info[THREADS_PER_STREAM] =
                    (model_prefer_threads == 0)
                        ? proc_type_table[0][EFFICIENT_CORE_PROC]
                        : std::min(proc_type_table[0][EFFICIENT_CORE_PROC], model_prefer_threads);
                streams_info_table.push_back(stream_info);
            } else if ((proc_type_table[0][EFFICIENT_CORE_PROC] > 0) &&
                       ((model_prefer_threads == 0) || (model_prefer_threads > proc_type_table[0][MAIN_CORE_PROC]))) {
                stream_info[PROC_TYPE] = ALL_PROC;
                stream_info[THREADS_PER_STREAM] =
                    (model_prefer_threads == 0 || model_prefer_threads > proc_type_table[0][MAIN_CORE_PROC])
                        ? proc_type_table[0][ALL_PROC]
                        : proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][HYPER_THREADING_PROC];
                streams_info_table.push_back(stream_info);
                UpdateMixStreamInfo();
            } else {
                stream_info[PROC_TYPE] = MAIN_CORE_PROC;
                stream_info[THREADS_PER_STREAM] =
                    proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][HYPER_THREADING_PROC];
                streams_info_table.push_back(stream_info);
            }
        }
        return streams_info_table;
    } else if ((input_streams_changed == false) && (input_perf_hint == CONFIG_VALUE(LATENCY))) {
        stream_info[PROC_TYPE] = MAIN_CORE_PROC;
        int max_per_numa_node = 0;
        int numa_node_cnt = 0;
        std::vector<int> proc_per_socket;
        proc_per_socket.resize(proc_type_table.size(), 0);
        for (long unsigned int i = 1; i < proc_type_table.size(); i++) {
            if (max_per_numa_node < proc_type_table[i][ALL_PROC]) {
                max_per_numa_node = proc_type_table[i][ALL_PROC];
                numa_node_cnt = 1;
            } else if (max_per_numa_node == proc_type_table[i][ALL_PROC]) {
                numa_node_cnt++;
            }
            proc_per_socket[proc_type_table[i][PROC_SOCKET_ID]] += proc_type_table[i][ALL_PROC];
        }
        if (scopeOflatencyCandidate == Config::LatencyThreadingMode::PER_NUMA_NODE) {
            stream_info[NUMBER_OF_STREAMS] = numa_node_cnt;
            stream_info[THREADS_PER_STREAM] = max_per_numa_node;
        } else {
            int max_per_socket = 0;
            int socket_cnt = 0;
            for (long unsigned int i = 0; i < proc_per_socket.size(); i++) {
                if (max_per_socket < proc_per_socket[i]) {
                    max_per_socket = proc_per_socket[i];
                    socket_cnt = 1;
                } else if (max_per_socket == proc_per_socket[i]) {
                    socket_cnt++;
                }
            }
            stream_info[NUMBER_OF_STREAMS] = socket_cnt;
            stream_info[THREADS_PER_STREAM] = max_per_socket;
        }
        streams_info_table.push_back(stream_info);
        return streams_info_table;
    } else {
        int n_streams = 0;
        int n_threads = 0;
        int n_threads_per_stream = 0;
        int base_type = MAIN_CORE_PROC;

        n_threads =
            (0 == input_threads) ? proc_type_table[0][ALL_PROC] : std::min(proc_type_table[0][ALL_PROC], input_threads);

        if ((input_streams_changed == true) && (input_streams > 0)) {
            base_type = (proc_type_table[0][MAIN_CORE_PROC] == 0) ? EFFICIENT_CORE_PROC : MAIN_CORE_PROC;
            n_streams = (input_infer_requests > 0) ? std::min(input_streams, input_infer_requests) : input_streams;
            if (n_streams >= n_threads) {
                n_streams = n_threads;
                n_threads_per_stream = 1;
            } else {
                n_threads_per_stream = std::min(std::max(1, n_threads / n_streams), proc_type_table[0][base_type]);
                if (proc_type_table.size() == 1) {
                    if ((n_threads_per_stream > proc_type_table[0][base_type]) &&
                        (n_threads_per_stream < proc_type_table[0][base_type] * 2)) {
                        n_threads_per_stream = proc_type_table[0][base_type];
                    } else if (n_threads_per_stream < proc_type_table[0][base_type]) {
                        n_threads_per_stream = static_cast<int>(
                            proc_type_table[0][base_type] /
                            ((proc_type_table[0][base_type] + n_threads_per_stream - 1) / n_threads_per_stream));
                    }
                }
            }
        } else {
            base_type = (proc_type_table[0][MAIN_CORE_PROC] == 0) ? EFFICIENT_CORE_PROC : MAIN_CORE_PROC;
            if (0 == model_prefer_threads) {
                int n_proc = (proc_type_table.size() == 1) ? std::min(n_threads, proc_type_table[0][base_type])
                                                           : std::min(n_threads, proc_type_table[1][base_type]);
                if (0 == n_proc % 4) {
                    n_threads_per_stream = 4;
                } else if (0 == n_proc % 5) {
                    n_threads_per_stream = 5;
                } else if (0 == n_proc % 3) {
                    n_threads_per_stream = 3;
                } else if (proc_type_table.size() == 1) {
                    n_threads_per_stream = n_proc;
                } else {
                    n_threads_per_stream = (n_proc > 16) ? 4 : std::max(1, static_cast<int>(n_proc / 4));
                }
                n_streams = static_cast<int>(n_threads / n_threads_per_stream);
                if ((input_infer_requests > 0) && (n_streams > input_infer_requests)) {
                    n_streams = input_infer_requests;
                    n_threads_per_stream =
                        std::min(static_cast<int>(n_threads / n_streams), proc_type_table[0][base_type]);
                } else {
                    while (n_streams * 2 <= n_threads_per_stream) {
                        n_threads_per_stream = static_cast<int>(n_threads_per_stream / 2);
                        n_threads_per_stream = static_cast<int>(
                            proc_type_table[0][base_type] /
                            ((proc_type_table[0][base_type] + n_threads_per_stream - 1) / n_threads_per_stream));
                        n_streams = static_cast<int>(n_threads / n_threads_per_stream);
                    }
                }
            } else if ((1 == model_prefer_threads) && (proc_type_table[0][EFFICIENT_CORE_PROC] > 0) &&
                       (proc_type_table[0][MAIN_CORE_PROC] > 0) && (n_threads > proc_type_table[0][MAIN_CORE_PROC])) {
                n_streams = (n_threads >= proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][EFFICIENT_CORE_PROC])
                                ? static_cast<int>(n_threads - proc_type_table[0][EFFICIENT_CORE_PROC] / 2)
                                : static_cast<int>(proc_type_table[0][MAIN_CORE_PROC] +
                                                   (n_threads - proc_type_table[0][MAIN_CORE_PROC]) / 2);
                n_streams = (input_infer_requests > 0) ? std::min(n_streams, input_infer_requests) : n_streams;
                n_threads_per_stream = -1;
            } else {
                n_streams = ((n_threads + model_prefer_threads - 1) / model_prefer_threads);
                n_streams = (input_infer_requests > 0) ? std::min(n_streams, input_infer_requests) : n_streams;
                n_threads_per_stream = std::min(static_cast<int>(n_threads / n_streams), proc_type_table[0][base_type]);
            }
        }

        stream_info[THREADS_PER_STREAM] = n_threads_per_stream;

        if (proc_type_table.size() == 1) {
            while (1) {
                for (int n = MAIN_CORE_PROC; n < PROC_TYPE_TABLE_SIZE; n++) {
                    if (0 != proc_type_table[0][n]) {
                        if (n_threads_per_stream == -1) {
                            stream_info[THREADS_PER_STREAM] = (n == EFFICIENT_CORE_PROC) ? 2 : 1;
                        }
                        stream_info[PROC_TYPE] = n;
                        stream_info[NUMBER_OF_STREAMS] =
                            static_cast<int>(proc_type_table[0][n] / stream_info[THREADS_PER_STREAM]);
                        if (n_streams <= stream_info[NUMBER_OF_STREAMS]) {
                            stream_info[NUMBER_OF_STREAMS] = n_streams;
                            streams_info_table.push_back(stream_info);
                            return streams_info_table;
                        } else {
                            streams_info_table.push_back(stream_info);
                            n_streams -= stream_info[NUMBER_OF_STREAMS];
                        }
                    }
                }
                if (1 == stream_info[THREADS_PER_STREAM]) {
                    return streams_info_table;
                } else {
                    stream_info[THREADS_PER_STREAM] -= 1;
                    std::vector<std::vector<int>>().swap(streams_info_table);
                }
            }
        } else {
            stream_info[NUMBER_OF_STREAMS] = n_streams;
            stream_info[PROC_TYPE] = MAIN_CORE_PROC;
            stream_info[THREADS_PER_STREAM] = n_threads_per_stream;
            streams_info_table.push_back(stream_info);
            return streams_info_table;
        }
    }
}

int get_model_prefer_threads(const int num_streams,
                             const std::vector<std::vector<int>> proc_type_table,
                             const std::shared_ptr<ngraph::Function>& ngraphFunc,
                             const InferenceEngine::IStreamsExecutor::Config streamExecutorConfig) {
    const int sockets = get_num_numa_nodes();
    auto model_prefer = 0;
    // latency
    if (num_streams <= sockets && num_streams > 0) {
        if (streamExecutorConfig._threadBindingType == IStreamsExecutor::ThreadBindingType::HYBRID_AWARE) {
            bool fp_intesive = !ov::op::util::has_op_with_type<ngraph::op::FakeQuantize>(ngraphFunc);
            const int int8_threshold = 4;  // ~relative efficiency of the VNNI-intensive code for Big vs Little cores;
            const int fp32_threshold = 2;  // ~relative efficiency of the AVX2 fp32 code for Big vs Little cores;
            // by default the latency case uses (faster) Big cores only, depending on the compute ratio
            model_prefer = proc_type_table[0][MAIN_CORE_PROC] > (proc_type_table[0][EFFICIENT_CORE_PROC] /
                                                                 (fp_intesive ? fp32_threshold : int8_threshold))
                               ? proc_type_table[0][MAIN_CORE_PROC]
                               : proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][EFFICIENT_CORE_PROC];
        }
    } else { // throughput
        const auto isa = dnnl::get_effective_cpu_isa();
        float isaSpecificThreshold = 1.0f;
        switch (isa) {
        case dnnl::cpu_isa::sse41:
            isaSpecificThreshold = 0.5f;
            break;
        case dnnl::cpu_isa::avx2:
        case dnnl::cpu_isa::avx512_core:
            isaSpecificThreshold = 1.0f;
            break;
        case dnnl::cpu_isa::avx512_core_vnni:
        case dnnl::cpu_isa::avx2_vnni:
            isaSpecificThreshold = 2.0f;
            break;
        case dnnl::cpu_isa::avx512_core_amx:
            isaSpecificThreshold = 4.0f;
            break;
        default:
            isaSpecificThreshold = 1.0f;
        }
        // the more "capable" the CPU in general, the more streams we may want to keep to keep it utilized
        const float memThresholdAssumeLimitedForISA = ov::MemBandwidthPressure::LIMITED / isaSpecificThreshold;
        const float L2_cache_size = dnnl::utils::get_cache_size(2 /*level*/, true /*per core */);
        ov::MemBandwidthPressure networkToleranceForLowCache =
            ov::MemBandwidthPressureTolerance(ngraphFunc, L2_cache_size, memThresholdAssumeLimitedForISA);
        model_prefer = IStreamsExecutor::Config::StreamMode::DEFAULT;
        if (networkToleranceForLowCache.max_mem_tolerance == ov::MemBandwidthPressure::UNKNOWN) {
            if ((networkToleranceForLowCache.ratio_compute_convs == ov::MemBandwidthPressure::ALL) ||
                (networkToleranceForLowCache.ratio_compute_deconvs == ov::MemBandwidthPressure::ALL)) {
                // all relevant layers (convs, etc) are compute-limited, the most aggressive val for #streams
                model_prefer = 1;
            }  // otherwise (no recognized layers) falling back to the default value
        } else if (networkToleranceForLowCache.max_mem_tolerance > memThresholdAssumeLimitedForISA) {
            // network is below the ISA-specific threshold
            model_prefer = 1;
        } else if (networkToleranceForLowCache.max_mem_tolerance > ov::MemBandwidthPressure::LIMITED) {
            // network is below general threshold
            model_prefer = 2;
        }
        if (model_prefer == 1 && proc_type_table[0][EFFICIENT_CORE_PROC] == 0 && sockets == 1) {
            model_prefer = 2;
        }
    }

    return model_prefer;
}

void generate_stream_info(const int streams,
                          const std::shared_ptr<ngraph::Function>& ngraphFunc,
                          Config& config,
                          int preferred_nthreads_per_stream) {
    int model_prefer_threads = preferred_nthreads_per_stream;
    InferenceEngine::IStreamsExecutor::Config& executor_config = config.streamExecutorConfig;
    auto& orig_proc_type_table = executor_config._orig_proc_type_table;
    std::vector<std::vector<int>> proc_type_table =
        apply_scheduling_core_type(config.schedulingCoreType, orig_proc_type_table);
    proc_type_table = apply_hyper_threading(config.enableHyperThreading,
                                            config.changedHyperThreading,
                                            config.perfHintsConfig.ovPerfHint,
                                            proc_type_table);
    executor_config._proc_type_table = proc_type_table;
    executor_config._cpu_pinning = get_cpu_pinning(config.enableCpuPinning,
                                                   config.changedCpuPinning,
                                                   streams,
                                                   executor_config._threadBindingType,
                                                   proc_type_table);
    if (-1 == preferred_nthreads_per_stream) {
        model_prefer_threads = get_model_prefer_threads(streams, proc_type_table, ngraphFunc, executor_config);
    }

    executor_config._streams_info_table = get_streams_info_table(executor_config._streams,
                                                                 executor_config._streams_changed,
                                                                 executor_config._threads,
                                                                 config.perfHintsConfig.ovPerfHintNumRequests,
                                                                 model_prefer_threads,
                                                                 config.perfHintsConfig.ovPerfHint,
                                                                 config.scopeOflatencyCandidate,
                                                                 proc_type_table);
}

void get_num_streams(const int streams,
                     const std::shared_ptr<ngraph::Function>& ngraphFunc,
                     Config& config) {
    InferenceEngine::IStreamsExecutor::Config& executor_config = config.streamExecutorConfig;
    std::vector<int> stream_ids;
    std::string log = "[ streams info ]";
    std::vector<std::string> core_type_str = {" Any core: ", " PCore: ", " ECore: ", " Logical core: "};

    std::vector<std::vector<int>> orig_proc_type_table = get_proc_type_table();

    executor_config._orig_proc_type_table = orig_proc_type_table;
    generate_stream_info(streams, ngraphFunc, config);

    executor_config._stream_core_ids = reserve_available_cpus(executor_config._streams_info_table);
    executor_config._threadsPerStream = executor_config._streams_info_table[0][THREADS_PER_STREAM];
    executor_config._streams = 0;
    executor_config._threads = 0;
    for (size_t i = 0; i < executor_config._streams_info_table.size(); i++) {
        executor_config._streams += executor_config._streams_info_table[i][NUMBER_OF_STREAMS];
        executor_config._threads += executor_config._streams_info_table[i][NUMBER_OF_STREAMS] *
                                    executor_config._streams_info_table[i][THREADS_PER_STREAM];
        stream_ids.insert(stream_ids.end(), executor_config._streams_info_table[i][NUMBER_OF_STREAMS], i);
        log += core_type_str[executor_config._streams_info_table[i][PROC_TYPE]] +
               std::to_string(executor_config._streams_info_table[i][NUMBER_OF_STREAMS]) + "(" +
               std::to_string(executor_config._streams_info_table[i][THREADS_PER_STREAM]) + ")";
    }
    executor_config._stream_ids = stream_ids;
    log += " Total: " + std::to_string(executor_config._streams) + "(" + std::to_string(executor_config._threads) + ")";
    DEBUG_LOG(log);
}
}  // namespace intel_cpu
}  // namespace ov
