// Copyright (c) 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "eltwise_kernel_b_fs_yx_fsv4.h"
#include "kernel_selector_utils.h"
#include <algorithm>
#include <string>
#include <vector>

namespace kernel_selector {
// static inline size_t GetBlockSize(const eltwise_params& params);
static std::vector<size_t> GetLocalWorkGroupSizes(std::vector<size_t> gws, const EngineInfo& info);
static inline bool InputHasFeatureBroadcast(const eltwise_params& params, const size_t op_num, int input_idx);
static inline bool OpHasFeatureBroadcast(const eltwise_params& params, const size_t op_num);

ParamsKey EltwiseKernel_b_fs_yx_fsv4::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableDifferentTypes();
    k.EnableBatching();
    k.EnableTensorPitches();
    k.EnableTensorOffset();
    k.EnableEltwiseBroadcast();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    // k.EnableAllInputLayout();
    // k.EnableAllOutputLayout();
    return k;
}

KernelsData EltwiseKernel_b_fs_yx_fsv4::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<eltwise_params>(params);
    eltwise_params& newParams = *static_cast<eltwise_params*>(kd.params.get());

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    DispatchData dispatchData = SetDefault(newParams);

    auto& kernel = kd.kernels[0];

    kernel.workGroups.global = dispatchData.gws;
    kernel.workGroups.local = dispatchData.lws;

    kernel.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, DEFAULT);
    kernel.arguments = GetArgsDesc((uint32_t)newParams.inputs.size(),
                                   false,
                                   false,
                                   GetFusedPrimitiveInputsCount(params));

    return {kd};
}

KernelsPriority EltwiseKernel_b_fs_yx_fsv4::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_1;
}

// Protected
bool EltwiseKernel_b_fs_yx_fsv4::Validate(const Params& params, const optional_params& o) const {
    if (!EltwiseKernelBase::Validate(params, o)) {
        return false;
    }

    const auto& ewParams = static_cast<const eltwise_params&>(params);

    const auto& output = ewParams.output;
    const auto count = output.PhysicalSize();

    if (count % 8 != 0)
        return false;

    for (size_t i = 0; i < ewParams.inputs.size(); i++) {
        if (ewParams.inputs[i].GetLayout() != DataLayout::b_fs_yx_fsv4) {
            return false;
        }
    }

    auto input0 = ewParams.inputs[0];

    // Check that padding before features doesn't miss-align the blocks
    // auto feature_block_size = 16;
    auto feature_block_size = 4;
    if (input0.Feature().pad.before % feature_block_size != 0 || output.Feature().pad.before % feature_block_size != 0) {
        return false;
    }

    auto compareTensors = [](const DataTensor& input0, const DataTensor& input1) -> bool {
        // Check all parameters except DataType
        auto& input0_dims = input0.GetDims();
        auto& input1_dims = input1.GetDims();
        bool same = input0.GetLayout() == input1.GetLayout() &&
                    input0.GetPaddedVal() == input1.GetPaddedVal() &&
                    input0.GetViewOffset() == input1.GetViewOffset() &&
                    input0_dims.size() == input1_dims.size();
        if (same) {
            for (size_t i = 0; i < input0_dims.size(); i++) {
                same &= input0_dims[i].v == input1_dims[i].v &&
                        input0_dims[i].pad.before == input1_dims[i].pad.before &&
                        input0_dims[i].pad.after == input1_dims[i].pad.after &&
                        input0_dims[i].pitch == input1_dims[i].pitch;
            }
        }
        return same;
    };

    for (size_t i = 1; i < ewParams.inputs.size(); i++) {
        if (ewParams.inputs[i].LogicalSize() == input0.LogicalSize() && !(compareTensors(ewParams.inputs[i], input0)))
            return false;
        if (ewParams.inputs[i].Feature().pad.before % feature_block_size != 0) {
            return false;
        }
    }

    return true;
}

JitConstants EltwiseKernel_b_fs_yx_fsv4::MakeLoadJitConstants(const eltwise_params& params, bool /*useVload8*/) const {
    JitConstants jit = {};
    std::string vload_decls;
    for (size_t op_num = 0; op_num < params.operations.size(); op_num++) {
        const std::string op_num_str = std::to_string(op_num);
        const auto &ew = params.operations[op_num];
        for (size_t input_idx = 0; input_idx < ew.inputs.size(); input_idx++) {
            const auto &input = ew.inputs[input_idx];
            const std::string name = "INPUT_" + op_num_str + "_" + std::to_string(input_idx);

            switch (input.mode) {
                case EltwiseInputMode::SCALAR:
                    jit.AddConstant(MakeJitConstant(name, input.scalar));
                    break;
                case EltwiseInputMode::INPUT_BUFFER:
                {
                    const std::string idx_order = "INPUT" + std::to_string(input.index) + "_IDX_ORDER";
                    // jit.AddConstant(MakeJitConstant(idx_order, "b, f_block*16, y, x"));
                    jit.AddConstant(MakeJitConstant(idx_order, "b, f_block*4, y, x"));

                    if (params.inputs[input.index].LogicalSize() == 1) {
                        jit.AddConstant(MakeJitConstant(name,
                                                        "input" + std::to_string(input.index) +
                                                        "[0]"));
                    } else {
                        bool feature_broadcasting = (params.inputs[input_idx].Feature().v == 1 && params.output.Feature().v != 1);

                        // if ((params.inputs[input.index].LogicalSize() == params.output.Feature().v &&
                        //     params.inputs[input.index].LogicalSize() == params.inputs[input.index].Feature().v) || true)
                        {
                            // const std::string block_read_str = "TO_TYPE(MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 1), BLOCK_READN(INPUT" +
                            //                                         std::to_string(input.index) + "_TYPE, 1, "
                            //                                         "input" + std::to_string(input.index) + ", " +
                            //                                         "GET_INDEX(INPUT, " + std::to_string(input.index) + ", " + idx_order + ")))";

                            if (feature_broadcasting) {
                                const std::string broadcast_name = "DO_FEATURE_BROADCAST" + std::to_string(op_num) + "_" + std::to_string(input_idx);
                                // std::string broadcast_value = "\\\n\tMAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 1) tmp_b" + std::to_string(op_num) +
                                //                             " = " + block_read_str + ";" +
                                //                             "\\\n\ttmp_b" + std::to_string(op_num) +
                                //                             " = sub_group_broadcast(tmp_b" + std::to_string(op_num) + ", 0);";
                                std::string broadcast_value = "\\\n\tMAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 4) tmp_b" + std::to_string(op_num) +
                                //                            " = (" + toCLType(params.inputs[input.index].GetDType()) + "4)(input" + std::to_string(input.index) +
                                                            " = " "(MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 4))"+"(input" + std::to_string(input.index) +
                                //                            " = " + "TO_TYPE(MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, " + std::to_string(vec_size) + "), &input" + std::to_string(input.index) +
                                                            "[GET_INDEX(INPUT, " + std::to_string(input.index) + ", " + idx_order + ")]);";

                                jit.AddConstant(MakeJitConstant(broadcast_name, broadcast_value));
                                jit.AddConstant(MakeJitConstant(name, "tmp_b" + std::to_string(op_num)));
                            } else {
                                const std::string vload_name = "DO_VLOAD" + std::to_string(op_num) + "_" + std::to_string(input_idx);
                                const std::string vload_value = "\\\n\tMAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 4) tmp_a" + std::to_string(op_num) + "_" + std::to_string(input_idx) +
                                //                                " = vload4(0, &input" + std::to_string(input.index) +
                                                                " = TO_TYPE(MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, " + std::to_string(vec_size) + "), vload4(0, &input" + std::to_string(input.index) +
                                                                "[GET_INDEX(INPUT," + std::to_string(input.index) + ", " + idx_order + ")]));";

                                jit.AddConstant(MakeJitConstant(vload_name, vload_value));
                                jit.AddConstant(MakeJitConstant(name, "tmp_a" + std::to_string(op_num) + "_" + std::to_string(input_idx)));
                            }
                        }
                    }
                    break;
                }
                case EltwiseInputMode::OUTPUT_BUFFER:
                    jit.AddConstant(MakeJitConstant(name, "output[off]"));
                    break;
                case EltwiseInputMode::UNORDERED_ACCESS_INPUT_BUFFER:
                    jit.AddConstant(MakeJitConstant(
                            name,
                            "input" + std::to_string(input.index) + "[(size_t)tmp" + std::to_string(input.tmpIndex) + "]"));
                    break;
                case EltwiseInputMode::INTERMEDIATE_RESULTS_INDEX:
                    jit.AddConstant(MakeJitConstant(name, "tmp" + std::to_string(input.tmpIndex)));
                    break;
                default:
                    break;
            }
        }
    }

    return jit;
}

JitConstants EltwiseKernel_b_fs_yx_fsv4::GetJitConstants(const eltwise_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    bool useVload8 = false;

    // auto blockSize = GetBlockSize(params);
    auto blockSize = vec_size;
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    jit.AddConstant(MakeJitConstant("BLOCK_SIZE", blockSize));
    jit.AddConstant(MakeJitConstant("BLOCKS_COUNT", CeilDiv(params.output.X().v, blockSize)));

    jit.Merge(MakeInputDeclsJitConstants(params, useVload8));
    jit.Merge(MakeLoadJitConstants(params, useVload8));
    jit.Merge(GetOperationsJitConstants(params, useVload8, vec_size));

    std::string do_eltwise;
    auto& operations = params.operations;
    for (size_t op_num = 0; op_num < operations.size(); op_num++) {
        for (size_t input_idx = 0; input_idx < params.inputs.size(); input_idx++) {
            if (InputHasFeatureBroadcast(params, op_num, input_idx)) {
                do_eltwise += "\\\n\tDO_FEATURE_BROADCAST" + std::to_string(op_num) + "_" + std::to_string(input_idx) + ";";
            } else {
                do_eltwise += "\\\n\tDO_VLOAD" + std::to_string(op_num) + "_" + std::to_string(input_idx) + ";";
            }
        }
        do_eltwise += "\\\n\tOPERATION" + std::to_string(op_num) + ";";
    }

    do_eltwise += "\\\n\tres = tmp" + std::to_string(operations.size() - 1) + ";";

    jit.AddConstant(MakeJitConstant("DO_ELTWISE", do_eltwise));

    if (params.layoutBased || params.int8_quantization || params.broadcast) {
        jit.Merge(GetTensorFriendlyWorkGroupsJit(params.output));
    }

    if (!params.stride.empty()) {
        jit.AddConstant(MakeJitConstant("INPUT_STRIDED", 1));
    }

    jit.Merge(MakeActivationJitConstants(params.activations, params.output.GetDType(), "_TYPED"));

    // if (params.output.Feature().v % 16 != 0)
    //    jit.AddConstant(MakeJitConstant("LEFTOVERS", params.output.Feature().v % 16));
    if (params.output.Feature().v % 4 != 0)
        jit.AddConstant(MakeJitConstant("LEFTOVERS", params.output.Feature().v % 4));

    if (!params.fused_ops.empty()) {
        kernel_selector::Datatype input_dt = GetAccumulatorType(params);
        std::vector<std::string> idx_order = {"b", "f_block*4", "y", "x"};
        FusedOpsConfiguration conf = {"", idx_order, "res", input_dt, (size_t)vec_size, LoadType::LT_ALIGNED_READ};
        conf.SetVectorAxis(Tensor::DataChannelName::FEATURE);

        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    jit.AddConstant(MakeJitConstant("ELTWISE_BROADCAST", params.broadcast));
    jit.AddConstant(MakeJitConstant("QUANTIZATION_TERM", params.int8_quantization));
    jit.AddConstant(MakeJitConstant("VEC_SIZE", vec_size));

    printf(">> Eltwise Kernel Opt fsv4 DataLayout: (input: %d, output: %d)\n", (int)(params.inputs[0].GetLayout()), (int)(params.output.GetLayout()));
    // for (auto& w : jit.GetDefinitions())
    //     printf(" %s : [ %s ]\n", w.first.c_str(), w.second.c_str());

    return jit;
}

EltwiseKernelBase::DispatchData EltwiseKernel_b_fs_yx_fsv4::SetDefault(const eltwise_params& params) const {
    DispatchData dispatchData;

    // dispatchData.gws[0] = Align(params.output.Feature().v, 16);
    // dispatchData.gws[1] = CeilDiv(params.output.X().v, GetBlockSize(params)) * params.output.Y().v;
    dispatchData.gws[0] = params.output.X().v * params.output.Y().v;
    // dispatchData.gws[1] = CeilDiv(Align(params.output.Feature().v, 4), 4);
    dispatchData.gws[1] = CeilDiv(params.output.Feature().v, 4);
    dispatchData.gws[2] = params.output.Batch().v;

    // dispatchData.lws[0] = 16;
    // dispatchData.lws[1] = 16;
    dispatchData.lws= GetLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    dispatchData.lws[1] = 1;
    // while (dispatchData.lws[1] > 1) {
    //     if (dispatchData.gws[1] % dispatchData.lws[1] == 0)
    //         break;
    //     dispatchData.lws[1]--;
    // }
    dispatchData.lws[2] = 1;

    printf("b_fs_yx_fsv4 gws[%lu, %lu, %lu] lws[%lu, %lu, %lu]\n", dispatchData.gws[0], dispatchData.gws[1], dispatchData.gws[2],
                                                dispatchData.lws[0], dispatchData.lws[1], dispatchData.lws[2]);
    return dispatchData;
}

// Local
static std::vector<size_t> GetLocalWorkGroupSizes(std::vector<size_t> gws, const EngineInfo& info) {
    const size_t lws_max = info.maxWorkGroupSize;
    const size_t optimal_lws_values[] = {256, 227, 224, 192, 160, 128, 96, 64, 32, 16, 8, 7, 6, 5, 4, 2, 1};
    size_t total_lws = 1;
    std::vector<size_t> lws;
    for (size_t i = 0; i < gws.size(); ++i) {
        auto rest_lws = lws_max / total_lws;
        size_t lws_idx = 0;
        while (rest_lws < optimal_lws_values[lws_idx]) lws_idx++;

        while (gws[i] % optimal_lws_values[lws_idx]) lws_idx++;

        lws.push_back(optimal_lws_values[lws_idx]);
        total_lws *= optimal_lws_values[lws_idx];
    }

    return lws;
}

// static inline size_t GetBlockSize(const eltwise_params& params) {
//     for (size_t i = 0; i < params.inputs.size(); i++) {
//         if (params.inputs[i].X().v == 1 && params.inputs[i].LogicalSize() != 1) {
//             return 1;
//         }
//     }

//     // size_t optimal_bs_values[] = {8, 4, 2, 1};
//     size_t optimal_bs_values[] = {1};

//     for (auto bs : optimal_bs_values) {
//         if ((params.output.X().v) % bs == 0) {
//             return bs;
//         }
//     }

//     return 1;
// }

static inline bool InputHasFeatureBroadcast(const eltwise_params& params, const size_t op_num, int input_idx) {
    const auto &ew = params.operations[op_num];

    const auto &input = ew.inputs[input_idx];
    if (input.mode == EltwiseInputMode::INPUT_BUFFER) {
        if (params.inputs[input_idx].LogicalSize() != 1
            && params.inputs[input_idx].Feature().v == 1
            && params.output.Feature().v != 1) {
                return true;
            }
    }

    return false;
}

static inline bool OpHasFeatureBroadcast(const eltwise_params& params, const size_t op_num) {
    const auto &ew = params.operations[op_num];

    for (size_t input_idx = 0; input_idx < ew.inputs.size(); input_idx++) {
        const auto &input = ew.inputs[input_idx];
        if (input.mode == EltwiseInputMode::INPUT_BUFFER) {
            if (params.inputs[input_idx].LogicalSize() != 1
                && params.inputs[input_idx].Feature().v == 1
                && params.output.Feature().v != 1) {
                    return true;
                }
        }
    }

    return false;
}
}  // namespace kernel_selector
