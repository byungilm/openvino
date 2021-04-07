// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/include_all.cl"
#include "include/common.cl"
#include "include/data_types.cl"

// KERNEL(eltwise_b_fs_yx_fsv4)(INPUTS_DECLS
//                            __global OUTPUT_TYPE* output)
// {
//     const uint global_id = get_global_id(0);
//     VLOAD_DECLS
//     MAKE_VECTOR_TYPE(OUTPUT_TYPE, 4) res;
//     DO_ELTWISE
//     res = ACTIVATION(res, ACTIVATION_PARAMS);
//     vstore4(res, global_id, output);
// }


#define unroll_for  __attribute__((opencl_unroll_hint())) for

#define OUTPUT_TYPE_BLOCK               MAKE_VECTOR_TYPE(OUTPUT_TYPE, BLOCK_SIZE)
#define TO_TYPE(type, val)              CAT(convert_, type)(val)

#if BLOCK_SIZE != 1
    #define READ_FUNC(ptr, offset) CAT(DT_INPUT_BLOCK_READ, BLOCK_SIZE)(ptr, offset)
    #define WRITE_FUNC(ptr, offset, val) CAT(DT_OUTPUT_BLOCK_WRITE, BLOCK_SIZE)(ptr, offset, val)
#else
    #define READ_FUNC(ptr, offset) DT_INPUT_BLOCK_READ(ptr, offset)
    #define WRITE_FUNC(ptr, offset, val) DT_OUTPUT_BLOCK_WRITE(ptr, offset, val)
#endif

#if ELTWISE_BROADCAST
    #define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX_SAFE)(idx_order)
#else
    #define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX)(idx_order)
#endif

KERNEL(eltwise_b_fs_yx_fsv4)(INPUTS_DECLS
                              __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
, FUSED_OPS_DECLS
#endif
)
{
    const uint y = (uint)get_global_id(0) / OUTPUT_SIZE_X;
    const uint x = (uint)get_global_id(0) % OUTPUT_SIZE_X;
    const uint f_block = get_group_id(1);
    const uint b = get_global_id(2);

    MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, VEC_SIZE) res;

    DO_ELTWISE
    int temp = 0;
// #if HAS_FUSED_OPS
//     FUSED_OPS;
//     OUTPUT_TYPE_BLOCK out = TO_TYPE(MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE), FUSED_OPS_RESULT);
// #else
//     OUTPUT_TYPE_BLOCK out = ACTIVATION_TYPED(TO_TYPE(MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE), res), ACTIVATION_PARAMS_TYPED);
// #endif
#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE_BLOCK out = TO_TYPE(MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE), FUSED_OPS_RESULT);
    temp = 1;
#else
#if QUANTIZATION_TERM && !OUTPUT_IS_FP
    OUTPUT_TYPE_BLOCK out;
    for (uint fp = 0; fp < VEC_SIZE; fp++) {
        out[fp] = TO_OUTPUT_TYPE_SAT(ACTIVATION(res[fp], ACTIVATION_PARAMS));
    }
#else
    OUTPUT_TYPE_BLOCK out = ACTIVATION_TYPED(TO_TYPE(MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE), res), ACTIVATION_PARAMS_TYPED);
#endif
#endif

#ifdef LEFTOVERS
    if ((f_block*VEC_SIZE + VEC_SIZE) > OUTPUT_FEATURE_NUM) {
        for (uint fp = OUTPUT_FEATURE_NUM % VEC_SIZE; fp < VEC_SIZE; fp++) {
            // output[OUTPUT_GET_INDEX(b, (f_block*VEC_SIZE)+fp, y, x)] = out[fp];
            out[fp] = OUTPUT_VAL_ZERO;
        }
    }
#endif

    vstore4(out, 0, &output[OUTPUT_GET_INDEX(b, (f_block*VEC_SIZE), y, x)]);

    if (f_block == 1) {
        // printf("f_block(%u) res[%d %d %d %d] out[%d %d %d %d]\n", f_block, (int)res[0], (int)res[1], (int)res[2], (int)res[3], (int)out[0], (int)out[1], (int)out[2], (int)out[3]);
        printf("FUSED(%d) f_block(%u) res[%.4f %.4f %.4f %.4f] out[%.4f %.4f %.4f %.4f]\n", temp, f_block, (float)res[0], (float)res[1], (float)res[2], (float)res[3], (float)out[0], (float)out[1], (float)out[2], (float)out[3]);
    }
}
