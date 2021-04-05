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

#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE_BLOCK out = TO_TYPE(MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE), FUSED_OPS_RESULT);
#else
    OUTPUT_TYPE_BLOCK out = ACTIVATION_TYPED(TO_TYPE(MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE), res), ACTIVATION_PARAMS_TYPED);
#endif

    // printf("f_block(%u) res[%.4f %.4f %.4f %.4f]\n", f_block, res[0], res[1], res[2], res[3]);

#ifdef LEFTOVERS
    if ((f_block*VEC_SIZE + VEC_SIZE) > OUTPUT_FEATURE_NUM) {
        for (uint fp = 0; fp < VEC_SIZE; fp++) {
            if (fp < OUTPUT_FEATURE_NUM % VEC_SIZE) {
                output[OUTPUT_GET_INDEX(b, (f_block*VEC_SIZE)+fp, y, x)] = out[fp];
            }
        }
    } else
#endif
    {
        // WRITE_FUNC(output, output_offset, out);
        vstore4(out, 0, &output[OUTPUT_GET_INDEX(b, (f_block*VEC_SIZE), y, x)]);
    }
}
