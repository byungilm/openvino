// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#if OUTPUT_DIMS != 4
#error "dynamic_quantize_gpu_opt.cl: Unsupported output dimension"
#endif

#define VLOAD_N CAT(vload, VEC_SIZE)
#define VSTORE_N CAT(vstore, VEC_SIZE)
#define HALF_N CAT(half, VEC_SIZE)
#define CONVERT_CHAR_N CAT(convert_char, VEC_SIZE)
#define AS_TYPE_N_(type, n, x) as_##type##n(x)
#define AS_TYPE_N(type, n, x) AS_TYPE_N_(half, n, x)
#define AS_HALF_TYPE_N(x) AS_TYPE_N(half, VEC_SIZE, x)

KERNEL(dynamic_quantize_gpu_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global OUTPUT1_TYPE* output_scale)
{
    const uint bf = (uint)get_global_id(2);

    const uint sglid = get_sub_group_local_id();
    const uint group_size = (INPUT0_FEATURE_PITCH / 16);
    const uint offset_sglid = group_size * sglid;
    const uint offset = bf * INPUT0_FEATURE_PITCH + offset_sglid;

    HALF_N val;
    half max = 0.0h;
    half grp_max = 0.001h;
    unroll_for (int i = 0; i < group_size / VEC_SIZE; ++i) {
        val = fabs(AS_HALF_TYPE_N(VLOAD_N(0, input + offset + (i * VEC_SIZE))));

        #if VEC_SIZE == 8
            max = fmax(fmax(fmax(val[0], val[1]), fmax(val[2], val[3])),
                                    fmax(fmax(val[4], val[5]), fmax(val[6], val[7])));
        #else
            for (int j = 0; j < VEC_SIZE; j++) {
                max = fmax(max, val[j]);
            }
        #endif

        grp_max = fmax(grp_max, max);
    }

    half max_value = sub_group_reduce_max(grp_max);
    half scale = 127.0h / max_value;

    unroll_for (int i = 0; i < group_size; i+=VEC_SIZE) {
        val = AS_HALF_TYPE_N(VLOAD_N(0, (ushort*)input + offset + i));
        val *= scale;
        VSTORE_N(CONVERT_CHAR_N(val), 0, output + offset + i);
    }

    if (sglid == 0)
        output_scale[bf] = 1.0h / scale;
}
