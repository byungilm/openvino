// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

// JIT Parameters:
// SIMD         - sub-group size/simd width, one of {8, 16};
// TILE_B       - number of batches processed by each work-item;
// TILE_OFM     - number of output features calculated by work-item, one of {1, 2, 4, 8};
// TILE_IFM     - number of input features loaded from input by work-item, one of {1, 2, 4, 8};
// TILE_K       - number of input features loaded from weights, one of {1, 2, 4, 8};
// TILE_K_OFM   - must be equal to TILE_OFM * TILE_K and less or equal to 8;
// DISPATCH_FSV - output coordinates for each sub-group are calculated from linearized coordinates
// DISPATCH_BSV   as if they laid in bs_fs_bsv_fsv format, these macros describe fsv and bsv factors;

// Verify JIT parameters.
#if SIMD != 8 && SIMD != 16
#   error "fully_connected_gpu_bf_tiled.cl - SIMD must be one of {8, 16}"
#endif

#if TILE_OFM != 1 && TILE_OFM != 2 && TILE_OFM != 4 && TILE_OFM != 8
#   error "fully_connected_gpu_bf_tiled.cl - TILE_OFM must be one of {1, 2, 4, 8}"
#endif

#if TILE_IFM != 1 && TILE_IFM != 2 && TILE_IFM != 4 && TILE_IFM != 8
#   error "fully_connected_gpu_bf_tiled.cl - TILE_IFM must be one of {1, 2, 4, 8}"
#endif

#if TILE_K != 1 && TILE_K != 2 && TILE_K != 4 && TILE_K != 8
#   error "fully_connected_gpu_bf_tiled.cl - TILE_K must be one of {1, 2, 4, 8}"
#endif

#if TILE_K_OFM != (TILE_K * TILE_OFM) || TILE_K_OFM > 8
#   error "fully_connected_gpu_bf_tiled.cl - TILE_K_OFM must be equal to TILE_K * TILE_OFM and at most 8"
#endif

#if COMPRESSED_WEIGHTS_INT4
#   if TILE_K_OFM != TILE_K_OFM_PACKED * 2
#       error "fully_connected_gpu_bf_tiled.cl - TILE_K_OFM must be divisible by 2 for 4-bit compressed case"
#   endif
#   if FILTER_LAYOUT_OS_IS_YX_OSV32_ISV2 && TILE_K != 4 && TILE_K != 2 && TILE_K != 1
#       error "fully_connected_gpu_bf_tiled.cl - TILE_K must be one of {1, 2, 4}"
#   endif
#endif
#if TILE_K == 4 && COMPRESSED_WEIGHTS_INT4 && FILTER_LAYOUT_OS_IS_YX_OSV32_ISV2
// Data stored in memory : f0k0k1|f16k0k1|f0k2k3|f16k2k3
// => unpack as f0k0k1|f0k2k3|f16k0k1|f16k2k3 so that the weight access order is preserved
#define UNPACK_INT4 UNPACK_INT4x2_OSV32_ISV2
#define UNPACK_MIXED_INT4 UNPACK_INT4x2_OSV32_ISV2
#else
#define UNPACK_INT4 UNPACK_INT4x2
#define UNPACK_MIXED_INT4 UNPACK_MIXED_INT4x2
#endif

// Macros for vectorized types.
#define INPUT_VEC_TYPE             MAKE_VECTOR_TYPE(INPUT0_TYPE, TILE_IFM)
#define ACCUMULATOR_VEC_TYPE       MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, TILE_OFM)
#define FILTER_VEC_TYPE            MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, TILE_K_OFM)
#define FILTER_PACKED_VEC_TYPE     MAKE_VECTOR_TYPE(FILTER_TYPE, TILE_K_OFM_PACKED)
#define BIAS_VEC_TYPE              MAKE_VECTOR_TYPE(BIAS_TYPE, TILE_OFM)
#define OUTPUT_VEC_TYPE            MAKE_VECTOR_TYPE(OUTPUT_TYPE, TILE_OFM)
#define ACTIVATION_VEC_TYPE        MAKE_VECTOR_TYPE(ACTIVATION_TYPE, TILE_OFM)
#define TO_OUTPUT_VEC_TYPE(x)      CAT(convert_, OUTPUT_VEC_TYPE)(x)
#define TO_ACTIVATION_VEC_TYPE(x)  CAT(convert_, ACTIVATION_VEC_TYPE)(x)
#define TO_FILTER_VEC_TYPE(x)      CAT(convert_, FILTER_VEC_TYPE)(x)
#define TO_ACCUMULATOR_VEC_TYPE(x) CAT(convert_, ACCUMULATOR_VEC_TYPE)(x)

#define INPUT_BLOCK_READ(ptr, offset)        BLOCK_READN(INPUT0_TYPE, TILE_IFM, ptr, offset)
#define FILTER_BLOCK_READ(ptr, offset)       BLOCK_READN(FILTER_TYPE, TILE_K_OFM_PACKED, ptr, offset)
#define BIAS_BLOCK_READ(ptr, offset)         BLOCK_READN(BIAS_TYPE, TILE_OFM, ptr, offset)
#define OUTPUT_BLOCK_WRITE(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, TILE_OFM, ptr, offset, val)

#define SLM_FILTER_VEC          MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, TILE_OFM)
#define SLM_FILTER_PACKED_VEC   MAKE_VECTOR_TYPE(FILTER_TYPE, FILTER_LOAD_BLOCK_SIZE)
#define SLM_FILTER_UNPACKED_VEC MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, FILTER_ELEMENTS_PER_LOAD)

// Check alignment restrictions for using block writes on output.
#define USE_BLOCK_WRITE ((OUTPUT_TYPE_SIZE * TILE_OUT_B_PITCH) % 16 == 0 && (OUTPUT_TYPE_SIZE * OUTPUT_OFFSET) % 16 == 0)


#if !REALIGN_FP16_OFFSET
#   if OUTPUT_3D
#       define MAIN_LOOP_ELEMENTS_COUNT  INPUT0_SIZE_Y
#   else
#       define MAIN_LOOP_ELEMENTS_COUNT  INPUT0_ELEMENTS_COUNT
#   endif
#else
// For REALIGN_FP16_OFFSET one feature is processed separately before entering main loop to correct alignment.
#   if OUTPUT_3D
#       define MAIN_LOOP_ELEMENTS_COUNT  (INPUT0_SIZE_Y - 1)
#   else
#       define MAIN_LOOP_ELEMENTS_COUNT  (INPUT0_ELEMENTS_COUNT - 1)
#   endif
#endif

#if OUTPUT_3D
#   define INPUT_ELEMENTS_COUNT INPUT0_SIZE_Y
#else
#   define INPUT_ELEMENTS_COUNT INPUT0_ELEMENTS_COUNT
#endif

#if IS_DYNAMIC && COMPRESSED_WEIGHTS_INT4
#pragma disable_includes_optimization
#define FORCED_TILE_B 1
#include "include/fully_connected_gpu_bf_tiled_common.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 2
#include "include/fully_connected_gpu_bf_tiled_common.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 3
#include "include/fully_connected_gpu_bf_tiled_common.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 4
#include "include/fully_connected_gpu_bf_tiled_common.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 5
#include "include/fully_connected_gpu_bf_tiled_common.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 6
#include "include/fully_connected_gpu_bf_tiled_common.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 7
#include "include/fully_connected_gpu_bf_tiled_common.cl"
#undef FORCED_TILE_B
#pragma enable_includes_optimization
#endif

// Dyc Quantize
#define INPUT_LOAD_SIZE                     4
#define DQ_TYPE                             char
#define PACKED_DQ_TYPE                      int
#define DQ_VEC_TYPE                         MAKE_VECTOR_TYPE(DQ_TYPE, TILE_IFM)
#define DQ_SLM_FILTER_VEC                   MAKE_VECTOR_TYPE(DQ_TYPE, 4)
#define DQ_SLM_FILTER_PACKED_VEC            MAKE_VECTOR_TYPE(FILTER_TYPE, FILTER_LOAD_BLOCK_SIZE)
#define DQ_SLM_FILTER_UNPACKED_VEC          MAKE_VECTOR_TYPE(DQ_TYPE, FILTER_ELEMENTS_PER_LOAD)
#define DQ_FILTER_VEC_TYPE                  MAKE_VECTOR_TYPE(DQ_TYPE, TILE_K_OFM)

#define TO_DQ_TYPE(x)                       CAT(CAT(convert_, DQ_TYPE),_sat)(x)
#define TO_DQ_VEC_TYPE(x)                   CAT(convert_, DQ_VEC_TYPE)(x)
#define TO_DQ_SLM_FILTER_UNPACKED_VEC(x)  CAT(convert_, DQ_SLM_FILTER_UNPACKED_VEC)(x)
#define TO_DQ_FILTER_VEC_TYPE(x)            CAT(convert_, DQ_FILTER_VEC_TYPE)(x)

#define AS_TYPE_N_(type, n, x)  as_##type##n(x)
#define AS_TYPE_N(type, n, x)   AS_TYPE_N_(type, n, x)
#define AS_DQ_TYPE_4(x)         AS_TYPE_N(DQ_TYPE, INPUT_LOAD_SIZE, x)


inline void FUNC(fc_bf_tiled_kernel_dynamic_quantized)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
#if DECOMPRESSION_SCALE_TERM
    const __global DECOMPRESSION_SCALE_TYPE* decompression_scale,
#endif
#if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
    const __global DECOMPRESSION_ZP_TYPE* decompression_zp,
#endif
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if USE_SLM
    // , __local ACCUMULATOR_TYPE* wei_local_mem
    , __local int* wei_local_mem
#endif
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
#if USE_SLM
    uint gid = (uint)get_group_id(0);
    uint local_id = (uint)get_local_id(2);
#else
    uint gid = (uint)get_group_id(0);
#endif

    uint bgid = (uint)get_group_id(2);
    uint sglid = (uint)get_sub_group_local_id();

    // Dispatch as bs_fs_bsv_fsv, where bsv = DISPATCH_BSV and fsv = DISPATCH_FSV.
    // This allows more fine grained control over dispatch order than using work-groups and
    // avoids requirement of threads being available for whole work-group.
    // It could hovewer have some drawbacks like not providing physical locality or not using
    // full dispatch pipeline.
    uint feature_mini_block = gid % DISPATCH_FSV;
    uint batch_mini_block = gid / DISPATCH_FSV % DISPATCH_BSV;
    uint feature_mega_block = gid / (DISPATCH_FSV * DISPATCH_BSV) % (CEIL_DIV(TILE_OUT_F_NUM, TILE_OFM * SIMD) / DISPATCH_FSV);
    uint batch_mega_block = gid / (DISPATCH_FSV * DISPATCH_BSV * CEIL_DIV(TILE_OUT_F_NUM, TILE_OFM * SIMD) / DISPATCH_FSV);

#if USE_SLM
    uint out_f = gid * (TILE_OFM * SIMD);
    uint out_b = LWS_BATCHES * TILE_B * bgid + local_id * TILE_B;
#else
    FILTER_VEC_TYPE wei = 0;
    uint out_f = (feature_mega_block * DISPATCH_FSV + feature_mini_block) * (TILE_OFM * SIMD);
    uint out_b = ((batch_mega_block * DISPATCH_BSV + batch_mini_block) * TILE_B);
#endif

#if OUTPUT_3D
    uint out_b0 = out_b / OUTPUT_FEATURE_NUM;
    uint out_b1 = out_b % OUTPUT_FEATURE_NUM;
    uint input_offset = out_b0 * INPUT0_BATCH_PITCH + out_b1 * INPUT0_FEATURE_PITCH + INPUT0_OFFSET;
#else
    uint input_offset = out_b * TILE_IN_B_PITCH + INPUT0_OFFSET;
#endif

#if COMPRESSED_WEIGHTS_INT4
    uint weights_offset = out_f * (INPUT_ELEMENTS_COUNT / 2);
#else
    uint weights_offset = out_f * INPUT_ELEMENTS_COUNT;
#endif

    ACCUMULATOR_VEC_TYPE    acc[TILE_B] = { };

    #if USE_SLM && COMPRESSED_WEIGHTS_INT4
        // Dyn Quan
        MAKE_VECTOR_TYPE(INPUT0_TYPE, INPUT_LOAD_SIZE)  tiled_input_0[HALF_TILE_B] = { };   // Load 4 linear inputs for packing
        PACKED_DQ_TYPE                                  packed_in_0[HALF_TILE_B] = { };     // Packing char4 inputs to 1 integer
    #else
        INPUT_VEC_TYPE       in_0[TILE_B] = { };
    #endif

#if COMPRESSED_WEIGHTS && DECOMPRESSION_SCALE_GROUPS_NUM == 1
    #if DECOMPRESSION_SCALE_LENGTH > 1 && DECOMPRESSION_SCALE_LENGTH % (TILE_OFM * SIMD) == 0
        ACCUMULATOR_VEC_TYPE d_scale = TO_ACCUMULATOR_VEC_TYPE(BLOCK_READN(DECOMPRESSION_SCALE_TYPE, TILE_OFM, decompression_scale, out_f));
    #elif DECOMPRESSION_SCALE_LENGTH > 1 && DECOMPRESSION_SCALE_LENGTH % (TILE_OFM * SIMD) != 0
        ACCUMULATOR_VEC_TYPE d_scale = 0;
        unroll_for(uint of = 0; of < TILE_OFM; ++of) {
            uint offset = out_f + of*SIMD + get_sub_group_local_id();
            if (offset < DECOMPRESSION_SCALE_LENGTH)
                ((ACCUMULATOR_TYPE*)(&d_scale))[of] = decompression_scale[offset];
        }
    #else
        ACCUMULATOR_VEC_TYPE d_scale = decompression_scale[0];
    #endif

    ACCUMULATOR_TYPE* d_scales = (ACCUMULATOR_TYPE*)(&d_scale);
#endif

#if COMPRESSED_WEIGHTS && DECOMPRESSION_ZP_TERM && DECOMPRESSION_ZP_GROUPS_NUM == 1 && !DECOMPRESSION_ZP_SCALAR
    #if DECOMPRESSION_ZP_LENGTH > 1 && DECOMPRESSION_ZP_LENGTH % (TILE_OFM * SIMD) == 0
        ACCUMULATOR_VEC_TYPE d_zp = TO_ACCUMULATOR_VEC_TYPE(BLOCK_READN(DECOMPRESSION_ZP_TYPE, TILE_OFM, decompression_zp, out_f));
    #elif DECOMPRESSION_ZP_LENGTH > 1 && DECOMPRESSION_ZP_LENGTH % (TILE_OFM * SIMD) != 0
        ACCUMULATOR_VEC_TYPE d_zp = 0;
        unroll_for(uint of = 0; of < TILE_OFM; ++of) {
            uint offset = out_f + of*SIMD + get_sub_group_local_id();
            if (offset < DECOMPRESSION_ZP_LENGTH)
                ((ACCUMULATOR_TYPE*)(&d_zp))[of] = decompression_zp[offset];
        }
    #else
        ACCUMULATOR_VEC_TYPE d_zp = decompression_zp[0];
    #endif
    ACCUMULATOR_TYPE* d_zps = (ACCUMULATOR_TYPE*)(&d_zp);
#endif

#if REALIGN_FP16_OFFSET
    // For fp16 we need to ensure that all block reads are aligned to 4 byte (2 words) boundary.
    // To do this solve first input feature separately.
    {
        INPUT0_TYPE tmp_input = input[input_offset + get_sub_group_local_id() % TILE_B * TILE_IN_B_PITCH];
        ACCUMULATOR_VEC_TYPE tmp_wei = TO_ACCUMULATOR_VEC_TYPE(BLOCK_READN(FILTER_TYPE, TILE_OFM, weights, weights_offset));
        #if COMPRESSED_WEIGHTS
            tmp_wei = (tmp_wei - d_zp) * d_scale;
        #endif
        unroll_for(uint bi = 0; bi < TILE_B; ++bi) {
            acc[bi] = _sub_group_shuffle(tmp_input, bi) * tmp_wei;
        }

        weights_offset += TILE_OFM * SIMD;
        input_offset += 1;
    }
#endif
    // =====================================================================================================================================
    // Main computation loop
    uint iterations = MAIN_LOOP_ELEMENTS_COUNT / (TILE_IFM * SIMD);
    uint idx_sglid = (sglid * TILE_K) % 32;       // same index for sglid 0~7 : to tile_k direction
    uint batch_sglid = (sglid * TILE_K) / 32;     // 0 to 1 : to batch direction

    __attribute__((opencl_unroll_hint(1)))
    for (uint ni = 0; ni < iterations; ++ni) {
        #if USE_SLM && COMPRESSED_WEIGHTS_INT4
            // Packing : Get 4(B)x4(K) integer vector (packing to 4x1 vector)
            uint in_offset = input_offset + (idx_sglid + batch_sglid * TILE_IN_B_PITCH);
            for (uint bi = 0; bi < HALF_TILE_B; ++bi) {
                tiled_input_0[bi] = vload4(0, &input[in_offset]);

                // Next batch
                in_offset += (TILE_IN_B_PITCH * 2);
            }

            input_offset += TILE_IFM * SIMD;
        #else
            #define LOAD_IN_0(bi) do {                                  \
                    in_0[bi] = INPUT_BLOCK_READ(input, input_offset);   \
                    input_offset += TILE_IN_B_PITCH;                    \
                } while (false)

            CONST_LOOP(TILE_B, LOAD_IN_0);
            #undef LOAD_IN_0
            input_offset += TILE_IFM * SIMD - TILE_IN_B_PITCH * TILE_B;
        #endif

        #if USE_SLM && DYNAMIC_QUANTIZE
            MAKE_VECTOR_TYPE(INPUT0_TYPE, HALF_TILE_B) de_quantize_scale[2] = { 0 };
            MAKE_VECTOR_TYPE(int, TILE_B) acc_tmp[TILE_OFM] = { 0 };
        #else
            ACCUMULATOR_VEC_TYPE acc_tmp[TILE_B] = { };
        #endif

        #if USE_SLM && COMPRESSED_WEIGHTS_INT4
            #if TILE_OFM != 2
            #error "FC bf_tiled kernel: can't use SLM optimization with TILE_OFM != 2"
            #endif

            // Skip first barrier synchronization if there is only single outer loop iteration.
            #if MAIN_LOOP_ELEMENTS_COUNT / (TILE_IFM * SIMD) > 1
                barrier(CLK_LOCAL_MEM_FENCE);
            #endif

            #if DYNAMIC_QUANTIZE
                // Quantizing for loaded input using max value
                #if 1
                    {
                        INPUT0_TYPE                                max[2][HALF_TILE_B] = { 0 };
                        // MAKE_VECTOR_TYPE(INPUT0_TYPE, HALF_TILE_B) quantize_scale = 1;
                        MAKE_VECTOR_TYPE(INPUT0_TYPE, HALF_TILE_B) dq_max_input[2] = { 0 };
                        MAKE_VECTOR_TYPE(INPUT0_TYPE, HALF_TILE_B) quan = 128;
                        unroll_for (uint bi = 0; bi < HALF_TILE_B; ++bi) {
                            max[batch_sglid][bi] = fmax(fmax(fabs(tiled_input_0[bi][0]), fabs(tiled_input_0[bi][1])), fmax(fabs(tiled_input_0[bi][2]), fabs(tiled_input_0[bi][3])));
                        }
                        unroll_for (uint bi = 0; bi < HALF_TILE_B; ++bi) {
                            dq_max_input[0][bi] = sub_group_reduce_max(max[0][bi]);
                            dq_max_input[1][bi] = sub_group_reduce_max(max[1][bi]);
                            // if (get_group_id(0) == 0 && get_group_id(2) == 0 && get_local_id(0) == 0 && get_local_id(2) == 0) {
                            //     printf("(%.3f,%.3f)", (float)dq_max_input[0][bi], (float)dq_max_input[1][bi]);
                            // }
                        }

                        de_quantize_scale[0] = dq_max_input[0] / quan;
                        de_quantize_scale[1] = dq_max_input[1] / quan;
                    }

                    // Packing 4 of converted inputs to integer type
                    unroll_for (uint bi = 0; bi < HALF_TILE_B; ++bi) {
                        // packed_in_0[bi] = as_int(CAT(convert_, MAKE_VECTOR_TYPE(DQ_TYPE, INPUT_LOAD_SIZE))(tiled_input_0[bi] / de_quantize_scale[bi]));
                        packed_in_0[bi] = as_int(CAT(convert_, MAKE_VECTOR_TYPE(DQ_TYPE, INPUT_LOAD_SIZE))(tiled_input_0[bi] / de_quantize_scale[batch_sglid][bi]));
                    }
                #else
                    // Quantizing for loaded input using max value
                    INPUT0_TYPE                                max[2][HALF_TILE_B] = { 0 };
                    MAKE_VECTOR_TYPE(INPUT0_TYPE, HALF_TILE_B) de_quantize_scale[2] = { 0 };
                    // MAKE_VECTOR_TYPE(INPUT0_TYPE, HALF_TILE_B) de_quantize_scale = 1;
                    // MAKE_VECTOR_TYPE(INPUT0_TYPE, TILE_B) de_quantize_scale = 1;
                    MAKE_VECTOR_TYPE(INPUT0_TYPE, HALF_TILE_B) quantize_scale = 1;
                    MAKE_VECTOR_TYPE(INPUT0_TYPE, HALF_TILE_B) quan = 128;
                    MAKE_VECTOR_TYPE(INPUT0_TYPE, HALF_TILE_B) dq_max_input[2] = { 0 };
                    unroll_for (uint bi = 0; bi < HALF_TILE_B; ++bi) {
                        // max[bi] = fmax(fmax(fabs(tiled_input_0[bi][0]), fabs(tiled_input_0[bi][1])), fmax(fabs(tiled_input_0[bi][2]), fabs(tiled_input_0[bi][3])));
                        // dq_max_input[bi] = sub_group_reduce_max(max[bi]);
                        // max[0][bi] = fmax(max[0][bi], max[1][bi]);
                        if (INPUT0_FEATURE_NUM*INPUT0_BATCH_NUM > ((bgid + local_id) * TILE_B + (bi+1) * 2)) {
                            max[0][bi] = fmax(fmax(fabs(tiled_input_0[bi][0]), fabs(tiled_input_0[bi][1])), fmax(fabs(tiled_input_0[bi][2]), fabs(tiled_input_0[bi][3])));
                        } else {
                            max[batch_sglid][bi] = fmax(fmax(fabs(tiled_input_0[bi][0]), fabs(tiled_input_0[bi][1])), fmax(fabs(tiled_input_0[bi][2]), fabs(tiled_input_0[bi][3])));
                        }
                    }

                    unroll_for (uint bi = 0; bi < HALF_TILE_B; ++bi) {
                        dq_max_input[0][bi] = sub_group_reduce_max(max[0][bi]);
                    }
                    quantize_scale = quan / dq_max_input[0];
                    de_quantize_scale[0] = dq_max_input[0] / quan;
                    de_quantize_scale[1] = de_quantize_scale[0];

                    // Packing 4 of converted inputs to integer type
                    unroll_for (uint bi = 0; bi < HALF_TILE_B; ++bi) {
                        packed_in_0[bi] = as_int(CAT(convert_, MAKE_VECTOR_TYPE(DQ_TYPE, INPUT_LOAD_SIZE))(tiled_input_0[bi] * quantize_scale[bi]));
                    }

                    // if (get_sub_group_local_id() == 0 && get_group_id(0) == 0 && get_group_id(2) == 0 &&
                    //     get_local_id(0) == 0 && get_local_id(2) == 0) {
                    //     printf(" >> de_quantize_scale(%d, %d, %d, %d, %d, %d, %d, %d)\n", de_quantize_scale[0], de_quantize_scale[1], de_quantize_scale[2],
                    //                 de_quantize_scale[3], de_quantize_scale[4], de_quantize_scale[5], de_quantize_scale[6], de_quantize_scale[7]);
                    // }
                #endif
            #endif

            // __local SLM_FILTER_VEC* char_slm_weight = (__local SLM_FILTER_VEC*)wei_local_mem;
            __local int* char_slm_weight = (__local int*)wei_local_mem;

            uint weights_idx = weights_offset + local_id * SIMD * FILTER_LOAD_ITERS * FILTER_LOAD_BLOCK_SIZE;
            uint wei_local_idx = local_id * SIMD * FILTER_LOAD_ITERS * (FILTER_LOAD_BLOCK_SIZE/2) + sglid * 2;

            // DECOMPRESSION_SCALE_POST_OP SHOULD be enabled for dynamic quantize FC : scale is ACCUMULATOR_VAL_ONE
            unroll_for(uint load_iter = 0; load_iter < FILTER_LOAD_ITERS; ++load_iter) {
                // uchar4 wei_packed = as_uchar4(_sub_group_block_read_uc4((const __global uchar *)(weights) + (weights_idx)));
                SLM_FILTER_PACKED_VEC wei_packed = BLOCK_READN(FILTER_TYPE, FILTER_LOAD_BLOCK_SIZE, weights, weights_idx);
                DQ_SLM_FILTER_UNPACKED_VEC dq_wei_unpacked = UNPACK_MIXED_INT4(DQ_TYPE, *((uint4x8_t *)&wei_packed));

                // Calculate zero-point and scale only for DECOMPRESSION_SCALE_POST_OP enabled
                #if DECOMPRESSION_ZP_TERM
                    #if DECOMPRESSION_ZP_SCALAR
                        DQ_SLM_FILTER_UNPACKED_VEC dzp = (DQ_SLM_FILTER_UNPACKED_VEC)(DECOMPRESSION_ZP_VALUE);
                    #elif DECOMPRESSION_ZP_GROUPS_NUM > 1
                        DQ_SLM_FILTER_UNPACKED_VEC dzp;
                        unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                            unroll_for(uint kii = 0; kii < FILTER_LOAD_BLOCK_SIZE; ++kii) {
                                const uint offset_ofm = out_f + fi*SIMD + sglid;
                                const uint offset_ifm = ni * TILE_IFM * SIMD + local_id * FILTER_LOAD_ITERS * FILTER_LOAD_BLOCK_SIZE + load_iter * FILTER_LOAD_BLOCK_SIZE + kii;
                                const uint zp_offset = (offset_ofm % DECOMPRESSION_ZP_BATCH_NUM) * DECOMPRESSION_ZP_BATCH_PITCH +
                                                       (offset_ifm / DECOMPRESSION_ZP_GROUP_SIZE) * DECOMPRESSION_ZP_FEATURE_PITCH;
                                dzp[W_IDX] = decompression_zp[zp_offset];
                            }
                        }
                    #else
                        DQ_SLM_FILTER_UNPACKED_VEC dzp = (DQ_SLM_FILTER_UNPACKED_VEC)(d_zps[0]);
                    #endif
                #else
                    DQ_SLM_FILTER_UNPACKED_VEC dzp = (DQ_SLM_FILTER_UNPACKED_VEC)(ACCUMULATOR_VAL_ZERO);
                #endif

                // Calculate weight : w = (w - dzp) * ds
                dq_wei_unpacked -= dzp;

                #if FILTER_LOAD_BLOCK_SIZE == 2
                    DQ_SLM_FILTER_VEC wei_1 = {dq_wei_unpacked.s01, dq_wei_unpacked.s23};
                    char_slm_weight[wei_local_idx] = as_int(wei_1);
                #elif FILTER_LOAD_BLOCK_SIZE == 4
                    DQ_SLM_FILTER_VEC wei_1 = {dq_wei_unpacked.s01, dq_wei_unpacked.s23};
                    char_slm_weight[wei_local_idx] = as_int(wei_1);
                    DQ_SLM_FILTER_VEC wei_2 = {dq_wei_unpacked.s45, dq_wei_unpacked.s67};
                    char_slm_weight[wei_local_idx+1] = as_int(wei_2);
                #elif FILTER_LOAD_BLOCK_SIZE == 8
                    DQ_SLM_FILTER_VEC wei_1 = {dq_wei_unpacked.s01, dq_wei_unpacked.s23};
                    char_slm_weight[wei_local_idx] = as_int(wei_1);
                    DQ_SLM_FILTER_VEC wei_2 = {dq_wei_unpacked.s45, dq_wei_unpacked.s67};
                    char_slm_weight[wei_local_idx+1] = as_int(wei_2);
                    DQ_SLM_FILTER_VEC wei_3 = {dq_wei_unpacked.s89, dq_wei_unpacked.sab};
                    char_slm_weight[wei_local_idx+2] = as_int(wei_3);
                    DQ_SLM_FILTER_VEC wei_4 = {dq_wei_unpacked.scd, dq_wei_unpacked.sef};
                    char_slm_weight[wei_local_idx+3] = as_int(wei_4);
                #else
                    #error "FC bf_tiled kernel: unsupported FILTER_LOAD_BLOCK_SIZE for SLM kernel"
                #endif

                wei_local_idx += SIMD * (FILTER_LOAD_BLOCK_SIZE/2);
                weights_idx += SIMD * FILTER_LOAD_BLOCK_SIZE;
            }

            wei_local_idx = sglid * 2;

            barrier(CLK_LOCAL_MEM_FENCE);
        #endif  // USE_SLM && COMPRESSED_WEIGHTS_INT4

        unroll_for(uint ki = 0; ki < (TILE_IFM * SIMD) / TILE_K; ++ki) {
            #if USE_SLM && COMPRESSED_WEIGHTS_INT4
                #if (TILE_K != 1) && (TILE_K != 2) && (TILE_K != 4)
                    #error "FC bf_tiled kernel: unsupported TILE_K size for SLM kernel"
                #endif
            #elif COMPRESSED_WEIGHTS_INT4
                FILTER_PACKED_VEC_TYPE wei_packed = FILTER_BLOCK_READ(weights, weights_offset);
                wei = UNPACK_INT4(ACCUMULATOR_TYPE, *((INT4_PACKED_TYPE*)&wei_packed));
            #else
                wei = TO_FILTER_VEC_TYPE(FILTER_BLOCK_READ(weights, weights_offset));
            #endif

            #if COMPRESSED_WEIGHTS && !USE_SLM
                ACCUMULATOR_TYPE* w = (ACCUMULATOR_TYPE*)(&wei);
                unroll_for(uint kii = 0; kii < TILE_K; ++kii) {
                    unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                        const uint offset_ofm = out_f + fi*SIMD + sglid;
                        // Valid only if DECOMPRESSION_SCALE_POST_OP is enabled
                        ACCUMULATOR_TYPE ds = ACCUMULATOR_VAL_ONE;

                        #if DECOMPRESSION_ZP_TERM
                            #if DECOMPRESSION_ZP_SCALAR
                                ACCUMULATOR_TYPE dzp = DECOMPRESSION_ZP_VALUE;
                            #elif DECOMPRESSION_ZP_GROUPS_NUM > 1
                                const uint zp_offset = (offset_ofm % DECOMPRESSION_ZP_BATCH_NUM) * DECOMPRESSION_ZP_BATCH_PITCH +
                                                    ((kii + ki*TILE_K + ni*TILE_IFM*SIMD) / DECOMPRESSION_ZP_GROUP_SIZE) * DECOMPRESSION_ZP_FEATURE_PITCH;
                                ACCUMULATOR_TYPE dzp = decompression_zp[zp_offset];
                            #else
                                ACCUMULATOR_TYPE dzp = d_zps[fi % DECOMPRESSION_ZP_LENGTH];
                            #endif
                        #else
                            ACCUMULATOR_TYPE dzp = ACCUMULATOR_VAL_ZERO;
                        #endif
                        w[W_IDX] = (w[W_IDX] - dzp) * ds;
                    }
                }
            #endif

            #if USE_SLM && COMPRESSED_WEIGHTS_INT4
                // Error if TILE_OFM != 2
                #if DYNAMIC_QUANTIZE
                    // Compute input * weight : packed char4 type
                    char4 input_val = AS_DQ_TYPE_4(_sub_group_shuffle(packed_in_0[0], ki));
                    char8 weight = vload8(0, (__local char *)(&char_slm_weight[wei_local_idx + 16*2*ki]));
                    char4 first_weight = weight.s0123;
                    char4 second_weight = weight.s4567;
                    unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
                        acc_tmp[0][bi] = imad_SW(acc_tmp[0][bi], input_val, first_weight);
                        acc_tmp[1][bi] = imad_SW(acc_tmp[1][bi], input_val, second_weight);
                        input_val = as_char4(_sub_group_shuffle(packed_in_0[(bi+1) / 2], ((bi+1) % 2) * 8 + ki));
                    }
                #else
                    char8 weight = vload8(0, (__local char *)(&char_slm_weight[wei_local_idx + 16*2*ki]));
                    char4 first_weight = weight.s0123;
                    char4 second_weight = weight.s4567;
                    unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
                        half4 in_val = as_half4(_sub_group_shuffle(((int2*)(&tiled_input_0[bi/2]))[0], (bi % 2) * 8 + ki));
                        unroll_for (uint kii = 0; kii < TILE_K; ++kii) {
                            ((ACCUMULATOR_TYPE*)(&acc_tmp[bi]))[0] += in_val[kii] * convert_half(first_weight[kii]);
                            ((ACCUMULATOR_TYPE*)(&acc_tmp[bi]))[1] += in_val[kii] * convert_half(second_weight[kii]);
                        }
                    }
                #endif
            #else
                unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
                    unroll_for (uint kii = 0; kii < TILE_K; ++kii) {
                        const uint total_k = ki * TILE_K + kii;
                        INPUT0_TYPE in_val = _sub_group_shuffle(((INPUT0_TYPE*)(&in_0[bi]))[total_k / SIMD], total_k % SIMD);
                        unroll_for (uint fi = 0; fi < TILE_OFM; ++fi) {
                            ((ACCUMULATOR_TYPE*)(&acc_tmp[bi]))[fi] += in_val * ((ACCUMULATOR_TYPE*)(&wei))[W_IDX];
                        }
                    }
                }

            #endif

            weights_offset += TILE_K_OFM_PACKED * SIMD;

            #if (TILE_IFM * SIMD > DECOMPRESSION_SCALE_GROUP_SIZE)
                unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
                    unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                        const uint offset_ofm = out_f + fi*SIMD + sglid;

                        #if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                            const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) * DECOMPRESSION_SCALE_BATCH_PITCH +
                                                    ((ni*TILE_IFM*SIMD + ki*TILE_K) / DECOMPRESSION_SCALE_GROUP_SIZE)*DECOMPRESSION_SCALE_FEATURE_PITCH;
                            ACCUMULATOR_TYPE ds = decompression_scale[scale_offset];
                        #else
                            ACCUMULATOR_TYPE ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
                        #endif

                        #if USE_SLM && DYNAMIC_QUANTIZE
                            // ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += convert_half(((int *)(&acc_tmp[fi]))[bi]) * ds * de_quantize_scale[bi];
                            // ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += convert_half(((int *)(&acc_tmp[fi]))[bi]) * ds * de_quantize_scale[bi / 2];
                            ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += convert_half(((int *)(&acc_tmp[fi]))[bi]) * ds * de_quantize_scale[bi % 2][bi / 2];
                            // ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += convert_half(((int *)(&acc_tmp[fi]))[bi]) * ds * de_quantize_scale[0][bi / 2];
                            acc_tmp[fi][bi] = 0;
                        #else
                            ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += ((ACCUMULATOR_TYPE*)(&acc_tmp[bi]))[fi] * ds;
                            acc_tmp[bi][fi] = 0;
                        #endif
                    }
                }
            #endif
        }  // Whole tile_k elements of each iteration : ki

        #if (TILE_IFM * SIMD <= DECOMPRESSION_SCALE_GROUP_SIZE)
            const uint ni_offset = ((ni*TILE_IFM*SIMD) / DECOMPRESSION_SCALE_GROUP_SIZE)*DECOMPRESSION_SCALE_FEATURE_PITCH;
            unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
                unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                    const uint offset_ofm = out_f + fi*SIMD + sglid;

                    #if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                        const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) * DECOMPRESSION_SCALE_BATCH_PITCH + ni_offset;
                        ACCUMULATOR_TYPE ds = decompression_scale[scale_offset];
                    #else
                        ACCUMULATOR_TYPE ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
                    #endif

                    #if USE_SLM && DYNAMIC_QUANTIZE
                        // ((ACCUMULATOR_TYPE*)(&acc[fi]))[bi] += convert_half(((int *)(&acc_tmp[fi]))[bi]) * ds * de_quantize_scale[bi];
                        // ((ACCUMULATOR_TYPE*)(&acc[fi]))[bi] += convert_half(((int *)(&acc_tmp[fi]))[bi]) * ds * de_quantize_scale[bi / 2];
                        ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += convert_half(((int *)(&acc_tmp[fi]))[bi]) * ds * de_quantize_scale[bi % 2][bi / 2];
                        // ((ACCUMULATOR_TYPE*)(&acc[fi]))[bi] += convert_half(((int *)(&acc_tmp[fi]))[bi]) * ds * de_quantize_scale[0][bi / 2];
                    #else
                        ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += ((ACCUMULATOR_TYPE*)(&acc_tmp[bi]))[fi] * ds;
                    #endif
                }
            }
        #endif
    }  // Done main compute loop : ni

    // =====================================================================================================================================
    // Leftovers
#if MAIN_LOOP_ELEMENTS_COUNT % (TILE_IFM * SIMD) != 0
    // Handle leftovers in normal case without alignment correction.
    #define LEFTOVER_IFM               (MAIN_LOOP_ELEMENTS_COUNT % (TILE_IFM * SIMD))
    {
        #define LOAD_IN_0(bi) do {                                  \
                in_0[bi] = INPUT_BLOCK_READ(input, input_offset);   \
                input_offset += TILE_IN_B_PITCH;                    \
            } while (false)

        CONST_LOOP(TILE_B, LOAD_IN_0);
        #undef LOAD_IN_0
        input_offset += TILE_IFM * SIMD - TILE_IN_B_PITCH * TILE_B;
        unroll_for(uint ki = 0; ki < CEIL_DIV(LEFTOVER_IFM, TILE_K); ++ki) {
            #if USE_SLM
                FILTER_VEC_TYPE wei = 0;
            #endif

            #if COMPRESSED_WEIGHTS_INT4
                FILTER_PACKED_VEC_TYPE wei_packed = FILTER_BLOCK_READ(weights, weights_offset);
                wei = UNPACK_INT4(ACCUMULATOR_TYPE, *((INT4_PACKED_TYPE*)&wei_packed));
            #else
                wei = TO_FILTER_VEC_TYPE(FILTER_BLOCK_READ(weights, weights_offset));
            #endif

            #if COMPRESSED_WEIGHTS
                ACCUMULATOR_TYPE* w = (ACCUMULATOR_TYPE*)(&wei);
                unroll_for(uint kii = 0; kii < TILE_K; ++kii) {
                    unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
                        uint offset_ofm = out_f + fi*SIMD + get_sub_group_local_id();
                        #if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                            const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) * DECOMPRESSION_SCALE_BATCH_PITCH +
                                                      ((kii + ki*TILE_K + iterations*TILE_IFM*SIMD) / DECOMPRESSION_SCALE_GROUP_SIZE)*DECOMPRESSION_SCALE_FEATURE_PITCH;
                            ACCUMULATOR_TYPE ds = decompression_scale[scale_offset];
                        #else
                            ACCUMULATOR_TYPE ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
                        #endif

                        #if DECOMPRESSION_ZP_TERM
                            #if DECOMPRESSION_ZP_SCALAR
                                ACCUMULATOR_TYPE dzp = DECOMPRESSION_ZP_VALUE;
                            #elif DECOMPRESSION_ZP_GROUPS_NUM > 1
                                const uint zp_offset = (offset_ofm % DECOMPRESSION_ZP_BATCH_NUM) * DECOMPRESSION_ZP_BATCH_PITCH +
                                                    ((kii + ki*TILE_K + iterations*TILE_IFM*SIMD) / DECOMPRESSION_ZP_GROUP_SIZE) * DECOMPRESSION_ZP_FEATURE_PITCH;
                                ACCUMULATOR_TYPE dzp = decompression_zp[zp_offset];
                            #else
                                ACCUMULATOR_TYPE dzp = d_zps[fi % DECOMPRESSION_ZP_LENGTH];
                            #endif
                        #else
                            ACCUMULATOR_TYPE dzp = ACCUMULATOR_VAL_ZERO;
                        #endif
                        w[W_IDX] = (w[W_IDX] - dzp) * ds;
                    }
                }
            #endif
            weights_offset += TILE_K_OFM_PACKED * SIMD;

            unroll_for (uint kii = 0; kii < TILE_K; ++kii) {
                unroll_for (uint fi = 0; fi < TILE_OFM; ++fi) {
                    unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
                        const uint total_k = ki * TILE_K + kii;
                        if (total_k < LEFTOVER_IFM) {
                            INPUT0_TYPE in_val = _sub_group_shuffle(((INPUT0_TYPE*)(&in_0[bi]))[total_k / SIMD], total_k % SIMD);
                            ((ACCUMULATOR_TYPE*)(&acc[bi]))[fi] += in_val * ((ACCUMULATOR_TYPE*)(&wei))[W_IDX];
                        }
                    }
                }
            }
        }
    }
    #undef LEFTOVER_IFM
#endif // MAIN_LOOP_ELEMENTS_COUNT % (TILE_IFM * SIMD) != 0

    // =====================================================================================================================================
    // Post-processing: bias, activation, fused-ops
    ACTIVATION_VEC_TYPE activated[TILE_B] = { };
    for (uint bi = 0; bi < TILE_B; ++bi) {
        activated[bi] = TO_ACTIVATION_VEC_TYPE(acc[bi]);
    }

#if BIAS_TERM
    #if TILE_OUT_F_NUM % (TILE_OFM * SIMD) == 0
        BIAS_VEC_TYPE bias = BIAS_BLOCK_READ(biases, out_f);
    #else
        BIAS_VEC_TYPE bias = 0;
        unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
            ((BIAS_TYPE*)(&bias))[fi] = biases[out_f + sglid + fi * SIMD];
        }
    #endif
    unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
        activated[bi] += TO_ACTIVATION_VEC_TYPE(bias);
    }
#endif

    OUTPUT_VEC_TYPE result[TILE_B] = { };
#if HAS_FUSED_OPS
    unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
    #if TILE_OFM > 1
        unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
            FUSED_OPS_VEC;
            result[bi][fi] = FUSED_OPS_RESULT_VEC;
        }
    #else
        FUSED_OPS_SCALAR;
        result[bi] = FUSED_OPS_RESULT_SCALAR;
    #endif // TILE_OFM > 1
    }
#else
    unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
        result[bi] = TO_OUTPUT_VEC_TYPE(ACTIVATION_TYPED(activated[bi], ACTIVATION_PARAMS_TYPED));
    }
#endif

    // =====================================================================================================================================
    // Write results
    uint output_offset = out_f * TILE_OUT_F_PITCH + out_b * TILE_OUT_B_PITCH + OUTPUT_OFFSET;

    if (USE_BLOCK_WRITE && (TILE_OUT_F_NUM % (TILE_OFM * SIMD) == 0 || out_f + (TILE_OFM * SIMD) <= TILE_OUT_F_NUM)) {
#if IS_DYNAMIC
        #define WRITE_OUTPUT(bi) do {                                       \
                if (bi + out_b < BATCH_SIZE)                                \
                    OUTPUT_BLOCK_WRITE(output, output_offset, result[bi]);  \
                output_offset += TILE_OUT_B_PITCH;                          \
            } while (false)
#else
        #define WRITE_OUTPUT(bi) do {                                       \
                OUTPUT_BLOCK_WRITE(output, output_offset, result[bi]);      \
                output_offset += TILE_OUT_B_PITCH;                          \
            } while (false)
#endif
        CONST_LOOP(TILE_B, WRITE_OUTPUT);
        #undef WRITE_OUTPUT
    } else {
        output_offset += sglid;

        for (uint bi = 0; bi < TILE_B; ++bi) {
            for (uint fi = 0; fi < TILE_OFM; ++fi) {
                const bool should_write =
#if IS_DYNAMIC
                    bi + out_b < BATCH_SIZE &&
#endif
                    (TILE_OUT_F_NUM % (TILE_OFM * SIMD) == 0 ||
                    out_f + fi * SIMD + sglid < TILE_OUT_F_NUM);
                if (should_write) {
                    output[output_offset] = ((OUTPUT_TYPE*)(&result[bi]))[fi];
                }
                output_offset += SIMD;
            }
            output_offset += TILE_OUT_B_PITCH - TILE_OFM * SIMD;
        }
    }
    // =====================================================================================================================================
}


REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(fc_quan_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
#if DECOMPRESSION_SCALE_TERM
    const __global DECOMPRESSION_SCALE_TYPE* decompression_scale,
#endif
#if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
    const __global DECOMPRESSION_ZP_TYPE* decompression_zp,
#endif
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
#if USE_SLM
    // __local ACCUMULATOR_TYPE wei_local_mem[TILE_IFM * SIMD * TILE_OFM * SIMD];
    __local int wei_local_mem[TILE_IFM * SIMD * TILE_OFM * SIMD];
#endif
#if IS_DYNAMIC && COMPRESSED_WEIGHTS_INT4
    const int batch_size = BATCH_SIZE;
    // if (get_group_id(0) == 0 && get_group_id(2) == 0 && get_local_id(2) == 0)
    //     printf(">> DYN QUAN Batch size : (%d)\n", batch_size);

    if (batch_size == 1) {
        FUNC_CALL(fc_bf_tiled_kernel_forced_tile_b1)(
            OPTIONAL_SHAPE_INFO_TENSOR
            input,
        #if DECOMPRESSION_SCALE_TERM
            decompression_scale,
        #endif
        #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
            decompression_zp,
        #endif
            output,
            weights
        #if BIAS_TERM
            , biases
        #endif
        #if HAS_FUSED_OPS_DECLS
            , FUSED_OPS_ARGS
        #endif
        );
    } else if (batch_size == 2) {
        FUNC_CALL(fc_bf_tiled_kernel_forced_tile_b2)(
            OPTIONAL_SHAPE_INFO_TENSOR
            input,
        #if DECOMPRESSION_SCALE_TERM
            decompression_scale,
        #endif
        #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
            decompression_zp,
        #endif
            output,
            weights
        #if BIAS_TERM
            , biases
        #endif
        #if HAS_FUSED_OPS_DECLS
            , FUSED_OPS_ARGS
        #endif
        );
    } else if (batch_size == 3) {
        FUNC_CALL(fc_bf_tiled_kernel_forced_tile_b3)(
            OPTIONAL_SHAPE_INFO_TENSOR
            input,
        #if DECOMPRESSION_SCALE_TERM
            decompression_scale,
        #endif
        #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
            decompression_zp,
        #endif
            output,
            weights
        #if BIAS_TERM
            , biases
        #endif
        #if HAS_FUSED_OPS_DECLS
            , FUSED_OPS_ARGS
        #endif
        );
    } else if (batch_size == 4) {
        FUNC_CALL(fc_bf_tiled_kernel_forced_tile_b4)(
            OPTIONAL_SHAPE_INFO_TENSOR
            input,
        #if DECOMPRESSION_SCALE_TERM
            decompression_scale,
        #endif
        #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
            decompression_zp,
        #endif
            output,
            weights
        #if BIAS_TERM
            , biases
        #endif
        #if HAS_FUSED_OPS_DECLS
            , FUSED_OPS_ARGS
        #endif
        );
    } else if (batch_size == 5) {
        FUNC_CALL(fc_bf_tiled_kernel_forced_tile_b5)(
            OPTIONAL_SHAPE_INFO_TENSOR
            input,
        #if DECOMPRESSION_SCALE_TERM
            decompression_scale,
        #endif
        #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
            decompression_zp,
        #endif
            output,
            weights
        #if BIAS_TERM
            , biases
        #endif
        #if HAS_FUSED_OPS_DECLS
            , FUSED_OPS_ARGS
        #endif
        );
    } else if (batch_size == 6) {
        FUNC_CALL(fc_bf_tiled_kernel_forced_tile_b6)(
            OPTIONAL_SHAPE_INFO_TENSOR
            input,
        #if DECOMPRESSION_SCALE_TERM
            decompression_scale,
        #endif
        #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
            decompression_zp,
        #endif
            output,
            weights
        #if BIAS_TERM
            , biases
        #endif
        #if HAS_FUSED_OPS_DECLS
            , FUSED_OPS_ARGS
        #endif
        );
    } else if (batch_size == 7) {
        FUNC_CALL(fc_bf_tiled_kernel_forced_tile_b7)(
            OPTIONAL_SHAPE_INFO_TENSOR
            input,
        #if DECOMPRESSION_SCALE_TERM
            decompression_scale,
        #endif
        #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
            decompression_zp,
        #endif
            output,
            weights
        #if BIAS_TERM
            , biases
        #endif
        #if HAS_FUSED_OPS_DECLS
            , FUSED_OPS_ARGS
        #endif
        );
    } else {
        // if (get_sub_group_local_id() == 0 && get_group_id(0) == 0 && get_group_id(2) == 0 &&
        //     get_local_id(0) == 0 && get_local_id(2) == 0) {
        //         printf(">>>> DYNAMIC : MAIN_LOOP_ELEMENTS_COUNT(%d) batch(%d) => group_0(%d) local_0(%d) / group2(%d) local2(%d)\n",
        //                 (int)MAIN_LOOP_ELEMENTS_COUNT, (int)INPUT0_FEATURE_NUM*INPUT0_BATCH_NUM,
        //                 (int)get_global_size(0), (int)get_local_size(0), (int)get_global_size(2), (int)get_local_size(2));
        // }
        FUNC_CALL(fc_bf_tiled_kernel_dynamic_quantized)(
            OPTIONAL_SHAPE_INFO_TENSOR
            input,
        #if DECOMPRESSION_SCALE_TERM
            decompression_scale,
        #endif
        #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
            decompression_zp,
        #endif
            output,
            weights
        #if USE_SLM
            , wei_local_mem
        #endif
        #if BIAS_TERM
            , biases
        #endif
        #if HAS_FUSED_OPS_DECLS
            , FUSED_OPS_ARGS
        #endif
        );
    }
#else
    FUNC_CALL(fc_bf_tiled_kernel_dynamic_quantized)(
        OPTIONAL_SHAPE_INFO_TENSOR
        input,
    #if DECOMPRESSION_SCALE_TERM
        decompression_scale,
    #endif
    #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
        decompression_zp,
    #endif
        output,
        weights
    #if USE_SLM
        , wei_local_mem
    #endif
    #if BIAS_TERM
        , biases
    #endif
    #if HAS_FUSED_OPS_DECLS
        , FUSED_OPS_ARGS
    #endif
    );
#endif
}

#undef INPUT_VEC_TYPE
#undef ACCUMULATOR_VEC_TYPE
#undef FILTER_VEC_TYPE
#undef BIAS_VEC_TYPE
#undef OUTPUT_VEC_TYPE
#undef ACTIVATION_VEC_TYPE
#undef TO_OUTPUT_VEC_TYPE
#undef TO_ACTIVATION_VEC_TYPE

#undef INPUT_BLOCK_READ
#undef FILTER_BLOCK_READ
#undef BIAS_BLOCK_READ
#undef OUTPUT_BLOCK_WRITE

#undef USE_BLOCK_WRITE

#undef MAIN_LOOP_ELEMENTS_COUNT
