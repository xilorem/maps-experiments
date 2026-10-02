// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Full-network layer dispatch adapted from the SDK onnx_mobilevit_v2 test.
// SPDX-License-Identifier: Apache-2.0
#include "tile.h"

#include "eventunit.h"
#include "fsync.h"

#include "kernel_test_utils.h"
#include "mobilevit_graph.h"

#include "utils/maps_operation_indexing.h"

#include "add_fp16_spatz.h"
#include "conv2dgemm_fp16_spatz.h"
#include "gemm_fp16_spatz.h"
#include "globalaveragepool_fp16_spatz.h"
#include "groupnorm_fp16_spatz.h"
#include "mul_bcast_fp16_spatz_params.h"
#include "mul_fp16_spatz.h"
#include "reducesum_fp16_spatz.h"
#include "relu_fp16_spatz.h"
#include "sigmoid_fp16_spatz.h"
#include "softmax_fp16_spatz.h"
#include "transpose_fp16_spatz.h"

#define HID get_hartid()

/* The blob, loaded from the ELF into L2 by mobilevit_data.S. */
extern const uint8_t mobilevit_data[];

/*
 * Every layer's output, all resident at once, so a residual branch can read a tensor
 * produced several layers earlier without any liveness analysis. NOLOAD, so it costs
 * nothing in the ELF; nothing zeroes it, which is fine because every element is written
 * before it is read.
 */
static float16 mobilevit_arena[MVIT_ARENA_BYTES / sizeof(float16)]
    __attribute__((section(".l2_arena"), aligned(4)));

extern uint32_t _spatz_binary_start; /* CV32 linker script */

static inline float16 *arena_at(uint32_t off)
{
    return (float16 *)((uintptr_t)mobilevit_arena + off);
}

static inline const float16 *blob_at(uint32_t off)
{
    return (const float16 *)((uintptr_t)mobilevit_data + off);
}

static inline const float16 *operand(uint8_t space, uint32_t off, uint32_t token)
{
    if (space == MVIT_SPACE_BLOB && off == MVIT_INPUT_OFF)
        return blob_at(MVIT_INPUTS_OFF) + token * MVIT_INPUT_ELEMENTS;
    return space == MVIT_SPACE_BLOB ? blob_at(off) : arena_at(off);
}

static void run_layer(const mvit_layer_t *L, uint32_t token)
{
    const float16 *src0 = operand(L->src0_space, L->src0_off, token);
    float16 *dst        = arena_at(L->dst_off);

    uint32_t in_shape[4]  = {L->in_shape[0], L->in_shape[1], L->in_shape[2], L->in_shape[3]};
    uint32_t out_shape[4] = {L->out_shape[0], L->out_shape[1], L->out_shape[2], L->out_shape[3]};

    switch (L->op) {
    case MVIT_CONV:
        MAGIA_conv2dgemm_fp16_spatz(src0,
                                    blob_at(L->w_off),
                                    L->has_bias ? blob_at(L->b_off) : NULL,
                                    dst,
                                    in_shape,
                                    out_shape,
                                    L->kernel_h,
                                    L->kernel_w,
                                    L->stride_h,
                                    L->stride_w,
                                    L->pad_h,
                                    L->pad_w,
                                    L->group,
                                    (int)L->has_bias);
        break;

    case MVIT_RELU:
        MAGIA_relu_fp16_spatz(src0, dst, L->numel);
        break;

    case MVIT_SIGMOID:
        MAGIA_sigmoid_fp16_spatz(src0, dst, L->numel);
        break;

    case MVIT_ADD:
        MAGIA_add_fp16_spatz(src0, operand(L->src1_space, L->src1_off, token), dst, L->numel);
        break;

    case MVIT_MUL:
        MAGIA_mul_fp16_spatz(src0, operand(L->src1_space, L->src1_off, token), dst, L->numel);
        break;

    case MVIT_MULBCAST:
        MAGIA_mul_bcast_fp16_spatz(src0,
                                   operand(L->src1_space, L->src1_off, token),
                                   dst,
                                   L->rows,
                                   L->row_len,
                                   L->bcast_mode);
        break;

    case MVIT_GROUPNORMALIZATION:
        MAGIA_groupnorm_fp16_spatz(src0,
                                   dst,
                                   blob_at(L->w_off),
                                   blob_at(L->b_off),
                                   in_shape,
                                   L->num_groups,
                                   (float16)MVIT_GN_EPS);
        break;

    case MVIT_SOFTMAX: {
        /* The kernel takes rows = shape[0]*shape[1]*shape[2] and a row length of
         * shape[3], so the axis-(-1) softmax is passed flattened that way. */
        uint32_t sm_shape[4] = {L->sm_rows, 1, 1, L->sm_len};

        MAGIA_softmax_fp16_spatz(src0, dst, sm_shape);
        break;
    }

    case MVIT_REDUCESUM:
        MAGIA_reducesum_fp16_spatz(src0, dst, L->outer, L->reduce, L->inner);
        break;

    case MVIT_TRANSPOSE: {
        /* Shapes and perm come from the table with the leading axes already merged; the
         * kernel takes them by non-const pointer, so copy them out. */
        uint32_t perm[MVIT_MAX_RANK];
        uint32_t t_in[MVIT_MAX_RANK];
        uint32_t t_out[MVIT_MAX_RANK];

        for (uint32_t i = 0; i < L->rank; i++) {
            perm[i]  = L->perm[i];
            t_in[i]  = L->t_in_shape[i];
            t_out[i] = L->t_out_shape[i];
        }

        MAGIA_transpose_fp16_spatz(src0, dst, perm, t_in, t_out, L->rank, L->iterations);
        break;
    }

    case MVIT_GLOBALAVERAGEPOOL:
        MAGIA_globalaveragepool_fp16_spatz(src0, dst, in_shape);
        break;

    case MVIT_GEMM: {
        /*
         * Called transposed: A is the [1000, 256] fc weight, B the 256-long embedding as
         * a column, Y a [1000, 1] column - which is the same bytes as [1, 1000]. The
         * kernel shards the GEMM's M, so the ONNX orientation (M = 1) would leave the
         * whole layer on tile 0.
         */
        uint32_t a_shape[2] = {L->out_shape[1], L->in_shape[1]};
        uint32_t b_shape[2] = {L->in_shape[1], 1};
        uint32_t c_shape[2] = {L->out_shape[1], 1};
        uint32_t y_shape[2] = {L->out_shape[1], 1};

        MAGIA_gemm_fp16_spatz(blob_at(L->w_off),
                              src0,
                              blob_at(L->b_off),
                              (float16)1.0f,
                              (float16)1.0f,
                              0,
                              0,
                              a_shape,
                              b_shape,
                              c_shape,
                              y_shape,
                              dst);
        break;
    }

    default:
        if (HID == 0)
            printf("[CV32 (%d)] unhandled op %d in layer %s\n", HID, L->op, L->name);
        break;
    }
}

static volatile uint32_t window_start __attribute__((section(".l2_arena")));
static volatile uint32_t output_cycles[NUM_HARTS][MVIT_TOKENS]
    __attribute__((section(".l2_arena")));
static float16 outputs[MVIT_TOKENS][MVIT_OUTPUT_LEN]
    __attribute__((section(".l2_arena"), aligned(4)));

static uint32_t read_cycle(void)
{
    uint32_t cycle;
    __asm__ volatile("rdcycle %0" : "=r"(cycle));
    return cycle;
}

static void barrier(fsync_controller_t *fsync, eu_controller_t *eu)
{
    fsync_sync_global(fsync);
    eu_fsync_wait(eu, WFE);
}

int main(void)
{
    fsync_config_t fc = {.hartid = HID};
    fsync_controller_t fsync = {.base = 0, .cfg = &fc, .api = &fsync_api};
    eu_config_t ec = {.hartid = HID};
    eu_controller_t eu = {.base = 0, .cfg = &ec, .api = &eu_api};
    kt_spatz_init((uint32_t)&_spatz_binary_start);
    eu_fsync_init(&eu, 0);
    fsync_init(&fsync);
    barrier(&fsync, &eu);
    if (HID == 0) window_start = read_cycle();
    barrier(&fsync, &eu);
    for (uint32_t token = 0; token < MVIT_TOKENS; ++token) {
        for (uint32_t i = 0; i < MVIT_NUM_LAYERS; ++i) {
            run_layer(&mobilevit_layers[i], token);
            if (i + 1 == MVIT_NUM_LAYERS)
                output_cycles[HID][token] = read_cycle() - window_start;
            barrier(&fsync, &eu);
        }
        if (HID == 0)
            for (uint32_t i = 0; i < MVIT_OUTPUT_LEN; ++i)
                outputs[token][i] = arena_at(MVIT_OUTPUT_OFF)[i];
        barrier(&fsync, &eu);
    }
    spatz_clk_dis();
    if (HID == 0) {
        printf("REFERENCE_TILES count=%u\n", NUM_HARTS);
        for (uint32_t token = 0; token < MVIT_TOKENS; ++token) {
            uint32_t completion = 0, mismatches = 0, nonfinite = 0;
            for (uint32_t tile = 0; tile < NUM_HARTS; ++tile)
                if (output_cycles[tile][token] > completion)
                    completion = output_cycles[tile][token];
            const float16 *ref = blob_at(MVIT_REFERENCES_OFF) + token * MVIT_OUTPUT_LEN;
            for (uint32_t i = 0; i < MVIT_OUTPUT_LEN; ++i) {
                uint16_t bits = ((const uint16_t *)outputs[token])[i];
                float actual = maps_operation_f16_to_f32(bits), expected = ref[i];
                float diff = actual > expected ? actual - expected : expected - actual;
                float magnitude = expected < 0 ? -expected : expected;
                if ((bits & 0x7c00u) == 0x7c00u) ++nonfinite;
                if (diff > 0.002f + 0.02f * magnitude) ++mismatches;
            }
            printf("REFERENCE_COMPLETION token=%u cycle=%u\n", token, completion);
            printf("REFERENCE_VALIDATION token=%u mismatches=%u nonfinite=%u\n",
                   token, mismatches, nonfinite);
        }
    }
    return 0;
}
