/*
 * Full-mesh reference trimmed from MAGIA SDK's onnx_mobilevit_v2 application.
 * Every kernel shards work over the complete Mesh and every layer ends in the same
 * global barrier as the original application. Tokens deliberately execute serially.
 */
#include "tile.h"

#include "eventunit.h"
#include "fsync.h"
#include "kernel_test_utils.h"
#include "reference_config.h"
#include "utils/maps_operation_indexing.h"

#include "add_fp16_spatz.h"
#include "conv2dgemm_fp16_spatz.h"
#include "groupnorm_fp16_spatz.h"
#include "mul_fp16_spatz.h"
#include "reducesum_fp16_spatz.h"
#include "relu_fp16_spatz.h"
#include "softmax_fp16_spatz.h"

#define HID get_hartid()
#define INPUT_ELEMENTS (8192u)
#define QKV_ELEMENTS (16448u)
#define QUERY_ELEMENTS (64u)
#define REDUCED_ELEMENTS (512u)

extern const uint8_t mobilevit_reference_data_start[];
extern uint32_t _spatz_binary_start;

static float16 norm[INPUT_ELEMENTS] __attribute__((section(".l2_arena"), aligned(4)));
static float16 qkv[QKV_ELEMENTS] __attribute__((section(".l2_arena"), aligned(4)));
static float16 query[QUERY_ELEMENTS] __attribute__((section(".l2_arena"), aligned(4)));
static float16 weighted[INPUT_ELEMENTS] __attribute__((section(".l2_arena"), aligned(4)));
static float16 reduced[REDUCED_ELEMENTS] __attribute__((section(".l2_arena"), aligned(4)));
static float16 activated[INPUT_ELEMENTS] __attribute__((section(".l2_arena"), aligned(4)));
static float16 fused[INPUT_ELEMENTS] __attribute__((section(".l2_arena"), aligned(4)));
static float16 projected[INPUT_ELEMENTS] __attribute__((section(".l2_arena"), aligned(4)));
static float16 outputs[REFERENCE_TOKEN_COUNT][INPUT_ELEMENTS]
    __attribute__((section(".l2_arena"), aligned(4)));
static volatile uint32_t window_start __attribute__((section(".l2_arena")));
static volatile uint32_t output_cycles[NUM_HARTS][REFERENCE_TOKEN_COUNT]
    __attribute__((section(".l2_arena")));

static inline const float16 *data_at(uint32_t offset)
{
    return (const float16 *)(mobilevit_reference_data_start + offset);
}

static inline uint32_t read_cycle(void)
{
    uint32_t cycle;
    __asm__ volatile("rdcycle %0" : "=r"(cycle));
    return cycle;
}

static inline void layer_barrier(fsync_controller_t *fsync, eu_controller_t *event_unit)
{
    fsync_sync_global(fsync);
    eu_fsync_wait(event_unit, WFE);
}

static void run_token(
    uint32_t token,
    fsync_controller_t *fsync,
    eu_controller_t *event_unit)
{
    const float16 *input = data_at(REFERENCE_INPUTS_OFFSET) + token * INPUT_ELEMENTS;
    uint32_t input_shape[4] = {1u, 128u, 4u, 16u};
    uint32_t qkv_shape[4] = {1u, 257u, 4u, 16u};
    uint32_t softmax_shape[4] = {4u, 1u, 1u, 16u};

    MAGIA_groupnorm_fp16_spatz(
        input,
        norm,
        data_at(REFERENCE_GN_SCALE_OFFSET),
        data_at(REFERENCE_GN_BIAS_OFFSET),
        input_shape,
        1u,
        (float16)1.00135803e-05f);
    layer_barrier(fsync, event_unit);

    MAGIA_conv2dgemm_fp16_spatz(
        norm,
        data_at(REFERENCE_QKV_WEIGHT_OFFSET),
        data_at(REFERENCE_QKV_BIAS_OFFSET),
        qkv,
        input_shape,
        qkv_shape,
        1u, 1u, 1u, 1u, 0u, 0u, 1u, 1);
    layer_barrier(fsync, event_unit);

    MAGIA_softmax_fp16_spatz(qkv, query, softmax_shape);
    layer_barrier(fsync, event_unit);

    MAGIA_mul_bcast_fp16_spatz(
        qkv + QUERY_ELEMENTS,
        query,
        weighted,
        128u,
        64u,
        0u);
    layer_barrier(fsync, event_unit);

    MAGIA_reducesum_fp16_spatz(weighted, reduced, 512u, 16u, 1u);
    layer_barrier(fsync, event_unit);

    MAGIA_relu_fp16_spatz(
        qkv + QUERY_ELEMENTS + INPUT_ELEMENTS,
        activated,
        INPUT_ELEMENTS);
    layer_barrier(fsync, event_unit);

    MAGIA_mul_bcast_fp16_spatz(activated, reduced, fused, 512u, 16u, 1u);
    layer_barrier(fsync, event_unit);

    MAGIA_conv2dgemm_fp16_spatz(
        fused,
        data_at(REFERENCE_OUT_WEIGHT_OFFSET),
        data_at(REFERENCE_OUT_BIAS_OFFSET),
        projected,
        input_shape,
        input_shape,
        1u, 1u, 1u, 1u, 0u, 0u, 1u, 1);
    layer_barrier(fsync, event_unit);

    MAGIA_add_fp16_spatz(input, projected, outputs[token], INPUT_ELEMENTS);
    output_cycles[HID][token] = read_cycle() - window_start;
    layer_barrier(fsync, event_unit);
}

int main(void)
{
    fsync_config_t fsync_config;
    fsync_controller_t fsync;
    eu_config_t event_config;
    eu_controller_t event_unit;
    uint32_t completion_cycles[REFERENCE_TOKEN_COUNT];
    bool valid = true;

    kt_spatz_init((uint32_t)&_spatz_binary_start);
    event_config.hartid = HID;
    event_unit.base = 0;
    event_unit.cfg = &event_config;
    event_unit.api = &eu_api;
    eu_fsync_init(&event_unit, 0);
    fsync_config.hartid = HID;
    fsync.base = 0;
    fsync.cfg = &fsync_config;
    fsync.api = &fsync_api;
    fsync_init(&fsync);

    layer_barrier(&fsync, &event_unit);
    if (HID == 0u)
        window_start = read_cycle();
    layer_barrier(&fsync, &event_unit);

    for (uint32_t token = 0u; token < REFERENCE_TOKEN_COUNT; ++token) {
        run_token(token, &fsync, &event_unit);
        if (HID == 0u) {
            uint32_t completion = 0u;
            for (uint32_t tile = 0u; tile < NUM_HARTS; ++tile)
                if (output_cycles[tile][token] > completion)
                    completion = output_cycles[tile][token];
            completion_cycles[token] = completion;
        }
    }

    spatz_clk_dis();
    if (HID == 0u) {
        printf("REFERENCE_TILES count=%u\n", NUM_HARTS);
        for (uint32_t token = 0u; token < REFERENCE_TOKEN_COUNT; ++token)
            printf("REFERENCE_COMPLETION token=%u cycle=%u\n",
                   token, completion_cycles[token]);
        const float16 *references = data_at(REFERENCE_REFERENCES_OFFSET);
        for (uint32_t token = 0u; token < REFERENCE_TOKEN_COUNT; ++token) {
            uint32_t mismatches = 0u;
            uint32_t nonfinite = 0u;
            for (uint32_t index = 0u; index < INPUT_ELEMENTS; ++index) {
                const uint16_t actual_bits = ((const uint16_t *)outputs[token])[index];
                const float actual = maps_operation_f16_to_f32(actual_bits);
                const float expected = references[token * INPUT_ELEMENTS + index];
                const float difference = actual > expected ? actual - expected : expected - actual;
                const float expected_absolute = expected < 0.0f ? -expected : expected;
                if ((actual_bits & 0x7c00u) == 0x7c00u)
                    ++nonfinite;
                if (difference > REFERENCE_ATOL + REFERENCE_RTOL * expected_absolute)
                    ++mismatches;
            }
            if (mismatches != 0u || nonfinite != 0u)
                valid = false;
            printf("REFERENCE_VALIDATION token=%u mismatches=%u nonfinite=%u\n",
                   token, mismatches, nonfinite);
        }
    }
    return valid ? 0 : -1;
}
