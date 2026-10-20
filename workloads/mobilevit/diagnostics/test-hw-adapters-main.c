#include "test_maps_hw_adapters_task_bin.h"
#include "tile.h"
#include "fsync.h"
#include "eventunit.h"
#include "redmule.h"
#include "utils/maps_operations.h"
#include "utils/printf.h"
@DECLARATIONS@
typedef struct {
    const char *kind_name;
    slice_desc_t slices[4];
    op_desc_t op;
    const uint8_t *inputs[2][3];
    uint32_t sizes[2][3];
    const uint8_t *gold[2];
    uint32_t output_bytes;
} test_case_t;
static const test_case_t cases[] = {@CASES@};
static volatile uint32_t errors;
static inline uint32_t clean_half(uint32_t x) {
    __asm__ volatile("slli %0, %0, 16\n\tsrli %0, %0, 16" : "+r"(x));
    return x;
}
int main(void) {
    uint32_t hartid = get_hartid();
    idma_init(); redmule_init(); fsync_init(); eu_init();
    eu_clear_events(0xffffffffu);
    eu_fsync_init(0u); eu_idma_init(0u); eu_redmule_init(0u);
#if MAPS_FIFO_INPLACE_RECV
    /* This diagnostic bypasses maps_fifo_init. */
    maps_slice_alias_reset();
#endif
    if (hartid == 0) {
        /* Field by field: CORE-V GCC ICEs (maybe_postinc) on a zero-filled
         * designated initialiser. */
        maps_operation_runtime_t runtime;
        runtime.kernel_abi_version=MAPS_KERNEL_ABI_VERSION;
        runtime.task_bundle_abi_version=MAPS_TASK_BUNDLE_ABI_VERSION;
        runtime.spatz_binary_start=SPATZ_BINARY_START;
        runtime.spatz_params=(void *)(get_l1_base(hartid)+MAPS_OPERATION_TASK_SCRATCH_OFFSET);
        runtime.spatz_params_bytes=MAPS_OPERATION_TASK_SCRATCH_BYTES;
        runtime.spatz_initialized=0u;
        runtime.background=0;
        runtime.background_ctx=0;
        runtime.add_fp16_task=0u; runtime.mul_bcast_fp16_task=MUL_BCAST_FP16_SPATZ_TASK; runtime.matmul_fp16_task=MATMUL_ROWS_FP16_SPATZ_TASK;
        runtime.relu_fp16_task=0u; runtime.sigmoid_fp16_task=0u;
        runtime.depthwise_conv_fp16_task=DEPTHWISE_CONV_FP16_SPATZ_TASK;
        runtime.softmax_exp_fp16_task=0u; runtime.group_reduce_fp16_task=0u;
        runtime.group_centered_reduce_fp16_task=0u; runtime.group_normalize_fp16_task=0u;
        runtime.reducesum_fp16_task=REDUCESUM_FP16_SPATZ_TASK; runtime.reducemax_fp16_task=0u; runtime.binary_bcast_fp16_task=0u;
        runtime.transpose_fp16_task=TRANSPOSE_STRIDED_FP16_SPATZ_TASK;
        runtime.maxpool_fp16_task=MAXPOOL2D_FP16_SPATZ_TASK;
        int init=maps_operation_runtime_init(&runtime);
        errors=init != 0;
        printf("HW_INIT code=%d cases=%u\n",init,(uint32_t)(sizeof(cases)/sizeof(cases[0])));
        for (uint32_t c=0; !init && c<sizeof(cases)/sizeof(cases[0]); ++c) {
            const test_case_t *tc=&cases[c];
            tile_plan_t plan={.hartid=hartid,.l1_data_base=get_l1_base(hartid),.num_slices=4,.slices=tc->slices};
            for (uint32_t token=0; token<2; ++token) {
                for (uint32_t j=0;j<tc->op.num_inputs;++j)
                    errors += maps_idma_copy_1d_ticketed(0u,(uint32_t)tc->inputs[token][j],plan.l1_data_base+tc->slices[j].l1_offset_bytes,tc->sizes[token][j]) != 0;
                volatile uint16_t *actual=(volatile uint16_t *)(plan.l1_data_base+tc->slices[3].l1_offset_bytes);
                for (uint32_t i=0;i<tc->output_bytes/2;++i) actual[i]=tc->op.kind==OP_IM2COL ? 0x3555u : 0u;
                const uint32_t start=maps_read_cycle();
                int result=maps_execute_operation(&plan,&tc->op,0,&runtime);
                const uint32_t cycles=maps_read_cycle()-start;
#if defined(MAPS_IM2COL_PROFILE) && MAPS_IM2COL_PROFILE
                if (tc->op.kind==OP_IM2COL) {
                    const volatile uint32_t *profile=(const volatile uint32_t *)(get_l1_base(hartid)+0xBFF00u);
                    printf("HW_IM2COL_PROFILE case=%u token=%u zero=%u dma=%u calls=%u bytes=%u\n",
                        c,token,profile[0],profile[1],profile[2],profile[3]);

                }
#endif
#if defined(MAPS_MATMUL_PROFILE) && MAPS_MATMUL_PROFILE
                if (tc->op.kind==OP_MATMUL) {
                    const volatile uint32_t *profile=(const volatile uint32_t *)
                        (((uintptr_t)runtime.spatz_params+sizeof(matmul_fp16_spatz_params_t)+15u)&~(uintptr_t)15u);
                    printf("HW_GEMM_PROFILE case=%u token=%u compute=%u bias=%u\n",c,token,profile[0],profile[1]);
                }
#endif
                printf("HW_CYCLES case=%u token=%u cycles=%u\n",c,token,cycles);
                uint32_t bad=0;
                const uint16_t *expected=(const uint16_t *)tc->gold[token];
                for (uint32_t i=0;i<tc->output_bytes/2;++i) {
                    uint32_t a=clean_half(actual[i]), b=clean_half(expected[i]);
                    if (a!=b && (tc->op.kind!=OP_DEPTHWISE_CONV || ((a|b)&0x7fff))) {
                        if (bad<4) printf("HW_BAD case=%u token=%u index=%u actual=%04x expected=%04x\n",c,token,i,a,b);
                        ++bad;
                    }
                }
                errors += bad + (result!=0);
                printf("HW_CHECK case=%u token=%u kind=%s values=%u errors=%u code=%d\n",c,token,tc->kind_name,tc->output_bytes/2,bad,result);
            }
        }
        spatz_clk_dis();
    }
    fsync_sync_global(); eu_fsync_wait(WFE);
    return errors ? -1 : 0;
}
