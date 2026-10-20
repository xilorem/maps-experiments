/* 2x2 GVSoC regression for event-sleep FIFO execution and minimal CSV timing.
 * Two tiles forward three distinct tokens through the production FIFO runner;
 * the other tiles participate in boot/final barriers. Build as maps_mobilevit
 * with MAPS_FIFO_EVENT_SLEEP=1, MAPS_RESULT_TRACE=1 and live progress enabled. */
#include "tile.h"
#include "fsync.h"
#include "utils/maps_utils_v2.h"

static uint32_t inputs[3][64] __attribute__((section(".l2_bulk.inputs")));
static volatile uint32_t outputs[3][64] __attribute__((section(".l2_arena.outputs")));
static volatile uint32_t errors;

int main(void)
{
    const uint32_t tile = get_hartid();
    perf_init();
    idma_init(); eu_init(); fsync_init();
    eu_clear_events(0xffffffffu); eu_idma_init(0); eu_fsync_init(0);
    const subslice_desc_t sub = {.slice_id=0,.rank=1,.shape={128},.elem_type=ELEM_F16,
        .elem_bytes=2,.strides_bytes={2}};
    const tensor_sub_slice_t copy = {.rank=1,.num_elems=128,.dims={{0,128,2}}};
    const slice_desc_t slice = {.slice_id=0,.rank=1,.shape={128},.elem_type=ELEM_F16,
        .elem_bytes=2,.strides_bytes={2},.num_slots=2,.slot_bytes=256};
    const op_desc_t op = {.kind=OP_RESHAPE,.num_inputs=1,.inputs={sub},
        .num_outputs=1,.outputs={sub}};
    const l2_read_desc_t read = {.src_l2_addr=(uint32_t)inputs,.src_l2_token_stride_bytes=256,
        .dst=sub,.copy_src=copy,.copy_dst=copy};
    const l2_write_desc_t write = {.dst_l2_addr=(uint32_t)outputs,.dst_l2_token_stride_bytes=256,
        .src=sub,.copy_src=copy,.copy_dst=copy};
    const fifo_send_desc_t send = {.transition_id=0,.src_hartid=0,.dst_hartid=1,
        .src=sub,.dst=sub,.copy_src=copy,.copy_dst=copy,.producer_idx=0};
    const fifo_recv_desc_t recv = {.transition_id=0,.src_hartid=0,.dst_hartid=1,
        .dst=sub,.producer_idx=0};
    maps_operation_runtime_t runtime;
    runtime.kernel_abi_version = MAPS_KERNEL_ABI_VERSION;
    runtime.task_bundle_abi_version = MAPS_TASK_BUNDLE_ABI_VERSION;
    runtime.spatz_binary_start = 0;
    runtime.add_fp16_task = 0;
    runtime.mul_bcast_fp16_task = 0;
    runtime.matmul_fp16_task = 0;
    runtime.relu_fp16_task = 0;
    runtime.maxpool_fp16_task = 0;
    runtime.sigmoid_fp16_task = 0;
    runtime.depthwise_conv_fp16_task = 0;
    runtime.transpose_fp16_task = 0;
    runtime.softmax_exp_fp16_task = 0;
    runtime.group_reduce_fp16_task = 0;
    runtime.group_centered_reduce_fp16_task = 0;
    runtime.group_normalize_fp16_task = 0;
    runtime.reducesum_fp16_task = 0;
    runtime.reducemax_fp16_task = 0;
    runtime.binary_bcast_fp16_task = 0;
    runtime.spatz_params = 0;
    runtime.spatz_params_bytes = 0;
    runtime.spatz_initialized = 0;
    runtime.background = 0;
    runtime.background_ctx = 0;

    fifo_tile_plan_t plan = {.hartid=tile,.l1_data_base=get_l1_base(tile)+0x20000,
        .num_token_slots=2,.num_slices=1,.slices=&slice,.num_ops=1,.ops=&op,
        .operation_runtime=&runtime,.fifo={.num_producers=1,.num_slots=2,.slot_data_size=256}};
    if (tile==0) {
        for (uint32_t token=0;token<3;++token)
            for (uint32_t i=0;i<64;++i) inputs[token][i]=0x3c003800u+token*64u+i;
        plan.num_l2_reads=1; plan.l2_reads=&read; plan.num_sends=1; plan.sends=&send;
    } else if (tile==1) {
        plan.num_recvs=1; plan.recvs=&recv; plan.num_l2_writes=1; plan.l2_writes=&write;
    }
    if (tile<2) maps_fifo_init(&plan);
    fsync_sync_global(); eu_fsync_wait(WFE);
    if (tile<2) {
        printf("MAPS_INIT_DONE tile=%u cycle=%u\n",tile,maps_read_cycle());
        printf("MAPS_TILE_READY tile=%u cycle=%u\n",tile,maps_read_cycle());
        maps_fifo_run_tile_tokens(&plan,3);
        maps_fifo_flush_experiment_trace(&plan,3);
    }
    eu_clear_events(0xffffffffu);
    fsync_sync_global(); eu_fsync_wait(WFE);
    if (tile==0) {
        for (uint32_t token=0;token<3;++token)
            for (uint32_t i=0;i<64;++i) errors+=outputs[token][i]!=inputs[token][i];
        printf("FIFO_RESULTS_SMOKE errors=%u tokens=3 sleep=%u cycle=%u\n",errors,MAPS_FIFO_EVENT_SLEEP,maps_read_cycle());
    }
    fsync_sync_global(); eu_fsync_wait(WFE);
    return errors?-1:0;
}
