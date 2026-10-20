#include "maps_depthwise_bench_task_bin.h"
#include "fsync.h"
#include "redmule.h"
#include "tile.h"
#include "utils/maps_operations.h"
#include "utils/printf.h"
@DECLARATIONS@
typedef struct {
    const char *name;
    const uint16_t *input, *weight, *bias, *golden;
    uint32_t ih, iw, oh, ow, ic, oc, multiplier, stride, stride_w, dilation, pad, tokens;
    uint32_t column_stride, input_row, output_row, origin_h, origin_w, output_origin_h, output_origin_w, input_c, weight_c, has_bias, scratch;
    uint32_t kernel_h, kernel_w, pad_w;
} benchmark_case_t;
static const benchmark_case_t cases[] = {@CASES@};
static volatile uint32_t errors[4] = {1, 1, 1, 1};

static subslice_desc_t sub(uint32_t id, uint32_t n, uint32_t c, uint32_t h, uint32_t w,
                           uint32_t batch, uint32_t channel, uint32_t row, uint32_t column)
{
    return (subslice_desc_t){.slice_id=id, .rank=4, .shape={n,c,h,w},
        .elem_type=ELEM_F16, .elem_bytes=2, .strides_bytes={batch,channel,row,column}};
}

int main(void)
{
    const uint32_t tile = get_hartid(), base = get_l1_base(tile);
    idma_init(); redmule_init(); fsync_init(); eu_init();
    eu_clear_events(0xffffffffu);
    eu_fsync_init(0); eu_idma_init(0); eu_redmule_init(0);
    maps_operation_runtime_t runtime;
    runtime.kernel_abi_version = MAPS_KERNEL_ABI_VERSION;
    runtime.task_bundle_abi_version = MAPS_TASK_BUNDLE_ABI_VERSION;
    runtime.spatz_binary_start = SPATZ_BINARY_START;
    runtime.add_fp16_task = 0;
    runtime.mul_bcast_fp16_task = 0;
    runtime.matmul_fp16_task = 0;
    runtime.relu_fp16_task = 0;
    runtime.maxpool_fp16_task = 0;
    runtime.sigmoid_fp16_task = 0;
    runtime.depthwise_conv_fp16_task = DEPTHWISE_CONV_FP16_SPATZ_TASK;
    runtime.transpose_fp16_task = 0;
    runtime.softmax_exp_fp16_task = 0;
    runtime.group_reduce_fp16_task = 0;
    runtime.group_centered_reduce_fp16_task = 0;
    runtime.group_normalize_fp16_task = 0;
    runtime.reducesum_fp16_task = 0;
    runtime.reducemax_fp16_task = 0;
    runtime.binary_bcast_fp16_task = 0;
    runtime.spatz_params = (void *)(base+MAPS_OPERATION_TASK_SCRATCH_OFFSET);
    runtime.spatz_params_bytes = MAPS_OPERATION_TASK_SCRATCH_BYTES;
    runtime.spatz_initialized = 0;
    runtime.background = 0;
    runtime.background_ctx = 0;

    if (maps_operation_runtime_init(&runtime)) return -1;
    errors[tile]=0;
    fsync_sync_global(); eu_fsync_wait(WFE);
    for (uint32_t ci=0; tile<@ACTIVE_TILES@ && ci<@CASE_COUNT@; ++ci) {
        const benchmark_case_t *b=&cases[tile*@CASE_COUNT@+ci];
        slice_desc_t slices[4] = {
            {.slice_id=0,.l1_offset_bytes=0x20000,.num_slots=1,.full_offset={0,b->input_c,b->origin_h,b->origin_w}},
            {.slice_id=1,.l1_offset_bytes=0x10000,.num_slots=1,.full_offset={b->weight_c}},
            {.slice_id=2,.l1_offset_bytes=0x12000,.num_slots=1},
            {.slice_id=3,.l1_offset_bytes=0x40000,.num_slots=1,.full_offset={0,b->weight_c,b->output_origin_h,b->output_origin_w}}};
        tile_plan_t plan = {.hartid=tile,.l1_data_base=base,.num_slices=4,.slices=slices,.num_token_slots=1};
        op_desc_t op = {.kind=OP_DEPTHWISE_CONV,.num_inputs=b->has_bias?3:2,.num_outputs=1,
            .params={b->stride,b->stride_w,b->dilation,b->dilation,b->pad,b->pad_w,b->multiplier}};
        op.inputs[0]=sub(0,1,b->ic,b->ih,b->iw,b->ic*b->ih*b->input_row,b->ih*b->input_row,b->input_row,b->column_stride);
        const uint32_t weight_bytes=b->kernel_h*b->kernel_w*2u;
        op.inputs[1]=sub(1,b->oc,1,b->kernel_h,b->kernel_w,weight_bytes,weight_bytes,b->kernel_w*2u,2);
        op.inputs[2]=(subslice_desc_t){.slice_id=2,.rank=1,.shape={b->oc},.elem_type=ELEM_F16,.elem_bytes=2,.strides_bytes={2}};
        op.outputs[0]=sub(3,1,b->oc,b->oh,b->ow,b->oc*b->oh*b->output_row,b->oh*b->output_row,b->output_row,b->column_stride);
        runtime.spatz_params_bytes=b->scratch;
        maps_idma_copy_1d_ticketed(0,(uint32_t)b->weight,base+0x10000,b->oc*weight_bytes);
        maps_idma_copy_1d_ticketed(0,(uint32_t)b->bias,base+0x12000,b->oc*2);
        for (uint32_t token=0; token<b->tokens; ++token) {
            const uint32_t input_bytes=b->ic*b->ih*b->input_row;
            maps_idma_copy_1d_ticketed(0,(uint32_t)b->input+token*input_bytes,base+0x20000,input_bytes);
            const uint32_t output_bytes=b->oc*b->oh*b->output_row;
            for (uint32_t i=0; i<output_bytes/2; ++i) ((uint16_t *)(base+0x40000))[i]=0x3555;
            printf("DW_BEGIN tile=%u case=%u token=%u name=%s cycle=%u\n",tile,ci,token,b->name,maps_read_cycle());
#if defined(MAPS_DEPTHWISE_PROFILE) && MAPS_DEPTHWISE_PROFILE
            for (uint32_t i=0;i<6;++i) maps_depthwise_profile[i]=0;
#endif
            const uint32_t start=maps_read_cycle();
            const int result=maps_execute_operation(&plan,&op,0,&runtime);
            const uint32_t cycles=maps_read_cycle()-start;
#if defined(MAPS_DEPTHWISE_PROFILE) && MAPS_DEPTHWISE_PROFILE
            printf("DW_PROFILE tile=%u case=%u token=%u prepare=%u spatz=%u copy=%u gathers=%u launches=%u stores=%u\n",
                tile,ci,token,maps_depthwise_profile[0],maps_depthwise_profile[1],maps_depthwise_profile[2],
                maps_depthwise_profile[3],maps_depthwise_profile[4],maps_depthwise_profile[5]);
            const depthwise_conv_fp16_spatz_params_t *dp=runtime.spatz_params;
            if (dp->vector_profile) {
                const volatile uint32_t *vp=(const volatile uint32_t *)dp->vector_profile;
                printf("DW_VECTOR_PROFILE tile=%u case=%u token=%u geometry=%u input=%u fma=%u output=%u\n",
                    tile,ci,token,vp[0],vp[1],vp[2],vp[3]);
            }
#endif
            uint32_t bad=0;
            for (uint32_t c=0;c<b->oc;++c)
                for (uint32_t y=0;y<b->oh;++y)
                    for (uint32_t x=0;x<b->ow;++x) {
                        const uint16_t actual=*(uint16_t *)(base+0x40000+c*b->oh*b->output_row+y*b->output_row+x*b->column_stride);
                        const uint16_t expected=b->golden[((token*b->oc+c)*b->oh+y)*b->ow+x];
                        if (actual!=expected && ((actual|expected)&0x7fff)) {
                            if (bad<3) printf("DW_MISMATCH tile=%u case=%u token=%u c=%u y=%u x=%u actual=%04x expected=%04x\n",tile,ci,token,c,y,x,actual,expected);
                            ++bad;
                        }
                    }
            /* Ensure row/column gaps are not overwritten by the DMA store. */
            for (uint32_t i=0;i<output_bytes/2;++i) {
                const uint32_t within=(i*2)%b->output_row;
                if ((within/b->column_stride>=b->ow || within%b->column_stride) && ((uint16_t *)(base+0x40000))[i]!=0x3555) ++bad;
            }
            printf("DW_CHECK tile=%u case=%u token=%u result=%d cycles=%u elements=%u errors=%u im2col=%u\n",
                tile,ci,token,result,cycles,b->oc*b->oh*b->ow,bad,((depthwise_conv_fp16_spatz_params_t *)runtime.spatz_params)->im2col!=0);
            errors[tile]+=bad+(result!=0);
        }
    }
    fsync_sync_global(); eu_fsync_wait(WFE);
    uint32_t total=0; for (uint32_t i=0;i<4;++i) total+=errors[i];
    if (!tile) printf("DW_RESULT errors=%u\n",total);
    spatz_clk_dis();
    return total?-1:0;
}
