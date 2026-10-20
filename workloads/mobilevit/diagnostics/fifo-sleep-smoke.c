/* GVSoC-only FIFO wake-up regression, run on a 2x2 mesh.
 * The forwarding DMA must be published while tile 1 waits for its next input:
 * 0 -> 1 -> 2 -> acknowledgement to 0 -> second input to 1.
 * A missed DMA wake or FIFO wake deadlocks this chain. */
#include "tile.h"
#include "fsync.h"
#include "utils/maps_utils_v2.h"
#include "utils/performance_utils.h"

#define BYTES 512u

static fifo_msg_t receive(fifo_tile_plan_t *plan,
                          maps_fifo_token_state_t *states)
{
    fifo_recv_desc_t recv = {.producer_idx = 0u};
    fifo_msg_t message;
    for (;;) {
        if (states != 0)
            maps_fifo_reap_completed_sends(plan, states);
        if (fifo_peek(plan->hartid, &message))
            return message;
#if MAPS_FIFO_EVENT_SLEEP
        maps_fifo_sleep_recv(plan, &recv, states);
#endif
    }
}

int main(void)
{
    const uint32_t hartid = get_hartid();
    perf_init();
    idma_init();
    eu_init();
    eu_clear_events(0xffffffffu);
    eu_idma_init( 0u);
    eu_fsync_init( 0u);
    fsync_init();
    fifo_init(hartid, 1u, 2u, BYTES);
    fifo_tile_plan_t plan = {.hartid = hartid, .num_token_slots = 1u};
    tensor_sub_slice_t slice = {
        .rank = 1u, .num_elems = BYTES,
        .dims = {{.start = 0u, .length = BYTES, .stride = 1u}}};
    uint32_t *source = (uint32_t *)(get_l1_base(hartid) + 0x30000u);
    for (uint32_t i = 0; i < BYTES / 4u; ++i)
        source[i] = i * 37u + 11u;
    fsync_sync_global();
    eu_fsync_wait( WFE);
    uint32_t errors = 0u;
    for (uint32_t iteration = 0; iteration < 8u; ++iteration) {
        if (hartid == 0u) {
            fifo_push_cpu(1u, 0u, source, &slice, iteration, 1u);
            fifo_msg_t ack = receive(&plan, 0);
            errors += ack.tag != iteration;
            fifo_release(hartid, ack.src);
            fifo_push_cpu(1u, 0u, source, &slice, iteration + 100u, 1u);
        } else if (hartid == 1u) {
            /* First publication can precede the receive and its sleep setup. */
            if ((iteration & 1u) == 0u)
                wait_nop(4096u);
            fifo_msg_t first = receive(&plan, 0);
            errors += first.tag != iteration;
            for (uint32_t i = 0; i < BYTES / 4u; ++i)
                errors += ((volatile uint32_t *)first.data_ptr)[i] != source[i];
            fifo_release(hartid, first.src);
            fifo_push_req_t req = {
                .target_hartid = 2u, .producer_idx = 0u,
                .src_base_addr = (uint32_t)source, .src = &slice,
                .desc = &slice, .tag = iteration, .elem_bytes = 1u};
            maps_fifo_pending_send_t pending = {0};
            maps_fifo_token_state_t states[1] = {{.sends = &pending, .num_sends = 1u}};
            errors += fifo_push_async_start( &req, &pending.push) != 0;
            fifo_msg_t second = receive(&plan, states);
            errors += second.tag != iteration + 100u;
            errors += states[0].sends != 0;
            for (uint32_t i = 0; i < BYTES / 4u; ++i)
                errors += ((volatile uint32_t *)second.data_ptr)[i] != source[i];
            fifo_release(hartid, second.src);
        } else if (hartid == 2u) {
            fifo_msg_t forwarded = receive(&plan, 0);
            errors += forwarded.tag != iteration;
            for (uint32_t i = 0; i < BYTES / 4u; ++i)
                errors += ((volatile uint32_t *)forwarded.data_ptr)[i] != source[i];
            fifo_release(hartid, forwarded.src);
            fifo_push_cpu(0u, 0u, source, &slice, iteration, 1u);
        }
    }
    printf("FIFO_SLEEP_SMOKE tile=%u sleep=%u errors=%u cycle=%u\n",
           hartid, MAPS_FIFO_EVENT_SLEEP, errors, maps_read_cycle());
    if (errors)
        maps_trap();
    fsync_sync_global();
    eu_fsync_wait( WFE);
    return 0;
}
