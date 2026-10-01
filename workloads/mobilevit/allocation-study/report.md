A subsequent [16x16 planning scalability fix](scaling-report.md) replaced exhaustive tile-count enumeration with coarse probes and local refinement. The measured 8x8 plan remains exactly identical. The algorithm description below records the original experiment implementation.

The revised allocation search improved this controlled MobileViT 8x8 run. Mean completion interval across the three available intervals fell from 95,005 to 57,944 cycles (39.0% lower); median interval fell from 103,417 to 57,428 cycles. Active tiles fell from 54 to 26. This is a four-token, instrumented comparison, not a long steady-state throughput measurement or independent validation of the cost model.

Both runs used the same dirty MAGIA SDK checkout, 8x8 simulator, four execution tokens, two token slots, MAPS_TIMINGS=1, stage-latency weight 1, and communication weight 1. The full-mesh control returned exactly the same completion cycles and intervals in both runs. The baseline used the user's existing uncommitted MAPS implementation, including its current DMA submission floor. The candidate used the implementation in allocator.patch on top of that work. Settings and source hashes are in settings.json. The runner now defaults to unit communication weight; this baseline was explicitly run with that same unit setting, rather than the previous runner default of 2.

| Token | Baseline completion | New completion | Baseline interval | New interval |
| --- | ---: | ---: | ---: | ---: |
| 0 | 487,623 | 396,753 | — | — |
| 1 | 561,935 | 449,399 | 74,312 | 52,646 |
| 2 | 665,352 | 513,158 | 103,417 | 63,759 |
| 3 | 772,639 | 570,586 | 107,287 | 57,428 |

Both MAPS runs report one checksum validation mismatch per token. That limits any correctness claim: the result establishes a timing improvement with unchanged mismatch counts, not successful numerical validation. The full-mesh control also reports checksum mismatches in both runs. Raw validation fields are retained in baseline-results.csv and new-results.csv.

The analyzer retains every feasible rectangular logical layout at each tile count. Seeding still uses the smallest L1-feasible count, but layouts remain available to whole-plan evaluation. Every iteration chooses the best globally evaluated single-stage replacement across all feasible counts and layouts, including shrinking and same-count layout changes. Atomic exchanges transfer any number of released donor tiles to another stage, checking both stages' layouts. Stages are reconsidered after each accepted move, and search continues when the mesh is full. Exact metric ties prefer fewer tiles. Stage formation is unchanged. This is deterministic neighborhood search with strict objective descent, not a proof of the global optimum.

The default objective orders stage service estimates from largest to smallest. A stage service estimate is intrinsic stage latency plus its worst virtual tile's accumulated transfer service. Intrinsic latency already sums tile-local phases separated by collective barriers and collective group costs. Transfer service uses the existing per-transfer DMA geometry, submission/setup/publication costs, payload bandwidth, and consumer unpack costs. Unit defaults compare cycles directly; explicit weight arguments remain supported for compatibility. No communication weight or cost estimate was fitted for this result. The existing provisional 2,000-cycle DMA submission floor remains unchanged and still derives from MobileViT traces; these runs do not independently validate it.

| Stage | Role | Baseline tiles / layout | New tiles / layout | Baseline intrinsic + transfers | New intrinsic + transfers |
| --- | --- | --- | --- | ---: | ---: |
| 0 | Group normalization | 12 / 6x2 | 1 / 1x1 | 13,625 + 30,720 = 44,345 | 24,139 + 27,632 = 51,771 |
| 1 | QKV projection | 14 / 14x1 | 10 / 10x1 | 36,258 + 30,589 = 66,847 | 49,025 + 4,616 = 53,641 |
| 2 | Split | 3 / 3x1 | 1 / 1x1 | 6,168 + 36,774 = 42,942 | 16,448 + 32,056 = 48,504 |
| 3 | Softmax | 1 / 1x1 | 1 / 1x1 | 5,642 + 18,012 = 23,654 | 5,642 + 8,010 = 13,652 |
| 4 | Multiply and reduction | 6 / 3x2 | 3 / 3x1 | 23,274 + 12,129 = 35,403 | 43,379 + 6,226 = 49,605 |
| 5 | ReLU | 1 / 1x1 | 1 / 1x1 | 2,412 + 23,024 = 25,436 | 2,412 + 5,024 = 7,436 |
| 6 | Multiply | 8 / 2x4 | 1 / 1x1 | 2,616 + 26,664 = 29,280 | 10,296 + 28,704 = 39,000 |
| 7 | Output projection and add | 9 / 9x1 | 8 / 8x1 | 29,421 + 20,632 = 50,053 | 31,271 + 6,640 = 37,911 |

The new objective scores the baseline bottleneck at 66,847 cycles and the selected plan at 53,641 cycles, both at stage 1. These are virtual allocation estimates, not the old physical placement diagnostic printed as execution_plan_stage_cost (36,258 baseline; 49,025 new). predictions.csv separately records local phase, collective, intrinsic, and transfer cycles. The baseline's own max-based objective scored its virtual stage-1 bottleneck at 36,258 cycles; it is a different metric and should not be compared directly with the new sum.

The actual candidate search grew stage 1 to 10 tiles, stage 7 to 9, and stage 4 to 3, then removed a tile from stage 7. The final 9→8 move kept the worst service estimate at 53,641 and improved the remaining ordered bottlenecks. It used 26 tiles and left 38 unused. The full trajectory and final component diagnostics are in the candidate build.log.

Measured token-3 collective phase maxima sum to 34,507→1,781 cycles in stage 0, 3,258→2,381 in stage 3, and 10,957→8,928 in stage 4. Collectives are included in operation timing, not transition timing. Summing operation maxima over phases gives 57,432→31,831 cycles for stage 0 and 43,551→56,724 for stage 1. The latter increases because stage 1 uses fewer tiles, while its communication fan-out decreases. These phase summaries can use different tiles for different phases and are not completion intervals.

| Edge | Transfers per token, baseline→new | Worst sender's summed service at token 3, baseline→new | Maximum visibility wait at token 3, baseline→new |
| --- | ---: | ---: | ---: |
| 0→1 | 168→10 | 47,843→59,550 | 18,372→17,778 |
| 1→2 | 42→10 | 10,119→4,696 | 18,260→10,319 |
| 2→3 | 3→1 | 3,776→2,495 | 38,749→36,754 |
| 2→4 | 6→3 | 13,357→13,491 | 18,511→4,356 |
| 3→4 | 6→3 | 19,525→8,214 | 48,449→8,074 |
| 2→5 | 3→1 | 5,698→4,939 | 46,449→61,995 |
| 5→6 | 8→1 | 30,106→5,495 | 8,287→5,667 |
| 4→6 | 24→3 | 25,330→3,180 | 56,173→11,428 |
| 6→7 | 72→8 | 44,605→49,785 | 35,340→20,384 |

Total intermediate transfers fall from 332 to 40 per token. Fewer transfers do not guarantee lower worst-sender service: stages 0 and 6 concentrate sends onto one tile. Visibility waits include producer scheduling and arrival delays and are not pure DMA costs. measured-stages.csv and measured-transitions.csv retain these summaries for all four tokens. Original timing CSVs retain each operation and transfer.

The estimator still has substantial errors. At token 3, baseline stage-0 collectives total 34,507 measured cycles versus 1,360 predicted; new stage-0 collectives total 1,781 versus 1,300. New stage-4 collectives measure 8,928 versus 2,700 predicted. New stage-3 intrinsic phases measure 17,104 versus 5,642 predicted. New stage-0 sender service alone measures 59,550 versus 27,632 predicted total transfer service, which also includes input traffic; contention, runtime overhead and timing-window effects remain unmodeled. Conversely, split intrinsic latency is overestimated (16,448 predicted versus 6,132 measured). Stage sums, overlap, FIFO scheduling, physical placement, and trace windows prevent treating the service estimate as exact runtime. These findings were recorded without adjusting the estimates or fitting a weight.

Run directories:

- Baseline: ../results/20261001T075754.896863025Z/
- Candidate: ../results/20261001T080104.265227388Z/
- The earlier failed attempt, 20261001T075725.684966540Z, stopped at a sandbox-blocked SDK rebuild and is excluded.

Reproduction used `MESH_SIZES=8 MAPS_TIMINGS=1 COMMUNICATION_WEIGHT=1.0 workloads/mobilevit/run-experiment.sh` for the baseline. The candidate command additionally set `MAPS_ROOT=/tmp/maps-allocation`, a writable snapshot of the dirty sibling MAPS checkout with the patch applied. Both used the same existing compiler tools and SDK. The implementation is subsequently applied to the sibling MAPS checkout; no commits or pushes are created. To reconstruct prediction components, run `PYTHONPATH=/home/ivan/repos/MAPS /home/ivan/repos/MAPS/.venv/bin/python workloads/mobilevit/allocation-study/predict.py BASELINE_EXECUTION_PLAN CANDIDATE_EXECUTION_PLAN`.

Focused tests cover retained layouts and cache reuse, communication-favorable layout selection including actual graph-I/O transfer compilation, harmful growth with unused capacity, removal from an overgrown seed, recovery through a full-mesh exchange, and combined intrinsic/transfer service. Planner interface snapshots were updated to reflect the revised default policy. Focused verification passed: 35 allocation tests; 154 planner/handoff tests with one SDK-dependent skip; all six MobileViT application tests. The experiment result-parser tests also passed (four cases). Eight pre-existing calibration formula expectations fail against the already-modified calibration.py; both calibration files were preserved rather than changing formulas or expectations to pass this allocator task. Final full-suite verification in the actual sibling MAPS checkout passed: 478 passed, three skipped. The only warning was pytest being unable to write its cache in the read-only sandbox. Both repository diffs pass git diff --check, and the runner passes bash -n. No commits or pushes were made.
