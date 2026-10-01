The 16x16 planner was doing excessive allocation work, and buffered diagnostics hid its progress. A profile of the real MobileViT allocation showed candidate construction and repeated tensor slice lookup dominating the first 15 seconds, before any move was accepted. This was not evidence of a deadlock. The partial profile is retained in scaling-before-profile.txt.

Two deterministic regressions reproduced the avoidable work before the fix:

- Slicing all 16 tiles resolved mesh tiles 256 times for a virtual submesh and 768 times for a physical submesh. The test bounds resolution to at most one pass over the submesh.
- A single allocation sweep on a 256-tile mesh requested all 256 counts, even when only the seed was feasible. The test bounds count probes to a logarithmic neighborhood while retaining small counts, power-of-two counts, and the full budget.

The fix caches tile objects and logical ownership ordinals on immutable submeshes. Tensor slicing looks up the ordinal directly, preserving virtual tile order and physical row-major order. Count search now checks powers of two and their neighbors, half/double the current count, adjacent current counts, and budget endpoints. Every feasible layout at each probed count is still evaluated against the whole-plan objective. Removal, same-count layout changes, tile exchanges, and unused capacity remain supported. This changes exhaustive count enumeration into a scalable local search; it can miss an isolated optimum outside its count neighborhood. The cycle model and objective were not changed.

Allocation diagnostics now flush immediately and print the sweep, stage, probed counts, and exchange phase before doing that work. The build therefore reports progress before the first accepted move.

Verification:

- Full MAPS suite in the writable snapshot: 481 passed, three skipped, in 67.38 seconds.
- Planner tests after applying the fix to the actual sibling MAPS checkout: 156 passed. Pytest only warned that the sandbox could not write its cache.
- The new 8x8 execution-plan JSON is exactly equal to the plan measured in the earlier allocation experiment. Its selected counts, layouts, placement, and transitions are unchanged.
- The 16x16 CLI plan completed with counts `[1, 10, 1, 1, 3, 1, 1, 8]`, using 26 tiles. The same two-token/two-slot 16x16 application build used by the reported command completed in 49.22 seconds in the writable snapshot while other verification was running, with peak RSS of 86,388 KB. This is a planning/build check, not a 16x16 simulator result.

The final build through `make build` in the actual sibling MAPS checkout completed in **36.38 seconds**, with two execution tokens, two token slots, and 26 active tiles. Peak RSS was 86,308 KB. Its diagnostics and timing are recorded in scaling-installed-build.log. The patch applies after allocator.patch and is retained as scaling.patch. Existing user changes in both repositories were preserved; no commits or pushes were created.

An already-running Python planner has loaded the old implementation. Cancel that run and restart the experiment to use the fix. No change to token or simulator settings is required.
