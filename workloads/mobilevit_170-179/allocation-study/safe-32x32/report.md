# Bounded 32×32 MobileViT attempt (2026-10-01)

The full experiment did not complete. The allocator and application generation succeeded, but GVSoC configuration generation exceeded the 4 GiB cgroup memory limit. The kernel killed only processes in the experiment cgroup. WSL remained responsive, with approximately 7.8 GiB available after termination. No simulation completion measurements were produced.

Experiment directory: `workloads/mobilevit/results/20261001T092930.842646048Z`.
Settings: 32×32, two execution tokens, two token slots, MAPS_TIMINGS=1, communication weight 1.0. Compiler parallelism was limited to two jobs. Existing source changes were preserved.

Enforced properties: MemoryMax=4G, MemorySwapMax=0, CPUQuota=200%, TasksMax=128, RuntimeMaxSec=1200, KillMode=control-group, OOMPolicy=kill, LimitCORE=0. These cover the process tree, including configuration, compilers, and simulator.

The initial service did not inherit the interactive Python environment and failed on a missing `prettytable` import. That attempt peaked at approximately 322 MiB. A resumed attempt reused the generated application and execution plan and prepended the experiments virtual environment to PATH. Its GVSoC configuration process reached approximately 4 GiB RSS and was killed by the cgroup OOM mechanism. The resumed service's recorded MemoryPeak (1.4 GiB) missed the final spike; kernel-oom.log is the evidence for the actual limit hit.

This reproduces excessive configuration memory under a controlled bound. It does not establish the precise root cause, or prove the earlier unbounded WSL crash was caused by this process. Raising the limit or adding swap was not attempted. Neither attempt is still running.

See simulator-setup.log, kernel-oom.log, and limits.txt. No SDK/MAPS source fix was made in this investigation. SDK build artifacts and the selected mesh configuration were updated by the ordinary build commands.

## User-requested 6 GiB retry

A second bounded setup attempt used MemoryMax=6G with the same no-swap, two-CPU, 20-minute settings. It reused the successful 32×32 plan/application, fixed the service PATH to include the experiments Python environment and `/opt/riscv/bin`, and selected the existing LLVM 20 installation. The prior 16×16 measurement was stopped before launching this attempt.

This attempt also ended with `Result=oom-kill` while generating the GVSoC configuration, before simulation. The kernel report confirms the cgroup limit; systemd's cached peak was again below the final spike. WSL remained responsive with approximately 7.6 GiB available after termination. No further limit increase was attempted.

The preceding 16×16 run completed GVSoC configuration and simulator build, then stopped on the service environment's missing RISC-V compiler path. GNU time recorded a maximum process RSS of 1,083,044 KiB (1.033 GiB) for that initial attempt. The resumed 16×16 attempt reached simulation and was explicitly stopped to avoid overlap with this retry; it did not yield a completed whole-run peak. Multiplying the successful 16×16 setup peak by four gives 4.132 GiB, whereas 32×32 exceeded even a 6 GiB aggregate cgroup limit. This comparison suggests more than simple linear memory scaling at setup, but the GNU time process RSS and aggregate cgroup memory are different metrics and do not establish an exact ratio.
