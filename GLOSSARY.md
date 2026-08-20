# Domain glossary

**Workload**: One model and its related, independently runnable Experiments.

**Experiment**: One complete command that reproduces a paper figure or table.

**Execution Token**: One independent set of Runtime Input values processed through an
Execution Plan.

**Token Slot**: One reusable storage position for an Execution Token's runtime Tensor
values.

**Completion Cycle**: Cycles elapsed from the Experiment's common pre-execution boundary
until one Execution Token's final output is globally valid.

**Completion Interval**: The difference between consecutive Completion Cycles. The first
Execution Token has no Completion Interval.
