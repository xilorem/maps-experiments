"""Reconstruct cycle components for baseline and candidate execution plans.

Run with the updated MAPS on PYTHONPATH, passing the two execution-plan.json
paths in baseline, candidate order. No allocation search or simulation is rerun.
"""

import csv
import json
import sys
from pathlib import Path

from maps.graph import import_onnx_model, run_graph_rewrites
from maps.operations.collective import AllReducePayload
from maps.planning.allocation.candidates import StageCandidateAnalyzer
from maps.planning.allocation.selection import evaluate_candidate_selection
from maps.planning.stages import form_stages, virtual_submesh
from maps.target import SpecializationOptions, magia_v3


def collective_cycles(plan):
    submesh = virtual_submesh(plan)
    total = 0
    for index, (node, layouts, device) in enumerate(
        zip(plan.nodes, plan.node_output_layouts, plan.device_names)
    ):
        if not isinstance(node.payload, AllReducePayload):
            continue
        costs = []
        for group in plan.virtual_collective_groups[index]:
            participants = tuple(
                tile for tile in submesh.tiles if tile.tile_id in group.virtual_tile_ids
            )
            representative = participants[0]
            work = node.payload.build_tile_work(
                output_layouts=layouts, tile=representative,
            )
            costs.append(node.payload.cost_model.collective_cost(
                work, representative, representative.device_by_name(device), participants,
            ))
        total += max(costs, default=0)
    return total


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: predict.py BASELINE_PLAN CANDIDATE_PLAN")
    model = Path(__file__).resolve().parents[1] / "model/mobilevit-nodes-170-179.onnx"
    mesh = magia_v3.build_mesh(width=8, height=8)
    rewritten = run_graph_rewrites(import_onnx_model(model))
    graph = magia_v3.specialize(rewritten, mesh, SpecializationOptions()).model.graph
    formation = form_stages(graph)
    analyzer = StageCandidateAnalyzer(formation, mesh, frozenset(graph.initializers), 2)
    rows = []
    for label, filename in zip(("baseline", "new"), sys.argv[1:]):
        saved = json.loads(Path(filename).read_text())
        selected = {}
        for stage, saved_stage in enumerate(saved["stages"]):
            count = len(saved_stage["virtual_to_physical"])
            expected = [
                [(o["layout"]["logical_width"], o["layout"]["logical_height"])
                 for o in layer["outputs"]]
                for layer in saved_stage["layers"]
            ]
            matches = [
                candidate for candidate in analyzer.candidates(stage, count)
                if [[(layout.logical_width, layout.logical_height) for layout in outputs]
                    for outputs in candidate.plan.node_output_layouts] == expected
            ]
            assert len(matches) == 1, (stage, count, len(matches))
            selected[stage] = matches[0]
        evaluation = evaluate_candidate_selection(selected, mesh, 1, 1, graph)
        for stage, candidate in selected.items():
            plan = candidate.plan
            collective = collective_cycles(plan)
            breakdown = evaluation.stage_breakdowns[stage]
            rows.append(dict(
                variant=label,
                stage=stage,
                nodes="+".join(node.name for node in plan.nodes),
                tiles=plan.tile_count,
                logical_width=plan.logical_shape[0],
                logical_height=plan.logical_shape[1],
                local_phase_cycles=candidate.stage_latency - collective,
                collective_cycles=collective,
                intrinsic_cycles=candidate.stage_latency,
                transfer_cycles=breakdown.communication_cycles,
                service_cycles=int(breakdown.weighted_bottleneck),
            ))
    output = Path(__file__).resolve().parent / "predictions.csv"
    with output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
