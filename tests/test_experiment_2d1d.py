import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "experiment_2d1d.py"
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_2d1d as d1d  # noqa: E402


def test_exact_budget_and_schedule():
    assert d1d.SOURCE_GLOBAL_UPDATE == 954
    assert d1d.LOCAL_UPDATES == 477
    assert d1d.FINAL_GLOBAL_UPDATE == 1431
    assert d1d.ADDITIONAL_TARGETS == 250_085_376
    assert d1d.FINAL_TARGETS == 750_256_128
    assert d1d.ALPHA == 0.03125
    assert d1d.SCIENTIFIC_LOCAL == (20, 48, 96, 191, 286, 477)


def test_exact_two_stage_geometry():
    assert d1d.stage_for_update(955)["stage"] == "B-R"
    assert d1d.stage_for_update(1050)["stage"] == "B-R"
    assert d1d.stage_for_update(1051)["stage"] == "C-R"
    assert d1d.stage_for_update(1431)["stage"] == "C-R"
    assert d1d.stage_for_update(1050)["windows"] == d1d.B12
    assert d1d.stage_for_update(1051)["windows"] == d1d.C12


def test_pass_cadence_uses_global_update():
    assert d1d.pass_count_for_update(960) == 3
    assert d1d.pass_count_for_update(959) == 2
    assert d1d.pass_count_for_update(992) == 3


def test_residual_equation_keeps_embedding_path():
    tree = ast.parse(SCRIPT.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "residual_make_input")
    assignments = [node for node in ast.walk(function) if isinstance(node, ast.Assign)]
    candidate = next(
        node.value
        for node in assignments
        if any(isinstance(target, ast.Name) and target.id == "candidate" for target in node.targets)
    )
    assert isinstance(candidate, ast.BinOp) and isinstance(candidate.op, ast.Add)
    assert isinstance(candidate.left, ast.Name) and candidate.left.id == "embedding"
    assert isinstance(candidate.right, ast.Name) and candidate.right.id == "alpha_fused"


def test_position_bins_partition_recurrent_suffix():
    positions = []
    for first, last in d1d.POSITION_BINS:
        positions.extend(range(first, last + 1))
    assert positions == list(range(1, 1024))


def test_32_pass_diagnostic_is_forced_no_gradient():
    tree = ast.parse(SCRIPT.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "self_composition")
    contexts = [item.context_expr for node in ast.walk(function) if isinstance(node, ast.With) for item in node.items]
    assert any(
        isinstance(context, ast.Call)
        and isinstance(context.func, ast.Attribute)
        and context.func.attr == "inference_mode"
        for context in contexts
    )
