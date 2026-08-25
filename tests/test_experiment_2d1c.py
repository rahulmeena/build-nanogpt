import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "experiment_2d1c.py"


def load_module():
    spec = importlib.util.spec_from_file_location("experiment_2d1c_tested", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_preregistered_alpha_list():
    module = load_module()
    assert module.ALPHAS == (
        0.0, 0.03125, 0.0625, 0.0827875253078167,
        0.125, 0.25, 0.5, 0.75, 1.0,
    )


def test_exact_d12_and_position_partition():
    module = load_module()
    assert module.D12 == (128, 154, 187, 225, 272, 330, 398, 481, 581, 702, 848, 1024)
    positions = []
    for first, last in module.POSITION_BINS:
        positions.extend(range(first, last + 1))
    assert positions == list(range(1, 1024))


def test_residual_equation_has_mandatory_embedding_path():
    tree = ast.parse(SCRIPT.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "residual_fusion_input")
    assignments = [node for node in ast.walk(function) if isinstance(node, ast.Assign)]
    candidate = next(node.value for node in assignments if any(isinstance(target, ast.Name) and target.id == "candidate" for target in node.targets))
    assert isinstance(candidate, ast.BinOp) and isinstance(candidate.op, ast.Add)
    assert isinstance(candidate.left, ast.Name) and candidate.left.id == "embedding"
    assert isinstance(candidate.right, ast.Name) and candidate.right.id == "alpha_fused"


def test_no_training_calls_or_objects():
    tree = ast.parse(SCRIPT.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"backward", "step", "zero_grad"}
        if isinstance(node, ast.Name):
            assert node.id not in {"Adam", "AdamW", "GradScaler", "configure_optimizer"}


def test_forbidden_counts_zero():
    module = load_module()
    assert set(module.FORBIDDEN_COUNTS.values()) == {0}


def test_stability_labels_follow_bounded_frontier():
    module = load_module()
    assert module.stability_classification({"bounded_positive_alphas": []}) == "ALL NONZERO RESIDUAL ALPHAS ARE EXPANSIVE"
    assert module.stability_classification({"bounded_positive_alphas": [0.03125, 0.0625]}) == "ONLY SMALL RESIDUAL ALPHAS ARE BOUNDED"
    assert module.stability_classification({"bounded_positive_alphas": list(module.ALPHAS[1:7])}) == "WIDE BOUNDED RESIDUAL-ALPHA RANGE"
    assert module.stability_classification({"bounded_positive_alphas": [0.03125, 0.125]}) == "MIXED/NONMONOTONIC RESIDUAL STABILITY"

