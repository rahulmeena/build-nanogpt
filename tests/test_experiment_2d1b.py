import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "experiment_2d1b.py"


def load_module():
    spec = importlib.util.spec_from_file_location("experiment_2d1b_tested", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_factorial_is_exact_and_complete():
    module = load_module()
    assert list(module.CONDITIONS) == ["A", "B", "C", "D"]
    assert module.CONDITIONS["A"]["windows"] == module.C12
    assert module.CONDITIONS["B"]["windows"] == module.C12
    assert module.CONDITIONS["C"]["windows"] == module.D12
    assert module.CONDITIONS["D"]["windows"] == module.D12
    assert [module.CONDITIONS[key]["rho"] for key in module.CONDITIONS] == [0.75, 1.0, 0.75, 1.0]


def test_position_bins_partition_recurrent_positions():
    module = load_module()
    positions = []
    for first, last in module.POSITION_BINS:
        positions.extend(range(first, last + 1))
    assert positions == list(range(1, 1024))


def test_factorial_contrast_arithmetic():
    module = load_module()
    controls = {
        "A": {"real": 1.0, "recurrent_gain": 0.1},
        "B": {"real": 2.0, "recurrent_gain": 0.2},
        "C": {"real": 4.0, "recurrent_gain": 0.4},
        "D": {"real": 8.0, "recurrent_gain": 0.8},
    }
    self_results = {
        name: {"summary": {"max_recurrent_input_rms": value}}
        for name, value in zip("ABCD", (1.0, 2.0, 4.0, 8.0))
    }
    row = module.factorial_contrasts(controls, self_results)["real_validation_loss"]
    assert row["rho_effect_at_C_windows_B_minus_A"] == 1.0
    assert row["window_effect_at_rho_075_C_minus_A"] == 3.0
    assert row["rho_x_window_interaction_D_minus_C_minus_B_plus_A"] == 3.0


def test_scale_decision_tree():
    module = load_module()

    def rows(a, b, c, d):
        return {name: {"summary": {"scale_bounded": value}} for name, value in zip("ABCD", (a, b, c, d))}

    assert module.scale_classification(rows(True, False, True, False)).startswith("RHO=1")
    assert module.scale_classification(rows(True, True, False, False)).startswith("WINDOW")
    assert "INTERACTION" in module.scale_classification(rows(True, True, True, False))
    assert module.scale_classification(rows(True, False, False, False)).startswith("BOTH")
    assert module.scale_classification(rows(True, True, True, True)).startswith("FROZEN")


def test_result_script_has_no_training_calls():
    tree = ast.parse(SCRIPT.read_text())
    forbidden_attributes = {"backward", "step", "zero_grad"}
    forbidden_names = {"Adam", "AdamW", "GradScaler", "configure_optimizer"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in forbidden_attributes
        if isinstance(node, ast.Name):
            assert node.id not in forbidden_names


def test_frozen_counts_are_zero():
    module = load_module()
    assert module.FORBIDDEN_COUNTS
    assert set(module.FORBIDDEN_COUNTS.values()) == {0}

