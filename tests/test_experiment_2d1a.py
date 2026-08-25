import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "experiment_2d1a.py"


def load_module():
    spec = importlib.util.spec_from_file_location("experiment_2d1a", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_protocol_constants():
    module = load_module()
    assert module.FROZEN_2D1_COMMIT == "2d4be75e0568d5e2df80b8963c1260db4982ca70"
    assert module.COMMON_C == (256, 290, 329, 373, 423, 481, 545, 619, 702, 796, 903, 1024)
    assert module.COMMON_B == (512, 545, 581, 618, 658, 702, 747, 796, 848, 903, 962, 1024)
    assert module.R_STOP == 10.0 * module.R_STAGE_A
    assert module.REPEATED_PASSES == 32


def test_no_training_counters_are_zero():
    module = load_module()
    assert module.FORBIDDEN_COUNTS == {
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "gradscaler_objects": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "training_targets": 0,
    }


def test_position_bins_exact():
    module = load_module()
    assert module.POSITION_BINS == (
        (1, 64), (65, 128), (129, 256), (257, 512),
        (513, 768), (769, 896), (897, 1023),
    )
