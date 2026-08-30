"""Fail-closed static scope tests for the Experiment 2D5C implementation.

These tests intentionally do not import the scientific modules.  The local
control environment may not have Torch, and importing a training driver is an
unnecessary way to verify immutable protocol constants and command surfaces.
Instead, JSON, AST, and narrowly scoped source checks bind the configuration,
driver, core, analysis artifact contract, and RunPod guard together.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "exp2d5c_fixed_writer_b3_b5_w2_matched_100m.json"
DRIVER_PATH = ROOT / "scripts" / "experiment_2d5c.py"
CORE_PATH = ROOT / "scripts" / "experiment_2d5c_core.py"
ANALYSIS_PATH = ROOT / "scripts" / "experiment_2d5c_analysis.py"
GUARD_PATH = ROOT / "scripts" / "experiment_2d5c_runpod_guard.py"

SOURCE_SHA256 = "de80d0886a42e4142fa8b30d27eae4302a298bb207961b593f9401e908faaf7b"
CONTROL_SHA256 = "e108e47b68a13b368bbd6a27bd1472b9740613a9d03896e900e158bb3ed708a8"
PROTOCOL_SHA256 = "eae21daf3859eb70d3342261d2891fbb4b2114235b1a87450eb2b4d201183b70"
FINGERPRINT_C = "019d822dd89986c269e985fba8d1277a15d476dd73a0dac0d8c35e07e7315c12"
FINGERPRINT_FIXED = "be345a9fe3b486f601c3af1564ce90f51de51c84daf6e89885126b094adfaac2"
POD_ID = "rvgztsr0azrwyo"
POD_NAME = "happy_apricot_stork"
VOLUME_ID = "yhzyb27fb5"
FINAL_PHRASE = "STOPPED AFTER C AT EXACTLY 191 UPDATES / 100,139,008 TARGETS"

CONTROLS = (
    "all_real",
    "b3_off",
    "b3_shuffled",
    "b5_off",
    "b5_shuffled",
    "b3_b5_off",
    "b3_b5_shuffled",
)

DRIVER_COMMANDS = {
    "prepare",
    "preflight",
    "train",
    "evaluate",
    "seal-final",
    "memory-audit",
    "representation-diagnostics",
    "analyze",
    "render-report",
    "postflight-audit",
}
GUARD_COMMANDS = {"preflight", "trigger", "stop", "watchdog"}

REQUIRED_ARTIFACTS = (
    "SCOPE_LOCK.json",
    "SOURCE_PROVENANCE.json",
    "FIXED_CONTROL_PROVENANCE.json",
    "ENVIRONMENT_MANIFEST.json",
    "ARCHITECTURE_MANIFEST_C.json",
    "DATA_REPLAY_LEDGER.jsonl",
    "DATA_REPLAY_AUDIT.json",
    "PANEL_MANIFEST_CORE.json",
    "PANEL_MANIFEST_LARGE.json",
    "PREFLIGHT_TESTS.json",
    "DISPOSABLE_SMOKE_REPORT.json",
    "INITIAL_GEOMETRY_SHOCK.json",
    "TRAINING_LOG.jsonl",
    "MILESTONE_CHECKPOINTS.json",
    "MIDPOINT_RESTART_AUDIT.json",
    "TRUE_INCREMENTAL_LONGITUDINAL_CORE.json",
    "REPRESENTATION_PRESSURE_DIAGNOSTICS.json",
    "LARGE_FINAL_PER_SEQUENCE_LOSSES.json",
    "LARGE_FINAL_BOOTSTRAP.json",
    "BF16_PERSISTENT_STATE_AUDIT.json",
    "FINAL_CHECKPOINT_PROVENANCE.json",
    "FINAL_AUDIT.json",
    "EXPERIMENT_2D5C_FINAL_REPORT.md",
)


def read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


DRIVER_TEXT = read_source(DRIVER_PATH)
CORE_TEXT = read_source(CORE_PATH)
ANALYSIS_TEXT = read_source(ANALYSIS_PATH)
GUARD_TEXT = read_source(GUARD_PATH)
DRIVER_TREE = ast.parse(DRIVER_TEXT, filename=str(DRIVER_PATH))
CORE_TREE = ast.parse(CORE_TEXT, filename=str(CORE_PATH))
ANALYSIS_TREE = ast.parse(ANALYSIS_TEXT, filename=str(ANALYSIS_PATH))
GUARD_TREE = ast.parse(GUARD_TEXT, filename=str(GUARD_PATH))


def safe_eval(node: ast.AST, environment: dict[str, object] | None = None):
    """Evaluate only the small literal expression subset used by constants."""

    environment = {} if environment is None else environment
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in environment:
            raise ValueError(node.id)
        return environment[node.id]
    if isinstance(node, ast.Tuple):
        return tuple(safe_eval(item, environment) for item in node.elts)
    if isinstance(node, ast.List):
        return [safe_eval(item, environment) for item in node.elts]
    if isinstance(node, ast.Set):
        return {safe_eval(item, environment) for item in node.elts}
    if isinstance(node, ast.Dict):
        return {
            safe_eval(key, environment): safe_eval(value, environment)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = safe_eval(node.operand, environment)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
        left = safe_eval(node.left, environment)
        right = safe_eval(node.right, environment)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        return left * right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in {"MappingProxyType", "_ImmutableJSONDict", "dict", "tuple", "frozenset"}:
            if len(node.args) != 1 or node.keywords:
                raise ValueError(node.func.id)
            value = safe_eval(node.args[0], environment)
            if node.func.id in {"MappingProxyType", "_ImmutableJSONDict", "dict"}:
                return dict(value)
            if node.func.id == "tuple":
                return tuple(value)
            return frozenset(value)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and isinstance(node.func.value, ast.Constant)
        and isinstance(node.func.value.value, str)
        and len(node.args) == 1
        and not node.keywords
    ):
        return node.func.value.value.join(safe_eval(node.args[0], environment))
    raise ValueError(ast.dump(node, include_attributes=False))


def literal_bindings(tree: ast.Module) -> dict[str, object]:
    bindings: dict[str, object] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        value_node = statement.value
        if value_node is None:
            continue
        try:
            value = safe_eval(value_node, bindings)
        except (ValueError, TypeError, KeyError):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                bindings[target.id] = value
    return bindings


DRIVER_CONSTANTS = literal_bindings(DRIVER_TREE)
CORE_CONSTANTS = literal_bindings(CORE_TREE)
ANALYSIS_CONSTANTS = literal_bindings(ANALYSIS_TREE)
GUARD_CONSTANTS = literal_bindings(GUARD_TREE)


def function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    rows = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(rows) != 1:
        raise AssertionError(f"expected one function {name}, found {len(rows)}")
    return rows[0]


def class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    rows = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    if len(rows) != 1:
        raise AssertionError(f"expected one class {name}, found {len(rows)}")
    return rows[0]


def source_of(path_text: str, node: ast.AST) -> str:
    source = ast.get_source_segment(path_text, node)
    if source is None:
        raise AssertionError("AST node has no source segment")
    return source


def parser_commands(tree: ast.Module) -> set[str]:
    result = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_parser" or not node.args:
            continue
        value = safe_eval(node.args[0])
        if not isinstance(value, str):
            raise AssertionError("nonliteral parser command")
        result.add(value)
    return result


def argument_call(tree: ast.Module, receiver: str, option: str) -> ast.Call:
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if (
            node.func.attr == "add_argument"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == receiver
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == option
        ):
            matches.append(node)
    if len(matches) != 1:
        raise AssertionError(
            f"expected one {receiver}.add_argument({option!r}), found {len(matches)}"
        )
    return matches[0]


def keyword_value(call: ast.Call, name: str):
    matches = [keyword.value for keyword in call.keywords if keyword.arg == name]
    if len(matches) != 1:
        raise AssertionError(f"expected one keyword {name}")
    return safe_eval(matches[0], DRIVER_CONSTANTS)


class Experiment2D5CStaticTests(unittest.TestCase):
    def test_all_scientific_sources_parse_without_importing_gpu_modules(self):
        for path, source in (
            (DRIVER_PATH, DRIVER_TEXT),
            (CORE_PATH, CORE_TEXT),
            (ANALYSIS_PATH, ANALYSIS_TEXT),
            (GUARD_PATH, GUARD_TEXT),
        ):
            compile(source, str(path), "exec", ast.PyCF_ONLY_AST)

    def test_config_is_exact_one_arm_c_scope_and_arithmetic(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["experiment"], "2D5C")
        self.assertEqual(config["protocol_sha256"], PROTOCOL_SHA256)
        self.assertEqual(config["scope"]["newly_trained_arms"], ["C"])
        self.assertEqual(config["scope"]["prohibited_arms"], ["A", "B", "Fixed"])
        self.assertIs(config["scope"]["continuation_beyond_local_update_191"], False)
        self.assertIs(config["scope"]["continuation_to_250m"], False)
        self.assertEqual(config["scope"]["fixed_control_optimizer_steps"], 0)
        self.assertEqual(config["source"]["checkpoint_sha256"], SOURCE_SHA256)
        self.assertEqual(config["fixed_control"]["checkpoint_sha256"], CONTROL_SHA256)

        training = config["training"]
        self.assertEqual(training["local_updates"], 191)
        self.assertEqual(training["targets_per_update"], 524_288)
        self.assertEqual(training["new_targets"], 100_139_008)
        self.assertEqual(training["local_updates"] * training["targets_per_update"], training["new_targets"])
        self.assertEqual(training["final_global_update"], 2_099)
        self.assertEqual(training["final_cumulative_targets"], 1_100_480_512)
        self.assertEqual(training["milestones"], [48, 96, 144, 191])
        self.assertEqual(training["mandatory_fresh_process_restart"], 96)
        self.assertEqual(config["fixed_control"]["optimizer_steps_in_2d5c"], 0)

        infrastructure = config["infrastructure"]
        self.assertEqual(infrastructure["pod_id"], POD_ID)
        self.assertEqual(infrastructure["pod_name"], POD_NAME)
        self.assertEqual(infrastructure["volume_id"], VOLUME_ID)
        self.assertIs(infrastructure["delete_pod"], False)
        self.assertIs(infrastructure["delete_volume"], False)

    def test_driver_constants_lineage_milestones_and_checkpoint_names_are_exact(self):
        expected = {
            "SOURCE_SHA256": SOURCE_SHA256,
            "CONTROL_SHA256": CONTROL_SHA256,
            "LOCAL_UPDATES": 191,
            "LOCAL_TARGETS": 100_139_008,
            "FINAL_GLOBAL_UPDATE": 2_099,
            "FINAL_CUMULATIVE_TARGETS": 1_100_480_512,
            "RESTART_LOCAL_UPDATE": 96,
            "MILESTONES": (48, 96, 144, 191),
            "POD_ID": POD_ID,
            "POD_NAME": POD_NAME,
            "VOLUME_ID": VOLUME_ID,
        }
        for name, value in expected.items():
            self.assertEqual(DRIVER_CONSTANTS[name], value, name)

        targets = {
            48: 1_025_507_328,
            96: 1_050_673_152,
            144: 1_075_838_976,
            191: 1_100_480_512,
        }
        self.assertEqual(DRIVER_CONSTANTS["MILESTONE_TARGETS"], targets)
        names = {
            update: f"scientific_cumulative_{targets[update]:012d}.pt"
            for update in (48, 96, 144, 191)
        }
        self.assertEqual(
            names,
            {
                48: "scientific_cumulative_001025507328.pt",
                96: "scientific_cumulative_001050673152.pt",
                144: "scientific_cumulative_001075838976.pt",
                191: "scientific_cumulative_001100480512.pt",
            },
        )
        checkpoint_source = source_of(DRIVER_TEXT, function_node(DRIVER_TREE, "checkpoint_name"))
        self.assertIn('f"scientific_cumulative_{MILESTONE_TARGETS[int(local_update)]:012d}.pt"', checkpoint_source)
        train_source = source_of(DRIVER_TEXT, function_node(DRIVER_TREE, "run_train"))
        self.assertIn("if local_update in MILESTONES:", train_source)
        self.assertIn("checkpoint_name(local_update)", train_source)

    def test_training_entrypoint_is_only_c_and_has_only_two_segments(self):
        self.assertEqual(parser_commands(DRIVER_TREE), DRIVER_COMMANDS)
        arm = argument_call(DRIVER_TREE, "train", "--arm")
        end = argument_call(DRIVER_TREE, "train", "--end-local-update")
        self.assertEqual(keyword_value(arm, "choices"), ("C",))
        self.assertEqual(keyword_value(end, "choices"), (96, 191))
        self.assertIs(keyword_value(arm, "required"), True)
        self.assertIs(keyword_value(end, "required"), True)

        run_train_functions = [
            node.name for node in DRIVER_TREE.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("run_train")
        ]
        self.assertEqual(run_train_functions, ["run_train"])
        train_source = source_of(DRIVER_TEXT, function_node(DRIVER_TREE, "run_train"))
        self.assertIn('if args.arm != "C":', train_source)
        self.assertIn("if end not in (96, 191):", train_source)
        self.assertIn("if (start, end) != (96, 191):", train_source)
        self.assertIn("if end != 96:", train_source)
        self.assertNotIn("(0, 191)", train_source)
        self.assertNotIn("(191, 250)", train_source)

        # No condition A/B or 250M command can be selected through argparse.
        lowered = {command.lower() for command in parser_commands(DRIVER_TREE)}
        self.assertFalse(lowered.intersection({"a", "b", "train-a", "train-b", "250m", "continue-250m"}))

    def test_update_96_is_a_mandatory_fresh_process_boundary(self):
        train_source = source_of(DRIVER_TEXT, function_node(DRIVER_TREE, "run_train"))
        main_source = source_of(DRIVER_TEXT, function_node(DRIVER_TREE, "main"))
        restart_source = source_of(DRIVER_TEXT, function_node(DRIVER_TREE, "midpoint_restart_audit"))
        preexit_source = source_of(DRIVER_TEXT, function_node(DRIVER_TREE, "preexit_restart_record"))

        for required in (
            'durable_json(output / "MIDPOINT_RESTART_PREEXIT.json", before)',
            'heartbeat(output, end, "fresh_process_restart_required")',
            "EXPERIMENT_2D5C_SEGMENT_COMPLETE 0->96 FRESH_PROCESS_REQUIRED",
            "return",
            "midpoint_restart_audit(",
            'durable_json(output / "MIDPOINT_RESTART_AUDIT.json", restart)',
        ):
            self.assertIn(required, train_source)
        self.assertIn('before["saved_process_id"] != os.getpid()', restart_source)
        self.assertIn('"saved_process_id": os.getpid()', preexit_source)
        self.assertIn("segment 0->96 cannot accept resume artifacts", main_source)
        self.assertIn("segment 96->191 requires both sealed midpoint artifacts", main_source)
        self.assertIn("args.resume_checkpoint and args.midpoint_preexit", main_source)

    def test_fourteen_condition_evaluation_and_frozen_statistics_are_locked(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        evaluation = config["evaluation"]
        self.assertEqual(evaluation["large_sequences"], 2_048)
        self.assertEqual(evaluation["large_targets_per_condition"], 2_097_152)
        self.assertEqual(evaluation["large_selection_seed"], 2_026_083_001)
        self.assertEqual(evaluation["shuffle_seed"], 2_026_083_002)
        self.assertEqual(evaluation["bootstrap_seed"], 2_026_083_003)
        self.assertEqual(evaluation["bootstrap_resamples"], 50_000)
        self.assertEqual(evaluation["noninferiority_margin_ce"], 0.001)

        self.assertEqual(DRIVER_CONSTANTS["CONTROLS"], CONTROLS)
        self.assertEqual(CORE_CONSTANTS["INCREMENTAL_CONTROLS"], CONTROLS)
        self.assertEqual(DRIVER_CONSTANTS["LARGE_SELECTION_SEED"], 2_026_083_001)
        self.assertEqual(DRIVER_CONSTANTS["SHUFFLE_SEED"], 2_026_083_002)
        self.assertEqual(DRIVER_CONSTANTS["BOOTSTRAP_SEED"], 2_026_083_003)
        self.assertEqual(DRIVER_CONSTANTS["BOOTSTRAP_RESAMPLES"], 50_000)
        self.assertEqual(DRIVER_CONSTANTS["NONINFERIORITY_MARGIN"], 0.001)
        self.assertEqual(ANALYSIS_CONSTANTS["BOOTSTRAP_SEED"], 2_026_083_003)
        self.assertEqual(ANALYSIS_CONSTANTS["BOOTSTRAP_RESAMPLES"], 50_000)
        self.assertEqual(ANALYSIS_CONSTANTS["NONINFERIORITY_MARGIN_CE"], 0.001)

        suffixes = DRIVER_CONSTANTS["CONDITION_CANONICAL_SUFFIX"]
        self.assertEqual(tuple(suffixes), CONTROLS)
        self.assertEqual(len(suffixes) * 2, 14)
        large_source = source_of(DRIVER_TEXT, function_node(DRIVER_TREE, "large_loss_artifact"))
        analyze_source = source_of(DRIVER_TEXT, function_node(DRIVER_TREE, "run_analyze"))
        audit_source = source_of(DRIVER_TEXT, function_node(DRIVER_TREE, "scientific_audit_checks"))
        self.assertIn('for prefix, evaluation in (("C", c_evaluation), ("F", fixed_evaluation))', large_source)
        self.assertIn("analysis.analyze_final_large_panel(", analyze_source)
        self.assertIn("seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES", analyze_source)
        self.assertIn('"all_14_large_conditions_completed"', audit_source)
        self.assertIn('"fourteen_condition_evaluation_not_reduced"', audit_source)
        self.assertIn("len(large_losses[\"conditions\"]) == 14", audit_source)

        main_source = source_of(DRIVER_TEXT, function_node(DRIVER_TREE, "main"))
        self.assertIn("C large evaluation requires --final-checkpoint-seal", main_source)

    def test_analysis_raw_artifact_identity_helpers_fail_closed(self):
        names = (
            "implementation_file_sha256", "nested_value", "valid_sha256",
            "canonical_identity_matches",
            "evaluation_artifact_identity_checks",
            "representation_artifact_identity_checks",
        )
        module = ast.fix_missing_locations(ast.Module(
            body=[copy.deepcopy(function_node(DRIVER_TREE, name)) for name in names],
            type_ignores=[],
        ))
        canonical_sha = lambda value: hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        environment = {
            "hashlib": hashlib,
            "math": math,
            "REPO_ROOT": ROOT,
            "sha256": lambda path: hashlib.sha256(
                Path(path).read_bytes()
            ).hexdigest(),
            "base": types.SimpleNamespace(T=1024),
            "canonical_sha": canonical_sha,
            "EXPERIMENT": "2D5C",
            "CORE_SHA256": DRIVER_CONSTANTS["CORE_SHA256"],
            "EVALUATION_IDENTITY_KEYS": DRIVER_CONSTANTS["EVALUATION_IDENTITY_KEYS"],
            "REPRESENTATION_IDENTITY_KEYS": DRIVER_CONSTANTS["REPRESENTATION_IDENTITY_KEYS"],
            "REPRESENTATION_DIAGNOSTIC_SCHEMA": DRIVER_CONSTANTS[
                "REPRESENTATION_DIAGNOSTIC_SCHEMA"
            ],
        }
        exec(compile(module, str(DRIVER_PATH), "exec"), environment)

        checkpoint_sha = "1" * 64
        model_sha = "2" * 64
        panel_file_sha = "3" * 64
        shuffle_file_sha = "4" * 64
        fingerprint = "5" * 64
        identity = {
            "family": "C",
            "checkpoint_sha256": checkpoint_sha,
            "local_update": 96,
            "model_state_sha256": model_sha,
            "architecture_fingerprint": fingerprint,
            "panel_manifest_path": "/remote/PANEL_MANIFEST_CORE.json",
            "panel_manifest_sha256": panel_file_sha,
            "shuffle_manifest_path": "/remote/SHUFFLE_MANIFEST.json",
            "shuffle_manifest_sha256": shuffle_file_sha,
        }
        identity["identity_sha256"] = canonical_sha(identity)
        batch_identity = {"combined_sha256": "6" * 64}
        panel = {
            "batch_indices_in_evaluation_order": [0],
            "batch_identities": [batch_identity],
            "subset_sha256": DRIVER_CONSTANTS["CORE_SHA256"],
            "sequences": 2,
            "targets_per_condition": 2048,
        }
        spec = {
            "family": "C", "local_update": 96,
            "checkpoint_sha256": checkpoint_sha,
            "architecture_fingerprint": fingerprint,
            "controls": ("all_real",),
        }
        condition = {
            "validation_loss": 3.0,
            "validation_targets": 2048,
            "paired_sequences": 2,
            "per_sequence_nll": [3072.0, 3072.0],
            "per_sequence_ce": [3.0, 3.0],
            "final_cache_audit": {"passed": True},
        }
        evaluation = {
            "experiment": "2D5C", "family": "C",
            "architecture_fingerprint": fingerprint,
            "evaluation_identity": identity,
            "panel_sha256": DRIVER_CONSTANTS["CORE_SHA256"],
            "controls_requested": ["all_real"],
            "conditions": {"all_real": condition},
            "batch_indices_in_evaluation_order": [0],
            "completed_batch_indices": [0],
            "batch_identities": [batch_identity],
            "same_sequence_order_all_conditions": True,
            "cache_reset_between_conditions": True,
            "all_real_terminal_sentinel": {"passed": True},
            "status": "complete", "passed": True,
        }
        check_evaluation = environment["evaluation_artifact_identity_checks"]
        self.assertTrue(check_evaluation(
            evaluation, spec, panel, panel_file_sha, shuffle_file_sha
        )["passed"])
        wrong = copy.deepcopy(evaluation)
        wrong["evaluation_identity"]["checkpoint_sha256"] = "7" * 64
        self.assertFalse(check_evaluation(
            wrong, spec, panel, panel_file_sha, shuffle_file_sha
        )["passed"])
        wrong = copy.deepcopy(evaluation)
        wrong["batch_identities"] = []
        self.assertFalse(check_evaluation(
            wrong, spec, panel, panel_file_sha, shuffle_file_sha
        )["passed"])

        selected = [
            {"combined_sha256": f"{index:064x}"}
            for index in range(1, 33)
        ]
        diagnostic_sha = canonical_sha(selected)
        optimizer_binding = {"method": "test", "checks": {"exact": True}, "passed": True}
        representation_identity = {
            "diagnostic_schema": DRIVER_CONSTANTS[
                "REPRESENTATION_DIAGNOSTIC_SCHEMA"
            ],
            "diagnostic_implementation_sha256": {
                name: environment["implementation_file_sha256"]()[name]
                for name in (
                    "scripts/experiment_2d5c.py",
                    "scripts/experiment_2d5c_core.py",
                )
            },
            "experiment": "2D5C", "label": "c96", "family": "C",
            "local_update": 96, "checkpoint_sha256": checkpoint_sha,
            "architecture_fingerprint": fingerprint,
            "model_state_sha256": model_sha,
            "optimizer_state_sha256": "8" * 64,
            "optimizer_model_binding_sha256": canonical_sha(optimizer_binding),
            "core_manifest_sha256": panel_file_sha,
            "core_subset_sha256": DRIVER_CONSTANTS["CORE_SHA256"],
            "diagnostic_subset_sha256": diagnostic_sha,
        }
        representation_identity["identity_sha256"] = canonical_sha(
            representation_identity
        )
        representation = {
            "experiment": "2D5C", "label": "c96", "family": "C",
            "local_update": 96, "checkpoint_sha256": checkpoint_sha,
            "architecture_fingerprint": fingerprint,
            "core_sha256": DRIVER_CONSTANTS["CORE_SHA256"],
            "diagnostic_subset_sha256": diagnostic_sha,
            "diagnostic_sequences": 32, "targets": 32 * 1024,
            "run_identity": representation_identity,
            "optimizer_model_binding": optimizer_binding,
            "per_sequence": [
                {"combined_sha256": item["combined_sha256"],
                 "run_identity_sha256": representation_identity["identity_sha256"]}
                for item in selected
            ],
            "frozen_artifact_checks": {"core_manifest_sha256": True},
            "state_invariance": {
                "model_unchanged": True, "optimizer_unchanged": True,
                "optimizer_steps_executed": 0,
                "no_parameter_gradients_retained": True,
            },
            "passed": True,
        }
        core_manifest = {
            "subset_sha256": DRIVER_CONSTANTS["CORE_SHA256"],
            "diagnostic_subset_sha256": diagnostic_sha,
            "diagnostic_sequence_identities": selected,
        }
        check_representation = environment["representation_artifact_identity_checks"]
        self.assertTrue(check_representation(
            representation, spec, core_manifest, panel_file_sha
        )["passed"])
        wrong = copy.deepcopy(representation)
        wrong["per_sequence"][0]["run_identity_sha256"] = "9" * 64
        self.assertFalse(check_representation(
            wrong, spec, core_manifest, panel_file_sha
        )["passed"])

    def test_analysis_requires_frozen_manifests_parallel_evidence_and_all_substantive_checks(self):
        parser_source = source_of(DRIVER_TEXT, function_node(DRIVER_TREE, "build_parser"))
        analyze_source = source_of(DRIVER_TEXT, function_node(DRIVER_TREE, "run_analyze"))
        identity_source = source_of(
            DRIVER_TEXT, function_node(DRIVER_TREE, "analysis_input_identity_audit")
        )
        scientific_source = source_of(
            DRIVER_TEXT, function_node(DRIVER_TREE, "scientific_audit_checks")
        )
        for argument in (
            "pretrain_freeze_audit", "core_panel_manifest", "large_panel_manifest",
            "shuffle_manifest", "milestone_manifest", "c0_parallel",
            "c96_parallel", "c191_parallel",
        ):
            self.assertIn(f'"{argument}"', parser_source)
        for required in (
            'set(parallel_evaluations) == {"c0", "c96", "c191"}',
            'for name in ("c0", "c96", "c191")',
            'f"parallel_{name}_same_identity"',
            '"preflight_freeze_manifest_exact"',
            '"file_sha256_matches_training_complete"',
            '"final_seal_milestone_sha"',
            '"core_{name}_model_matches_milestone"',
            '"representation_c191_optimizer_matches_seal"',
        ):
            self.assertIn(required, identity_source)
        self.assertIn('"analysis_input_identities_exact"', scientific_source)
        self.assertIn('"secondary_parallel_c0_c96_c191_completed"', scientific_source)
        self.assertIn(
            'post_analysis_operational = {"git_branch_commit_tag_pushed_verified"}',
            analyze_source,
        )
        self.assertIn("substantive_failures = [", analyze_source)
        self.assertIn("scientific_integrity_passed = not substantive_failures", analyze_source)
        self.assertIn("classification withheld", analyze_source)

    def test_mandatory_preflight_exercises_runtime_evidence_fail_closed(self):
        runtime_class = class_node(CORE_TREE, "_B3B5ControlledFixedWriterGPT")
        selector_source = source_of(
            CORE_TEXT,
            function_node_in_class(runtime_class, "_incremental_bank_from_ring"),
        )
        for required in (
            'ring.size(1) != len(logical_positions)',
            'logical_positions != expected',
            'logical_positions[-1] >= int(state_position)',
            'lags <= 0',
        ):
            self.assertIn(required, selector_source)

        ring_source = source_of(
            DRIVER_TEXT, function_node(DRIVER_TREE, "ring_index_mapping_test")
        )
        for required in (
            '"lag_1023"',
            '"lag_2"',
            '"physical_ring_index": capacity - 2',
            '"current_position_metadata_rejected"',
            '"future_position_metadata_rejected"',
            '"excluded_newest_cannot_change_selected_bank"',
        ):
            self.assertIn(required, ring_source)

        causality_source = source_of(
            DRIVER_TEXT, function_node(DRIVER_TREE, "causality_test")
        )
        self.assertIn('"post_rollover_ring_causality"', causality_source)
        self.assertIn('post_rollover_probe["passed"]', causality_source)

        state_source = source_of(
            DRIVER_TEXT, function_node(DRIVER_TREE, "incremental_state_comparison")
        )
        self.assertIn("zip(left.caches, right.caches)", state_source)
        for ring in ('"h7"', '"h8"', '"h10"', '"h12"'):
            self.assertIn(ring, state_source)
        reload_source = source_of(
            DRIVER_TEXT, function_node(DRIVER_TREE, "incremental_reload_equivalence")
        )
        self.assertIn('"all_kv_caches_exact"', reload_source)
        self.assertIn('"all_h7_h8_h10_h12_rings_exact"', reload_source)

        controls_source = source_of(
            DRIVER_TEXT, function_node(DRIVER_TREE, "control_specificity_test")
        )
        for required in (
            'shuffle_manifest["permutations"][str(batch_size)]',
            '"logits_changed_vs_all_real"',
            '"frozen_donor_permutation_applied_to_actual_source_reads"',
            '"intervention_outputs"',
        ):
            self.assertIn(required, controls_source)

        forbidden_source = source_of(
            DRIVER_TEXT, function_node(DRIVER_TREE, "forbidden_component_audit")
        )
        for required in (
            'read_json(config_path)["architecture_c"]',
            "model.named_modules()",
            "unexpected_projection_modules",
            '"runtime_manifest_auxiliary_false"',
            '"auxiliary_objective"',
        ):
            self.assertIn(required, forbidden_source)

        feasibility_source = source_of(
            DRIVER_TEXT,
            function_node(DRIVER_TREE, "representation_diagnostic_feasibility_smoke"),
        )
        for required in (
            "ExplicitShardLoader([val_path], 1, base.T)",
            "one_sequence_representation_diagnostic(",
            "torch.cuda.reset_peak_memory_stats(device)",
            "time.perf_counter()",
            '"peak_allocated_bytes"',
            '"source_gradient_nonzero_pairs"',
        ):
            self.assertIn(required, feasibility_source)
        preflight_source = source_of(
            DRIVER_TEXT, function_node(DRIVER_TREE, "run_preflight")
        )
        for key in (
            '"ring_index_mapping"',
            '"optimizer_name_rebinding"',
            '"representation_diagnostic_production_shape_feasibility"',
        ):
            self.assertIn(key, preflight_source)

        rebind_source = source_of(
            DRIVER_TEXT,
            function_node(DRIVER_TREE, "rebind_optimizer_by_parameter_name"),
        )
        for required in (
            "source_model.named_parameters()",
            "evaluator_model.named_parameters()",
            "source_optimizer.param_groups",
            "copy.deepcopy(state)",
            '"state_values_exact"',
            '"state_tensor_storage_independent"',
            '"group_parameter_names_and_metadata_exact"',
        ):
            self.assertIn(required, rebind_source)
        diagnostic_source = source_of(
            DRIVER_TEXT, function_node(DRIVER_TREE, "run_representation_diagnostics")
        )
        self.assertNotIn("optimizer.zero_grad(set_to_none=True)", diagnostic_source)
        self.assertGreaterEqual(
            diagnostic_source.count("model.zero_grad(set_to_none=True)"), 2
        )
        self.assertIn('"optimizer_model_binding"', diagnostic_source)
        self.assertIn('"diagnostic_schema"', diagnostic_source)
        self.assertIn('"diagnostic_implementation_sha256"', diagnostic_source)
        self.assertIn("finite_numeric_tree", DRIVER_TEXT)
        self.assertIn(
            "optimizer_model_binding_sha256",
            DRIVER_CONSTANTS["REPRESENTATION_IDENTITY_KEYS"],
        )

    def test_historical_disjointness_binds_dataset_and_exact_spans_without_overclaim(self):
        history_source = source_of(
            DRIVER_TEXT, function_node(DRIVER_TREE, "historical_panel_rows")
        )
        for required in (
            "base.batch_identity(*batch_at_index(val_path, index))",
            '"same_index_batch_identity_replay_exact"',
            '"dataset_identity"',
            '"available_in_historical_manifest"',
            '"unavailable; no historical per-sequence disjointness claim is made"',
        ):
            self.assertIn(required, history_source)
        prepare_source = source_of(
            DRIVER_TEXT, function_node(DRIVER_TREE, "prepare_panels")
        )
        for required in (
            "half_open_span_intersections(",
            '"canonical_target_span_intersections"',
            '"canonical_span_nonoverlap_verified"',
            '"sequence_hash_intersection": sequence_intersection',
            'panel["dataset_identity"]["verified"]',
        ):
            self.assertIn(required, prepare_source)
        scientific_source = source_of(
            DRIVER_TEXT, function_node(DRIVER_TREE, "scientific_audit_checks")
        )
        self.assertIn("historical_dataset_and_span_evidence_exact", scientific_source)
        self.assertIn('tests["ring_index_mapping"]["passed"]', scientific_source)
        self.assertIn(
            'tests["representation_diagnostic_production_shape_feasibility"]["passed"]',
            scientific_source,
        )

    def test_required_driver_and_guard_commands_and_artifacts_are_complete(self):
        self.assertEqual(parser_commands(DRIVER_TREE), DRIVER_COMMANDS)
        self.assertEqual(parser_commands(GUARD_TREE), GUARD_COMMANDS)
        self.assertEqual(ANALYSIS_CONSTANTS["REQUIRED_ARTIFACTS"], REQUIRED_ARTIFACTS)

        # Artifacts with fixed production names must be emitted/referenced by
        # the driver.  The final provenance/audit/report filenames are dynamic
        # output paths, so the analysis validator binds their exact names while
        # the corresponding three required commands bind their producers.
        dynamic = {
            "FINAL_CHECKPOINT_PROVENANCE.json": "seal-final",
            "FINAL_AUDIT.json": "postflight-audit",
            "EXPERIMENT_2D5C_FINAL_REPORT.md": "render-report",
        }
        for artifact in REQUIRED_ARTIFACTS:
            if artifact not in dynamic:
                self.assertIn(artifact, DRIVER_TEXT, artifact)
        self.assertTrue(set(dynamic.values()).issubset(parser_commands(DRIVER_TREE)))
        for command in dynamic.values():
            self.assertIn(command, DRIVER_TEXT)

        self.assertEqual(DRIVER_TEXT.count(FINAL_PHRASE), 1)
        report_source = source_of(DRIVER_TEXT, function_node(DRIVER_TREE, "run_render_report"))
        self.assertIn(FINAL_PHRASE, report_source)

    def test_core_geometry_writer_identity_and_fingerprints_match_driver_manifest(self):
        expected_fixed_geometry = (
            (1, 0, 2, 12, 2, 1023),
            (2, 1, 1024, None, None, None),
            (3, 2, 32, 10, 32, 1023),
            (4, 3, 1024, None, None, None),
            (5, 4, 64, 8, 64, 1023),
            (6, 5, 512, 7, 512, 1023),
            *tuple((block, block - 1, 1024, None, None, None) for block in range(7, 13)),
        )
        expected_c_geometry = (
            (1, 0, 2, 12, 2, 1023),
            (2, 1, 1024, None, None, None),
            (3, 2, 2, 10, 2, 1023),
            (4, 3, 1024, None, None, None),
            (5, 4, 2, 8, 2, 1023),
            (6, 5, 512, 7, 512, 1023),
            *tuple((block, block - 1, 1024, None, None, None) for block in range(7, 13)),
        )
        self.assertEqual(CORE_CONSTANTS["FIXED_CONTROL_BLOCK_GEOMETRY"], expected_fixed_geometry)
        self.assertEqual(CORE_CONSTANTS["C_BLOCK_GEOMETRY"], expected_c_geometry)
        self.assertEqual(CORE_CONSTANTS["FIXED_WRITER_SOURCES"], {0: 11, 2: 9, 4: 7, 5: 6})
        self.assertEqual(
            CORE_CONSTANTS["FIXED_WRITERS"],
            {"B1": "B12", "B3": "B10", "B5": "B8", "B6": "B7"},
        )
        self.assertEqual(CORE_CONSTANTS["EXPECTED_PARAMETER_COUNT"], 124_475_908)
        self.assertEqual(CORE_CONSTANTS["C_SOURCE_CHECKPOINT_SHA256"], SOURCE_SHA256)
        self.assertEqual(CORE_CONSTANTS["FIXED_CONTROL_CHECKPOINT_SHA256"], CONTROL_SHA256)
        self.assertEqual(CORE_CONSTANTS["ARCHITECTURE_FINGERPRINT_C"], FINGERPRINT_C)
        self.assertEqual(CORE_CONSTANTS["ARCHITECTURE_FINGERPRINT_FIXED"], FINGERPRINT_FIXED)

        # Execute only the isolated pure manifest builder AST, never the driver
        # module or its Torch imports, and require its canonical fingerprints.
        manifest_function = copy.deepcopy(function_node(DRIVER_TREE, "architecture_manifest"))
        module = ast.fix_missing_locations(ast.Module(body=[manifest_function], type_ignores=[]))
        environment = {
            "EXPERIMENT": "2D5C",
            "SOURCE_SHA256": SOURCE_SHA256,
            "CONTROL_SHA256": CONTROL_SHA256,
            "PARAMETERS": 124_475_908,
            "canonical_sha": lambda value: hashlib.sha256(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        exec(compile(module, str(DRIVER_PATH), "exec"), environment)
        c_manifest = environment["architecture_manifest"]("C")
        fixed_manifest = environment["architecture_manifest"]("Fixed")
        self.assertEqual(c_manifest["fingerprint_sha256"], FINGERPRINT_C)
        self.assertEqual(fixed_manifest["fingerprint_sha256"], FINGERPRINT_FIXED)
        self.assertEqual(c_manifest["new_parameters"], 0)
        self.assertEqual(c_manifest["state_dict_key_changes"], 0)
        self.assertEqual(c_manifest["fixed_writer_identity"], CORE_CONSTANTS["FIXED_WRITERS"])

        fixed_class = source_of(CORE_TEXT, class_node(CORE_TREE, "FixedControlEvaluationGPT"))
        c_class = source_of(
            CORE_TEXT, class_node(CORE_TREE, "FixedWriterB3B5W2RepresentationPressureGPT")
        )
        self.assertIn("architecture_fingerprint_sha256 = ARCHITECTURE_FINGERPRINT_FIXED", fixed_class)
        self.assertIn("architecture_fingerprint_sha256 = ARCHITECTURE_FINGERPRINT_C", c_class)

    def test_runpod_guard_targets_one_exact_pod_and_exposes_no_destructive_command(self):
        expected_constants = {
            "POD_ID": POD_ID,
            "POD_NAME": POD_NAME,
            "GPU_COUNT": 1,
            "NETWORK_VOLUME_ID": VOLUME_ID,
            "NETWORK_VOLUME_NAME": "unlikely_lime_flamingo",
            "NETWORK_VOLUME_SIZE_GB": 150,
            "VOLUME_MOUNT_PATH": "/workspace",
            "EXACT_STOP_ARGV": ("runpodctl", "pod", "stop", POD_ID, "-o", "json"),
            "EXACT_STOP_COMMAND": f"runpodctl pod stop {POD_ID} -o json",
        }
        for name, value in expected_constants.items():
            self.assertEqual(GUARD_CONSTANTS[name], value, name)

        client = class_node(GUARD_TREE, "RunPodClient")
        calls = []
        for node in ast.walk(client):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_call"
                and node.args
            ):
                calls.append(tuple(safe_eval(node.args[0], GUARD_CONSTANTS)))
        self.assertCountEqual(
            calls,
            [
                ("pod", "get", POD_ID, "--include-network-volume", "-o", "json"),
                ("pod", "list", "--all", "-o", "json"),
                ("network-volume", "get", VOLUME_ID, "-o", "json"),
                ("pod", "stop", POD_ID, "-o", "json"),
            ],
        )
        mutating = [argv for argv in calls if argv[:2] not in {("pod", "get"), ("pod", "list"), ("network-volume", "get")}]
        self.assertEqual(mutating, [("pod", "stop", POD_ID, "-o", "json")])
        destructive = {"delete", "remove", "terminate", "destroy", "purge", "reset"}
        self.assertFalse(
            destructive.intersection(
                token.lower() for argv in calls for token in argv if isinstance(token, str)
            )
        )

        guard_class = class_node(GUARD_TREE, "Experiment2D5CRunPodGuard")
        stop_source = source_of(
            GUARD_TEXT, function_node_in_class(guard_class, "stop")
        )
        self.assertIn("stop_response = self.client.stop_exact_pod()", stop_source)
        self.assertIn("stop_response.get(\"id\") not in (None, POD_ID)", stop_source)
        self.assertIn("validate_exact_volume(final_volume)", stop_source)
        self.assertIn('final_pod.get("networkVolumeId") != NETWORK_VOLUME_ID', stop_source)
        self.assertIn('"stopped_and_volume_retained_verified"', stop_source)
        self.assertIn('candidate.get("runtimeStatus") == "stopped"', stop_source)

        supervise_source = source_of(
            GUARD_TEXT,
            function_node_in_class(guard_class, "supervise_and_stop"),
        )
        self.assertIn('if outcome == "failure":', supervise_source)
        self.assertIn('"pod_stop_attempted": False', supervise_source)
        self.assertIn('"trigger_artifact_created": False', supervise_source)
        self.assertIn('"retained_for_recoverable_diagnosis": True', supervise_source)
        self.assertLess(
            supervise_source.index('if outcome == "failure":'),
            supervise_source.index("self.create_trigger("),
        )

        for node in ast.walk(GUARD_TREE):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"run", "Popen", "call", "check_call", "check_output"}:
                    shell_keywords = [keyword for keyword in node.keywords if keyword.arg == "shell"]
                    self.assertTrue(
                        not shell_keywords
                        or all(safe_eval(keyword.value, GUARD_CONSTANTS) is False for keyword in shell_keywords)
                    )

    def test_disposable_continuation_requires_exact_boundary_not_bitwise_cuda_backward(self):
        smoke_source = source_of(
            DRIVER_TEXT, function_node(DRIVER_TREE, "disposable_smoke")
        )
        for required in (
            '"saved_boundary_exact": all(boundary_checks.values())',
            '"forward_losses_exact"',
            '"post_step_bitwise"',
            '"informational_only": True',
            'numerical_comparison["passed"]',
            'all(boundary_checks.values())',
            'all(exact_continuation_checks.values())',
        ):
            self.assertIn(required, smoke_source)
        self.assertNotIn("all(continuation_checks.values())", smoke_source)
        self.assertIn(
            "inherited CUDA backward is demonstrably non-bitwise",
            smoke_source,
        )


def function_node_in_class(parent: ast.ClassDef, name: str) -> ast.FunctionDef:
    rows = [
        node for node in parent.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(rows) != 1:
        raise AssertionError(f"expected one {parent.name}.{name}, found {len(rows)}")
    return rows[0]


if __name__ == "__main__":
    unittest.main()
