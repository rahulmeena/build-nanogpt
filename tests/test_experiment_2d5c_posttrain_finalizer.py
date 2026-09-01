import importlib
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class PostTrainingFinalizerBindingTests(unittest.TestCase):
    def test_binds_only_replacement_pod_identity(self) -> None:
        frozen = importlib.import_module("experiment_2d5c_finalizer")
        wrapper = importlib.import_module("experiment_2d5c_posttrain_finalizer")
        original_volume = (
            frozen.VOLUME_ID,
            frozen.VOLUME_NAME,
            frozen.VOLUME_SIZE_GB,
            frozen.VOLUME_DATACENTER,
            frozen.VOLUME_MOUNT_PATH,
        )

        wrapper.bind_finalizer_identity()

        self.assertEqual(frozen.POD_ID, "7kk5yyti00rnrp")
        self.assertEqual(frozen.POD_NAME, "grand_amber_catshark")
        self.assertEqual(
            frozen.EXACT_STOP_COMMAND,
            "runpodctl pod stop 7kk5yyti00rnrp -o json",
        )
        self.assertEqual(
            (
                frozen.VOLUME_ID,
                frozen.VOLUME_NAME,
                frozen.VOLUME_SIZE_GB,
                frozen.VOLUME_DATACENTER,
                frozen.VOLUME_MOUNT_PATH,
            ),
            original_volume,
        )


if __name__ == "__main__":
    unittest.main()
