import json
import tempfile
import unittest
from pathlib import Path

from graph_of_thoughts.language_models.gemini_grouped_failover import (
    gemini_parallel_groups_configured,
)


class GeminiParallelGroupsConfiguredTest(unittest.TestCase):
    def _write_config(self, payload):
        tmpdir = tempfile.TemporaryDirectory()
        path = Path(tmpdir.name) / "config.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.addCleanup(tmpdir.cleanup)
        return str(path)

    def test_empty_config_has_no_parallel_groups(self):
        path = self._write_config({})
        self.assertFalse(gemini_parallel_groups_configured(path))

    def test_gcli_keys_do_not_count_as_native_parallel_groups(self):
        path = self._write_config(
            {
                "gemini-2.5-flash-gcli": {"api_key": "ignored"},
                "gemini-2.5-flash-gcli-1": {"api_key": "ignored"},
            }
        )
        self.assertFalse(gemini_parallel_groups_configured(path))

    def test_native_suffix_keys_enable_parallel_groups(self):
        path = self._write_config(
            {
                "gemini-2.5-flash-1": {"api_key": "key-1"},
                "gemini-2.5-flash-2": {"api_key": "key-2"},
            }
        )
        self.assertTrue(gemini_parallel_groups_configured(path))


if __name__ == "__main__":
    unittest.main()
