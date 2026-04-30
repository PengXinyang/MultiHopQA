import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "multi_hop_qa"))

controller_stub = types.ModuleType("graph_of_thoughts.controller")
auto_goo_stub = types.ModuleType("graph_of_thoughts.auto_goo_designer")
goo_builder_stub = types.ModuleType("graph_of_thoughts.goo_builder")


def _unused(*args, **kwargs):
    raise AssertionError("unexpected call")


auto_goo_stub.requestGotGooDesign = _unused
auto_goo_stub.saveDesignResult = _unused
goo_builder_stub.loadGooDesignFromFile = _unused
goo_builder_stub.buildGraphFromGooDesign = _unused
sys.modules["graph_of_thoughts.controller"] = controller_stub
sys.modules["graph_of_thoughts.auto_goo_designer"] = auto_goo_stub
sys.modules["graph_of_thoughts.goo_builder"] = goo_builder_stub

import utils


class FailingController:
    def __init__(self, lm, graph, prompter, parser, problem_parameters, event_sink=None):
        self.lm = lm
        self.graph = types.SimpleNamespace(leaves=[])

    def run(self):
        raise RuntimeError("controller failed")

    def output_graph(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("[]")


def failing_method():
    return object()


controller_stub.Controller = FailingController


class DummyLM:
    prompt_tokens = 0
    completion_tokens = 0
    cost = 0.0


class RunSingleMethodTests(unittest.TestCase):
    def test_controller_failure_is_propagated_after_outputs_are_written(self):
        item = {
            "_id": "sample-1",
            "question": "What failed?",
            "context": [],
            "answer": "failure",
        }

        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, failing_method.__name__))

            with self.assertRaisesRegex(RuntimeError, "controller failed"):
                utils.runSingleMethod(
                    item=item,
                    method=failing_method,
                    lm=DummyLM(),
                    prompter=object(),
                    parser=object(),
                    run_dir=tmp,
                )

            self.assertTrue(
                os.path.exists(
                    os.path.join(tmp, failing_method.__name__, "sample-1.json")
                )
            )
            self.assertTrue(
                os.path.exists(
                    os.path.join(
                        tmp, failing_method.__name__, "sample-1.summary.json"
                    )
                )
            )


if __name__ == "__main__":
    unittest.main()
