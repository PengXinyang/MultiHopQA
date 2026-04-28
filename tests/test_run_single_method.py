import importlib
import os
import sys
import tempfile
import types
import unittest


def _install_utils_import_stubs():
    graph_pkg = types.ModuleType("graph_of_thoughts")
    controller_mod = types.ModuleType("graph_of_thoughts.controller")
    auto_mod = types.ModuleType("graph_of_thoughts.auto_goo_designer")
    goo_mod = types.ModuleType("graph_of_thoughts.goo_builder")

    auto_mod.requestGotGooDesign = lambda *args, **kwargs: {}
    auto_mod.saveDesignResult = lambda *args, **kwargs: ""
    goo_mod.loadGooDesignFromFile = lambda *args, **kwargs: {}
    goo_mod.buildGraphFromGooDesign = lambda *args, **kwargs: None

    graph_pkg.controller = controller_mod
    sys.modules["graph_of_thoughts"] = graph_pkg
    sys.modules["graph_of_thoughts.controller"] = controller_mod
    sys.modules["graph_of_thoughts.auto_goo_designer"] = auto_mod
    sys.modules["graph_of_thoughts.goo_builder"] = goo_mod
    return controller_mod


class RunSingleMethodTests(unittest.TestCase):
    def test_run_single_method_reraises_controller_failure_after_outputs(self):
        controller_mod = _install_utils_import_stubs()
        sys.path.insert(0, os.path.join(os.getcwd(), "multi_hop_qa"))
        utils = importlib.import_module("utils")

        class FailingController:
            def __init__(self, *args, **kwargs):
                self.graph = types.SimpleNamespace(leaves=[])
                self.lm = None

            def run(self):
                raise RuntimeError("controller failed")

            def output_graph(self, path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("diagnostic graph")

        controller_mod.Controller = FailingController

        def fake_method():
            return object()

        fake_method.__name__ = "io"

        item = {"_id": "sample-1", "question": "q", "answer": "gold", "context": []}
        with tempfile.TemporaryDirectory() as run_dir:
            os.makedirs(os.path.join(run_dir, "io"))
            with self.assertRaisesRegex(RuntimeError, "controller failed"):
                utils.runSingleMethod(
                    item=item,
                    method=fake_method,
                    lm=types.SimpleNamespace(cost=0.0),
                    prompter=object(),
                    parser=object(),
                    run_dir=run_dir,
                )

            self.assertTrue(os.path.exists(os.path.join(run_dir, "io", "sample-1.json")))
            self.assertTrue(
                os.path.exists(os.path.join(run_dir, "io", "sample-1.summary.json"))
            )


if __name__ == "__main__":
    unittest.main()
