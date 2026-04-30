import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "multi_hop_qa"))

from graph_of_thoughts import operations
import utils


class FailingOperation(operations.Operation):
    operation_type = operations.OperationType.generate

    def __init__(self):
        super().__init__()
        self.thoughts = []

    def _execute(self, lm, prompter, parser, **kwargs):
        raise RuntimeError("controller failed")

    def get_thoughts(self):
        return self.thoughts


def failing_method():
    graph = operations.GraphOfOperations()
    graph.append_operation(FailingOperation())
    return graph


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
