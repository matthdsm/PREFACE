import unittest
import onnx
from preface.lib.neural import create_model, neural_export


class TestNeuralExport(unittest.TestCase):
    def test_neural_export(self):
        # 1. Create a dummy model
        input_dim = 10
        n_layers = 2
        hidden_size = 32
        learning_rate = 0.001
        dropout_rate = 0.2

        model = create_model(
            input_dim=input_dim,
            n_layers=n_layers,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
        )

        # 2. Export to ONNX
        onnx_model = neural_export(model)

        # 3. Verify
        self.assertIsInstance(onnx_model, onnx.ModelProto)

        # Check if the graph has nodes
        self.assertGreater(len(onnx_model.graph.node), 0)

        # Check inputs
        self.assertEqual(len(onnx_model.graph.input), 1)
        self.assertEqual(onnx_model.graph.input[0].name, "neural_input")

        # Validate the model using onnx.checker
        onnx.checker.check_model(onnx_model)


if __name__ == "__main__":
    unittest.main()
