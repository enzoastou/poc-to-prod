import unittest
from predict.predict.run import TextPredictionModel


class TestPredict(unittest.TestCase):
    def test_predict(self):
        model = TextPredictionModel.from_artefacts(
            "D:\\EPF\\5A\\POCtoPROD\\poc-to-prod-capstone\\train\\data\\artefacts\\2023-01-04-22-31-58")
        predictions = model.predict("Is it possible to execute the procedure of a function in the scope of the caller?",
                                    top_k=1)
        # Assert that the test sentence is correctly predicted
        self.assertEqual(predictions, ["php"])
