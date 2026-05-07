import unittest
from app import detect_deepfake
from PIL import Image
import requests
from io import BytesIO

class TestDeepfakeApp(unittest.TestCase):
    def test_detect_deepfake(self):
        # Load a sample image from a URL
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))

        # Test the detection function
        results = detect_deepfake(img)

        # Assertions
        self.assertIsNotNone(results)
        self.assertIsInstance(results, dict)
        self.assertIn("Realism", results)
        self.assertIn("Deepfake", results)

        # Check that scores are floats and sum to approximately 1.0
        scores = list(results.values())
        for score in scores:
            self.assertIsInstance(score, float)
            self.assertTrue(0 <= score <= 1.0)

        self.assertAlmostEqual(sum(scores), 1.0, places=4)

if __name__ == "__main__":
    unittest.main()
