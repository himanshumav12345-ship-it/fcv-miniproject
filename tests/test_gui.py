"""
Tests for the GUI server (src/gui.py), using Flask's built-in test client - no browser needed.

Run:  python -m unittest discover -s tests -t . -v
"""
import io
import unittest

import cv2
import numpy as np

from src import config, gui
from tests.test_pipeline import synthetic_leaf

MODEL_EXISTS = (config.MODELS_DIR / "classifier.joblib").exists()


def png_bytes(image):
    ok, buffer = cv2.imencode(".png", image)
    return buffer.tobytes()


class GuiServerTests(unittest.TestCase):
    def setUp(self):
        self.client = gui.create_app().test_client()

    def post_photo(self, raw_bytes, name="photo.png"):
        return self.client.post("/analyse", data={"photo": (io.BytesIO(raw_bytes), name)},
                                content_type="multipart/form-data")

    def test_page_is_served(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Tomato leaf check", response.get_data(as_text=True))

    def test_request_without_a_file_is_rejected_politely(self):
        response = self.client.post("/analyse")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["kind"], "no_file")

    def test_text_file_is_reported_as_invalid_image(self):
        response = self.post_photo(b"this is not an image", "notes.txt")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()["kind"], "invalid_image")

    def test_empty_file_is_reported_as_invalid_image(self):
        response = self.post_photo(b"")
        self.assertEqual(response.get_json()["kind"], "invalid_image")

    def test_tiny_image_is_reported_as_invalid_image(self):
        response = self.post_photo(png_bytes(np.zeros((20, 20, 3), np.uint8)))
        self.assertEqual(response.get_json()["kind"], "invalid_image")

    def test_oversized_upload_is_refused(self):
        response = self.post_photo(b"x" * (gui.MAX_UPLOAD_BYTES + 1024))
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.get_json()["kind"], "too_large")

    def test_severity_bands_match_the_config(self):
        bands = gui.severity_bands()
        self.assertEqual([b["to"] for b in bands], [limit for limit, _ in config.SEVERITY_LEVELS])
        self.assertEqual(bands[0]["from"], 0.0)
        self.assertEqual(bands[1]["from"], bands[0]["to"])

    @unittest.skipUnless(MODEL_EXISTS, "train the model first: python -m src.train")
    def test_blank_picture_reports_no_leaf(self):
        blank = np.full((256, 256, 3), 128, np.uint8)
        response = self.post_photo(png_bytes(blank))
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()["kind"], "no_leaf")

    @unittest.skipUnless(MODEL_EXISTS, "train the model first: python -m src.train")
    def test_leaf_photo_returns_label_severity_and_six_pictures(self):
        response = self.post_photo(png_bytes(synthetic_leaf(lesion=True)))
        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data["ok"])
        self.assertIn(data["disease"], config.CLASS_NAMES)
        self.assertAlmostEqual(sum(data["probabilities"].values()), 1.0, places=5)
        self.assertGreater(data["severity_percent"], 5)
        self.assertEqual(len(data["panels"]), 6)
        self.assertTrue(all(p["image"].startswith("data:image/png;base64,") for p in data["panels"]))


if __name__ == "__main__":
    unittest.main()
