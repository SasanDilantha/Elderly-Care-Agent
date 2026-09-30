import unittest

from elderly_care_agent.config import ApplicationSettings, VisionSettings, VlmSettings


class ApplicationSettingsTests(unittest.TestCase):
    def test_required_ollama_model_is_the_default(self) -> None:
        settings = ApplicationSettings()

        self.assertEqual("qwen3-vl:4b-instruct", settings.vlm.model)
        self.assertIn(0.0, settings.vlm.context_offsets_sec)

    def test_invalid_confidence_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            VisionSettings(minimum_detection_confidence=1.1)

    def test_context_offsets_must_be_sorted(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be sorted"):
            VlmSettings(context_offsets_sec=(0.0, -2.0, 2.0))


if __name__ == "__main__":
    unittest.main()
