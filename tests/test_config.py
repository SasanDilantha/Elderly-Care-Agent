import unittest

from elderly_care_agent.config import (
    ApplicationSettings,
    RuleSettings,
    TemporalSettings,
    VisionSettings,
    VlmSettings,
)


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

    def test_rule_thresholds_are_visible_and_ordered(self) -> None:
        settings = ApplicationSettings()

        self.assertEqual(1.5, settings.temporal.state_confirmation_sec)
        self.assertEqual(30.0, settings.to_dict()["rules"]["lying_max_torso_angle_deg"])
        with self.assertRaisesRegex(ValueError, "ordered"):
            RuleSettings(lying_max_torso_angle_deg=65.0)
        with self.assertRaisesRegex(ValueError, "finite and greater"):
            TemporalSettings(state_confirmation_sec=float("nan"))


if __name__ == "__main__":
    unittest.main()
