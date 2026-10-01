import io
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from elderly_care_agent.application.video_sampling import VideoSamplingReport
from elderly_care_agent.application.vision_features import VisionAnalysisReport
from elderly_care_agent.cli import CliApplication
from elderly_care_agent.domain.dataset import DatasetPreparationReport
from elderly_care_agent.domain.video import FrameReference
from elderly_care_agent.domain.vision import BedRegion
from elderly_care_agent.infrastructure.logging_config import LoggingConfigurator


class CliApplicationTests(unittest.TestCase):
    def test_show_config_prints_required_model(self) -> None:
        output = io.StringIO()

        exit_code = CliApplication(output=output).run(["show-config"])

        payload = json.loads(output.getvalue())
        self.assertEqual(0, exit_code)
        self.assertEqual("qwen3-vl:4b-instruct", payload["vlm"]["model"])

    def test_logging_options_are_delegated_without_polluting_json_output(self) -> None:
        output = io.StringIO()
        configurator = Mock(spec=LoggingConfigurator)

        exit_code = CliApplication(
            output=output,
            logging_configurator=configurator,
        ).run(["--log-level", "DEBUG", "--console-only", "show-config"])

        self.assertEqual(0, exit_code)
        configurator.configure.assert_called_once_with("DEBUG", None)
        self.assertEqual("qwen3-vl:4b-instruct", json.loads(output.getvalue())["vlm"]["model"])

    def test_validate_annotations_prints_summary(self) -> None:
        output = io.StringIO()

        exit_code = CliApplication(output=output).run(
            ["validate-annotations", str(Path("examples/ground_truth.example.json"))]
        )

        payload = json.loads(output.getvalue())
        self.assertEqual(0, exit_code)
        self.assertTrue(payload["valid"])
        self.assertEqual(4, payload["segment_count"])
        self.assertEqual(10.0, payload["occupancy_durations_sec"]["in_bed"])

    @patch("elderly_care_agent.cli.DatasetPreparationPipeline.from_config")
    def test_prepare_dataset_prints_pipeline_report(self, from_config: Mock) -> None:
        report = DatasetPreparationReport(
            status="ready",
            dataset_name="GMDCSA-24",
            dataset_version="v2.0",
            source_owner="Ekram Alam",
            source_doi="10.5281/zenodo.12921216",
            source_repository="https://example.test/upstream",
            mirror_repository="https://example.test/mirror",
            mirror_commit="a" * 40,
            dataset_license="CC-BY-4.0",
            source_mode="existing_source_directory",
            manifest_sha256="b" * 64,
            selected_video_count=2,
            copied_video_count=0,
            reused_video_count=2,
            total_size_bytes=10,
            split_counts={"development": 1, "evaluation": 1},
            report_path="report.json",
            inventory_path="inventory.csv",
            completed_at_utc="2026-09-30T00:00:00+00:00",
        )
        from_config.return_value.run.return_value = report
        output = io.StringIO()

        exit_code = CliApplication(output=output).run(["prepare-dataset"])

        payload = json.loads(output.getvalue())
        self.assertEqual(0, exit_code)
        self.assertEqual("ready", payload["status"])
        self.assertEqual(2, payload["selected_video_count"])

    def test_inspect_video_prints_timestamped_sampling_report(self) -> None:
        service = Mock()
        service.inspect.return_value = VideoSamplingReport(
            video_id="sample.mp4",
            source_path="sample.mp4",
            source_fps=10.0,
            frame_count=20,
            width=64,
            height=48,
            duration_sec=2.0,
            requested_sample_fps=2.0,
            effective_sample_fps=2.0,
            sampled_frame_count=4,
            samples=(
                FrameReference(0, 0.0),
                FrameReference(5, 0.5),
                FrameReference(10, 1.0),
                FrameReference(15, 1.5),
            ),
        )
        output = io.StringIO()

        exit_code = CliApplication(
            output=output,
            video_sampling_service=service,
        ).run(["inspect-video", "sample.mp4", "--sample-fps", "2"])

        payload = json.loads(output.getvalue())
        self.assertEqual(0, exit_code)
        self.assertEqual("ready", payload["status"])
        self.assertEqual(4, payload["sampled_frame_count"])
        self.assertEqual(1.5, payload["samples"][-1]["timestamp_sec"])
        service.inspect.assert_called_once_with(Path("sample.mp4"), 2.0)

    def test_analyze_video_accepts_bed_region_and_returns_features(self) -> None:
        service = Mock()
        bed = BedRegion(0.1, 0.2, 0.9, 0.8)
        service.analyze.return_value = VisionAnalysisReport(
            video_id="sample.mp4",
            source_path="sample.mp4",
            duration_sec=2.0,
            source_fps=10.0,
            sample_fps=2.0,
            model_path="model.task",
            bed_region=bed,
            sampled_frame_count=0,
            detected_frame_count=0,
            unknown_frame_count=0,
            frames=(),
        )
        output = io.StringIO()

        exit_code = CliApplication(output=output, vision_analysis_service=service).run(
            ["analyze-video", "sample.mp4", "--sample-fps", "2", "--bed-region", "0.1,0.2,0.9,0.8"]
        )

        self.assertEqual(0, exit_code)
        self.assertEqual("ready", json.loads(output.getvalue())["status"])
        service.analyze.assert_called_once_with(Path("sample.mp4"), 2.0, bed)


if __name__ == "__main__":
    unittest.main()
