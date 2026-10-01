"""Thin command-line delivery layer."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

from elderly_care_agent.application.bed_events import (
    BedEventAnalysisService,
    ConservativeTimelineFusion,
)
from elderly_care_agent.application.observation_summary import ObservationSummaryService
from elderly_care_agent.application.rules import RuleBasedStateClassifier
from elderly_care_agent.application.services import AnnotationValidationService
from elderly_care_agent.application.timeline import TimelineAnalysisService
from elderly_care_agent.application.video_sampling import VideoSamplingService
from elderly_care_agent.application.vision_features import VisionAnalysisService
from elderly_care_agent.application.vlm_review import VlmReviewService
from elderly_care_agent.config import ApplicationSettings
from elderly_care_agent.domain.exceptions import (
    AnnotationFormatError,
    DatasetPipelineError,
    DomainValidationError,
    VideoInputError,
    VisionModelError,
    VlmServiceError,
)
from elderly_care_agent.domain.vision import BedRegion
from elderly_care_agent.infrastructure.json_annotation_repository import (
    JsonAnnotationRepository,
)
from elderly_care_agent.infrastructure.logging_config import LoggingConfigurator
from elderly_care_agent.infrastructure.mediapipe_pose import MediaPipePoseEstimatorFactory
from elderly_care_agent.infrastructure.opencv_video_source import OpenCvVideoSourceFactory
from elderly_care_agent.infrastructure.pose_model_store import PoseModelStore
from elderly_care_agent.infrastructure.vlm_adapters import (
    OllamaVisionLanguageModel,
    OpenCvJpegEncoder,
)
from elderly_care_agent.pipeline.dataset_pipeline import DatasetPreparationPipeline

logger = logging.getLogger(__name__)


class CliApplication:
    """Parses commands and delegates work to application services."""

    def __init__(
        self,
        settings: ApplicationSettings | None = None,
        validation_service: AnnotationValidationService | None = None,
        output: TextIO | None = None,
        logging_configurator: LoggingConfigurator | None = None,
        video_sampling_service: VideoSamplingService | None = None,
        vision_analysis_service: VisionAnalysisService | None = None,
        timeline_analysis_service: TimelineAnalysisService | None = None,
        vlm_review_service: VlmReviewService | None = None,
        bed_event_service: BedEventAnalysisService | None = None,
        observation_summary_service: ObservationSummaryService | None = None,
        pose_model_provider: PoseModelStore | None = None,
    ) -> None:
        self._settings = settings or ApplicationSettings()
        self._validation_service = validation_service or AnnotationValidationService(
            JsonAnnotationRepository()
        )
        self._output = output or sys.stdout
        self._logging_configurator = logging_configurator
        self._video_sampling_service = video_sampling_service
        self._vision_analysis_service = vision_analysis_service
        self._timeline_analysis_service = timeline_analysis_service
        self._vlm_review_service = vlm_review_service
        self._bed_event_service = bed_event_service
        self._observation_summary_service = observation_summary_service
        self._pose_model_provider = pose_model_provider

    def run(self, arguments: Sequence[str] | None = None) -> int:
        parser = self._build_parser()
        namespace = parser.parse_args(arguments)
        if self._logging_configurator is not None:
            log_file = None if namespace.console_only else namespace.log_file
            self._logging_configurator.configure(namespace.log_level, log_file)
        logger.info("Command started | command=%s", namespace.command)

        if namespace.command == "show-config":
            self._write_json(self._settings.to_dict())
            return 0

        if namespace.command == "validate-annotations":
            try:
                report = self._validation_service.validate(namespace.path)
            except AnnotationFormatError as error:
                logger.error("Annotation validation failed | error=%s", error)
                self._write_json({"valid": False, "error": str(error)})
                return 1
            self._write_json({"valid": True, **asdict(report)})
            return 0

        if namespace.command == "prepare-dataset":
            try:
                report = DatasetPreparationPipeline.from_config(namespace.config).run(
                    force=namespace.force
                )
            except (DatasetPipelineError, OSError) as error:
                logger.error("Dataset preparation failed | error=%s", error)
                self._write_json({"status": "failed", "error": str(error)})
                return 1
            self._write_json(report.to_dict())
            return 0

        if namespace.command == "inspect-video":
            sample_fps = (
                namespace.sample_fps
                if namespace.sample_fps is not None
                else self._settings.vision.sample_fps
            )
            service = self._video_sampling_service or VideoSamplingService(
                OpenCvVideoSourceFactory()
            )
            try:
                report = service.inspect(namespace.path, sample_fps)
            except (DomainValidationError, VideoInputError) as error:
                logger.error("Video inspection failed | error=%s", error)
                self._write_json({"status": "failed", "error": str(error)})
                return 1
            self._write_json({"status": "ready", **report.to_dict()})
            return 0

        if namespace.command == "prepare-vision-model":
            provider = self._pose_model_provider or PoseModelStore()
            try:
                model_path = provider.ensure_available()
            except VisionModelError as error:
                logger.error("Pose model preparation failed | error=%s", error)
                self._write_json({"status": "failed", "error": str(error)})
                return 1
            self._write_json({"status": "ready", "model_path": str(model_path)})
            return 0

        if namespace.command == "analyze-video":
            service = self._get_vision_service()
            try:
                report = service.analyze(namespace.path, namespace.sample_fps, namespace.bed_region)
            except (DomainValidationError, VideoInputError, VisionModelError) as error:
                logger.error("Vision analysis failed | error=%s", error)
                self._write_json({"status": "failed", "error": str(error)})
                return 1
            self._write_json({"status": "ready", **report.to_dict()})
            return 0

        if namespace.command == "infer-timeline":
            service = self._get_timeline_service()
            try:
                report = service.analyze(namespace.path, namespace.sample_fps, namespace.bed_region)
            except (DomainValidationError, VideoInputError, VisionModelError) as error:
                logger.error("Timeline analysis failed | error=%s", error)
                self._write_json({"status": "failed", "error": str(error)})
                return 1
            self._write_json({"status": "ready", **report.to_dict()})
            return 0

        if namespace.command == "review-uncertain":
            service = self._get_vlm_review_service()
            try:
                report = service.review(
                    namespace.path,
                    namespace.sample_fps,
                    namespace.bed_region,
                    namespace.max_segments,
                )
            except (
                DomainValidationError,
                VideoInputError,
                VisionModelError,
                VlmServiceError,
            ) as error:
                logger.error("VLM review failed | error=%s", error)
                self._write_json({"status": "failed", "error": str(error)})
                return 1
            self._write_json({"status": "ready", **report.to_dict()})
            return 0

        if namespace.command == "analyze-bed-events":
            service = self._get_bed_event_service(namespace.with_vlm)
            try:
                report = service.analyze(
                    namespace.path,
                    namespace.sample_fps,
                    namespace.bed_region,
                    with_vlm=namespace.with_vlm,
                    max_segments=namespace.max_segments,
                )
            except (
                DomainValidationError,
                VideoInputError,
                VisionModelError,
                VlmServiceError,
            ) as error:
                logger.error("Bed-event analysis failed | error=%s", error)
                self._write_json({"status": "failed", "error": str(error)})
                return 1
            self._write_json({"status": "ready", **report.to_dict()})
            return 0

        if namespace.command == "summarize-observation":
            service = self._observation_summary_service or ObservationSummaryService(
                self._get_bed_event_service(namespace.with_vlm),
                self._settings.decision,
            )
            try:
                report = service.summarize(
                    namespace.path,
                    namespace.sample_fps,
                    namespace.bed_region,
                    with_vlm=namespace.with_vlm,
                    max_segments=namespace.max_segments,
                )
            except (
                DomainValidationError,
                VideoInputError,
                VisionModelError,
                VlmServiceError,
            ) as error:
                logger.error("Observation summary failed | error=%s", error)
                self._write_json({"status": "failed", "error": str(error)})
                return 1
            self._write_json({"status": "ready", **report.to_dict()})
            return 0

        parser.error("a command is required")
        return 2

    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="elderly-care-agent")
        parser.add_argument(
            "--log-level",
            choices=("DEBUG", "INFO", "WARNING", "ERROR"),
            default=os.environ.get("ELDERLY_CARE_LOG_LEVEL", "INFO").upper(),
            help="terminal and file logging detail",
        )
        parser.add_argument(
            "--log-file",
            type=Path,
            default=Path(
                os.environ.get(
                    "ELDERLY_CARE_LOG_FILE",
                    "logs/elderly-care-agent.log",
                )
            ),
            help="rotating application log path",
        )
        parser.add_argument(
            "--console-only",
            action="store_true",
            help="show logs in the terminal without writing a log file",
        )
        subparsers = parser.add_subparsers(dest="command")
        subparsers.add_parser("show-config", help="print the active settings")
        validate_parser = subparsers.add_parser(
            "validate-annotations",
            help="validate one ground-truth JSON file",
        )
        validate_parser.add_argument("path", type=Path)
        prepare_parser = subparsers.add_parser(
            "prepare-dataset",
            help="download, verify, select, and validate the configured dataset",
        )
        prepare_parser.add_argument(
            "--config",
            type=Path,
            default=Path("config/dataset.yml"),
            help="dataset YAML configuration path",
        )
        prepare_parser.add_argument(
            "--force",
            action="store_true",
            help="replace a generated staged file only when its checksum differs",
        )
        inspect_parser = subparsers.add_parser(
            "inspect-video",
            help="read video metadata and list timestamped sampled frames",
        )
        inspect_parser.add_argument("path", type=Path, help="input video path")
        inspect_parser.add_argument(
            "--sample-fps",
            type=float,
            default=None,
            help="sampling rate; defaults to the application vision setting",
        )
        subparsers.add_parser(
            "prepare-vision-model",
            help="download and verify the pinned MediaPipe pose model when absent",
        )
        analyze_parser = subparsers.add_parser(
            "analyze-video",
            help="extract person, pose, and optional bed-region features from sampled frames",
        )
        analyze_parser.add_argument("path", type=Path, help="input video path")
        analyze_parser.add_argument(
            "--sample-fps",
            type=float,
            default=None,
            help="sampling rate; defaults to the application vision setting",
        )
        analyze_parser.add_argument(
            "--bed-region",
            type=BedRegion.parse,
            default=None,
            metavar="LEFT,TOP,RIGHT,BOTTOM",
            help="camera-specific normalized rectangle; omit for unknown bed relation",
        )
        timeline_parser = subparsers.add_parser(
            "infer-timeline",
            help="classify stable activity and bed states and report durations",
        )
        timeline_parser.add_argument("path", type=Path, help="input video path")
        timeline_parser.add_argument(
            "--sample-fps",
            type=float,
            default=None,
            help="sampling rate; defaults to the application vision setting",
        )
        timeline_parser.add_argument(
            "--bed-region",
            type=BedRegion.parse,
            default=None,
            metavar="LEFT,TOP,RIGHT,BOTTOM",
            help="camera-specific normalized rectangle; omit for unknown bed relation",
        )
        review_parser = subparsers.add_parser(
            "review-uncertain",
            help="ask local Ollama to review uncertain timeline intervals",
        )
        review_parser.add_argument("path", type=Path, help="input video path")
        review_parser.add_argument("--sample-fps", type=float, default=None)
        review_parser.add_argument(
            "--bed-region",
            type=BedRegion.parse,
            default=None,
            metavar="LEFT,TOP,RIGHT,BOTTOM",
        )
        review_parser.add_argument(
            "--max-segments",
            type=int,
            default=None,
            help="maximum uncertain intervals to review; defaults to the VLM setting",
        )
        event_parser = subparsers.add_parser(
            "analyze-bed-events",
            help="fuse safe activity proposals and confirm bed exit/return transitions",
        )
        CliApplication._add_event_options(event_parser)
        summary_parser = subparsers.add_parser(
            "summarize-observation",
            help="report timeline, durations, bed events, and contextual decision",
        )
        CliApplication._add_event_options(summary_parser)
        return parser

    @staticmethod
    def _add_event_options(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("path", type=Path, help="input video path")
        parser.add_argument("--sample-fps", type=float, default=None)
        parser.add_argument(
            "--bed-region",
            type=BedRegion.parse,
            default=None,
            metavar="LEFT,TOP,RIGHT,BOTTOM",
        )
        parser.add_argument(
            "--with-vlm",
            action="store_true",
            help="review uncertain intervals using local Ollama before event detection",
        )
        parser.add_argument("--max-segments", type=int, default=None)

    def _get_bed_event_service(self, with_vlm: bool) -> BedEventAnalysisService:
        return self._bed_event_service or BedEventAnalysisService(
            self._get_timeline_service(),
            self._settings.temporal,
            self._get_vlm_review_service() if with_vlm else None,
            ConservativeTimelineFusion(self._settings.vlm.minimum_proposal_confidence),
        )

    def _get_vlm_review_service(self) -> VlmReviewService:
        return self._vlm_review_service or VlmReviewService(
            self._get_timeline_service(),
            OpenCvVideoSourceFactory(),
            OpenCvJpegEncoder(),
            OllamaVisionLanguageModel(self._settings.vlm),
            self._settings.vlm,
        )

    def _get_timeline_service(self) -> TimelineAnalysisService:
        return self._timeline_analysis_service or TimelineAnalysisService(
            self._get_vision_service(),
            self._settings.temporal,
            RuleBasedStateClassifier(self._settings.rules),
        )

    def _get_vision_service(self) -> VisionAnalysisService:
        return self._vision_analysis_service or VisionAnalysisService(
            OpenCvVideoSourceFactory(),
            MediaPipePoseEstimatorFactory(),
            self._pose_model_provider or PoseModelStore(),
            self._settings.vision,
        )

    def _write_json(self, payload: object) -> None:
        json.dump(payload, self._output, indent=2)
        self._output.write("\n")


def main() -> None:
    """Minimal composition root used by the installed console command."""

    raise SystemExit(CliApplication(logging_configurator=LoggingConfigurator()).run())
