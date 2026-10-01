"""Composition root for the dataset-preparation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from elderly_care_agent.components.data_ingestion import DataIngestion
from elderly_care_agent.configuration.dataset_configuration import (
    DatasetConfigurationManager,
)
from elderly_care_agent.domain.dataset import DatasetPreparationReport
from elderly_care_agent.infrastructure.csv_dataset_selection_repository import (
    CsvDatasetSelectionRepository,
)
from elderly_care_agent.infrastructure.dataset_io import (
    HttpDatasetDownloader,
    OpenCvVideoProbe,
    SafeZipDatasetExtractor,
)

logger = logging.getLogger(__name__)


class DatasetPreparationPipeline:
    """Small stage facade matching the project's end-to-end pipeline style."""

    def __init__(self, ingestion: DataIngestion) -> None:
        self._ingestion = ingestion

    @classmethod
    def from_config(
        cls,
        config_path: Path = Path("config/dataset.yml"),
    ) -> DatasetPreparationPipeline:
        """Build production adapters from one version-controlled config file."""

        logger.info("Building dataset preparation pipeline | config=%s", config_path)
        config = DatasetConfigurationManager(config_path).load()
        return cls(
            DataIngestion(
                config=config,
                downloader=HttpDatasetDownloader(),
                extractor=SafeZipDatasetExtractor(),
                selections=CsvDatasetSelectionRepository(),
                video_probe=OpenCvVideoProbe(),
            )
        )

    def run(self, force: bool = False) -> DatasetPreparationReport:
        """Execute the complete dataset stage."""

        logger.info("Dataset pipeline started | force=%s", force)
        report = self._ingestion.run(force=force)
        logger.info(
            "Dataset pipeline finished | status=%s | selected=%d | copied=%d | reused=%d",
            report.status,
            report.selected_video_count,
            report.copied_video_count,
            report.reused_video_count,
        )
        return report
