import tempfile
import unittest
from pathlib import Path

from elderly_care_agent.configuration.dataset_configuration import (
    DatasetConfigurationManager,
)
from elderly_care_agent.domain.exceptions import DatasetConfigurationError


class DatasetConfigurationManagerTests(unittest.TestCase):
    def test_loads_pinned_source_and_honours_resource_override(self) -> None:
        resource_dir = Path("X:/portable-dataset-cache")

        config = DatasetConfigurationManager(
            Path("config/dataset.yml"),
            environment={"GMDCSA24_CACHE_DIR": str(resource_dir)},
        ).load()

        self.assertEqual(resource_dir / "GMDCSA24-v2.0.zip", config.archive_path)
        self.assertEqual(resource_dir / "source", config.source_dir)
        self.assertEqual("Ekram Alam", config.source.owner)
        self.assertEqual(40, len(config.source.mirror_commit))
        self.assertEqual("CC-BY-4.0", config.source.dataset_license)
        self.assertTrue(config.selection_manifest.is_absolute())

    def test_default_cache_is_inside_project_data_directory(self) -> None:
        config = DatasetConfigurationManager(
            Path("config/dataset.yml"),
            environment={},
        ).load()

        project_root = Path.cwd().resolve()
        self.assertEqual(
            project_root / "data/cache/gmdcsa24/GMDCSA24-v2.0.zip",
            config.archive_path,
        )
        self.assertEqual(project_root / "data/cache/gmdcsa24/source", config.source_dir)

    def test_rejects_non_mapping_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "dataset.yml"
            config_path.write_text("- not-a-mapping\n", encoding="utf-8")

            with self.assertRaisesRegex(DatasetConfigurationError, "root must be a mapping"):
                DatasetConfigurationManager(config_path).load()


if __name__ == "__main__":
    unittest.main()
