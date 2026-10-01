import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from elderly_care_agent.application.dataset_ports import DatasetDownloader
from elderly_care_agent.domain.exceptions import VisionModelError
from elderly_care_agent.infrastructure.pose_model_store import PoseModelStore


class PoseModelStoreTests(unittest.TestCase):
    def test_downloads_once_and_reuses_verified_asset(self) -> None:
        payload = b"pose model fixture"
        digest = hashlib.sha256(payload).hexdigest()
        downloader = Mock(spec=DatasetDownloader)
        downloader.download.side_effect = lambda url, path: path.write_bytes(payload)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.task"
            with patch.object(PoseModelStore, "SHA256", digest):
                store = PoseModelStore(path, downloader)
                first = store.ensure_available()
                second = store.ensure_available()

            self.assertEqual(path.resolve(), first)
            self.assertEqual(first, second)
            downloader.download.assert_called_once_with(store.SOURCE_URL, path)

    def test_bad_cached_asset_is_replaced_and_bad_download_is_removed(self) -> None:
        payload = b"correct pose model"
        digest = hashlib.sha256(payload).hexdigest()
        downloader = Mock(spec=DatasetDownloader)
        downloader.download.side_effect = lambda url, path: path.write_bytes(payload)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.task"
            path.write_bytes(b"corrupt")
            with patch.object(PoseModelStore, "SHA256", digest):
                PoseModelStore(path, downloader).ensure_available()
            self.assertEqual(payload, path.read_bytes())

            path.unlink()
            downloader.download.side_effect = lambda url, path: path.write_bytes(b"wrong")
            with (
                patch.object(PoseModelStore, "SHA256", digest),
                self.assertRaisesRegex(VisionModelError, "checksum mismatch"),
            ):
                PoseModelStore(path, downloader).ensure_available()
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
