"""Filesystem, network, archive, and media adapters for dataset ingestion."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import shutil
import stat
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import cv2

from elderly_care_agent.application.dataset_ports import (
    DatasetDownloader,
    DatasetExtractor,
    VideoProbe,
)
from elderly_care_agent.domain.dataset import VideoProbeResult
from elderly_care_agent.domain.exceptions import DatasetIntegrityError

logger = logging.getLogger(__name__)


class FileHasher:
    """Calculates streaming digests without loading large videos into memory."""

    @staticmethod
    def digest(path: Path, algorithm: str = "sha256") -> str:
        hasher = hashlib.new(algorithm)
        with path.open("rb") as source_file:
            for block in iter(lambda: source_file.read(1024 * 1024), b""):
                hasher.update(block)
        return hasher.hexdigest()

    @classmethod
    def verify(cls, path: Path, expected: str) -> None:
        algorithm, expected_digest = expected.split(":", maxsplit=1)
        logger.info(
            "Checksum verification started | file=%s | algorithm=%s | size=%s",
            path,
            algorithm,
            HttpDatasetDownloader.format_bytes(path.stat().st_size),
        )
        actual_digest = cls.digest(path, algorithm)
        if actual_digest != expected_digest:
            logger.error(
                "Checksum verification failed | file=%s | expected=%s | actual=%s:%s",
                path,
                expected,
                algorithm,
                actual_digest,
            )
            raise DatasetIntegrityError(
                f"checksum mismatch for {path}: expected {expected}, "
                f"got {algorithm}:{actual_digest}"
            )
        logger.info("Checksum verified | file=%s | digest=%s:%s", path, algorithm, actual_digest)


class HttpDatasetDownloader(DatasetDownloader):
    """Streams a remote archive to a temporary file, then renames atomically."""

    def __init__(self, max_attempts: int = 3) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero")
        self._max_attempts = max_attempts

    def download(self, source_url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._remove_stale_parts(destination)
        request = urllib.request.Request(
            source_url,
            headers={"User-Agent": "Elderly-Care-Agent/0.1 dataset-pipeline"},
        )
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            logger.info(
                "Download attempt started | attempt=%d/%d | source=%s | destination=%s",
                attempt,
                self._max_attempts,
                source_url,
                destination,
            )
            try:
                self._download_once(request, destination)
                logger.info(
                    "Download completed | destination=%s | size=%s",
                    destination,
                    self.format_bytes(destination.stat().st_size),
                )
                return
            except Exception as error:
                last_error = error
                logger.warning(
                    "Download attempt failed | attempt=%d/%d | error=%s",
                    attempt,
                    self._max_attempts,
                    error,
                )
                if attempt < self._max_attempts:
                    delay_sec = min(2 ** (attempt - 1), 4)
                    logger.info("Download retry scheduled | delay_sec=%d", delay_sec)
                    time.sleep(delay_sec)
        raise DatasetIntegrityError(
            f"download failed after {self._max_attempts} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _download_once(request: urllib.request.Request, destination: Path) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".part",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                    expected_size = HttpDatasetDownloader._content_length(response)
                    downloaded_size = 0
                    next_progress_size = (
                        max(expected_size // 20, 1)
                        if expected_size is not None
                        else 64 * 1024 * 1024
                    )
                    last_progress_time = time.monotonic()
                    while block := response.read(1024 * 1024):
                        temporary_file.write(block)
                        downloaded_size += len(block)
                        now = time.monotonic()
                        if downloaded_size >= next_progress_size or now - last_progress_time >= 10:
                            HttpDatasetDownloader._log_progress(
                                downloaded_size,
                                expected_size,
                            )
                            while downloaded_size >= next_progress_size:
                                next_progress_size += (
                                    max(expected_size // 20, 1)
                                    if expected_size is not None
                                    else 64 * 1024 * 1024
                                )
                            last_progress_time = now
            if expected_size is not None and downloaded_size != expected_size:
                raise DatasetIntegrityError(
                    f"incomplete download: expected {expected_size} bytes, "
                    f"received {downloaded_size} bytes"
                )
            os.replace(temporary_path, destination)
        except BaseException:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _remove_stale_parts(destination: Path) -> None:
        prefix = f".{destination.name}."
        for path in destination.parent.iterdir():
            if path.is_file() and path.name.startswith(prefix) and path.name.endswith(".part"):
                logger.warning("Removing stale partial download | path=%s", path)
                path.unlink()

    @staticmethod
    def _log_progress(downloaded_size: int, expected_size: int | None) -> None:
        if expected_size is None:
            logger.info(
                "Download progress | received=%s | total=unknown",
                HttpDatasetDownloader.format_bytes(downloaded_size),
            )
            return
        percentage = downloaded_size / expected_size * 100
        logger.info(
            "Download progress | percent=%.1f%% | received=%s | total=%s",
            percentage,
            HttpDatasetDownloader.format_bytes(downloaded_size),
            HttpDatasetDownloader.format_bytes(expected_size),
        )

    @staticmethod
    def format_bytes(size: int) -> str:
        value = float(size)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if value < 1024 or unit == "TiB":
                return f"{value:.1f} {unit}"
            value /= 1024
        raise AssertionError("unreachable")

    @staticmethod
    def _content_length(response: object) -> int | None:
        headers = getattr(response, "headers", None)
        value = headers.get("Content-Length") if headers is not None else None
        if value is None:
            return None
        try:
            size = int(value)
        except (TypeError, ValueError) as error:
            raise DatasetIntegrityError(f"invalid Content-Length header: {value}") from error
        if size < 0:
            raise DatasetIntegrityError(f"invalid Content-Length header: {value}")
        return size


class SafeZipDatasetExtractor(DatasetExtractor):
    """Extracts ZIP files while rejecting traversal paths and symbolic links."""

    def extract(self, archive_path: Path, destination: Path) -> None:
        if destination.exists():
            raise DatasetIntegrityError(f"extraction destination already exists: {destination}")
        destination.mkdir(parents=True)
        destination_root = destination.resolve()

        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = archive.infolist()
                logger.info(
                    "Archive extraction started | archive=%s | destination=%s | members=%d",
                    archive_path,
                    destination,
                    len(members),
                )
                strip_root = self._shared_root(members)
                for index, member in enumerate(members, start=1):
                    self._extract_member(archive, member, destination_root, strip_root)
                    if index % 25 == 0 or index == len(members):
                        logger.info(
                            "Archive extraction progress | completed=%d/%d",
                            index,
                            len(members),
                        )
        except (OSError, zipfile.BadZipFile) as error:
            raise DatasetIntegrityError(f"cannot extract {archive_path}: {error}") from error
        logger.info("Archive extraction completed | destination=%s", destination)

    @staticmethod
    def _shared_root(members: list[zipfile.ZipInfo]) -> str | None:
        paths = [
            PurePosixPath(member.filename.replace("\\", "/")).parts
            for member in members
            if member.filename.strip("/\\")
        ]
        first_parts = {parts[0] for parts in paths if parts}
        return next(iter(first_parts)) if len(first_parts) == 1 else None

    @staticmethod
    def _extract_member(
        archive: zipfile.ZipFile,
        member: zipfile.ZipInfo,
        destination_root: Path,
        strip_root: str | None,
    ) -> None:
        member_path = PurePosixPath(member.filename.replace("\\", "/"))
        parts = member_path.parts
        if member_path.is_absolute() or ".." in parts:
            raise DatasetIntegrityError(f"unsafe ZIP member path: {member.filename}")
        if stat.S_ISLNK(member.external_attr >> 16):
            raise DatasetIntegrityError(f"symbolic links are not allowed: {member.filename}")
        if strip_root is not None and parts and parts[0] == strip_root:
            parts = parts[1:]
        if not parts:
            return

        output_path = destination_root.joinpath(*parts).resolve()
        if not output_path.is_relative_to(destination_root):
            raise DatasetIntegrityError(f"unsafe ZIP destination: {member.filename}")
        if member.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source_file, output_path.open("wb") as destination_file:
            shutil.copyfileobj(source_file, destination_file, length=1024 * 1024)


class OpenCvVideoProbe(VideoProbe):
    """Uses the production decoder to reject missing or unreadable video files."""

    def inspect(self, video_path: Path) -> VideoProbeResult:
        logger.debug("Video validation started | path=%s", video_path)
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise DatasetIntegrityError(f"OpenCV cannot open video: {video_path}")
            readable, _ = capture.read()
            if not readable:
                raise DatasetIntegrityError(f"OpenCV cannot decode first frame: {video_path}")
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()

        if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
            raise DatasetIntegrityError(f"invalid video metadata: {video_path}")
        result = VideoProbeResult(
            frame_count=frame_count,
            fps=round(fps, 6),
            duration_sec=round(frame_count / fps, 6),
        )
        logger.debug(
            "Video validation completed | path=%s | frames=%d | fps=%.3f | duration=%.3f",
            video_path,
            result.frame_count,
            result.fps,
            result.duration_sec,
        )
        return result
