import io
import logging
import tempfile
import unittest
from pathlib import Path

from elderly_care_agent.infrastructure.logging_config import LoggingConfigurator


class LoggingConfiguratorTests(unittest.TestCase):
    def test_writes_human_readable_console_and_rotating_file_logs(self) -> None:
        root_logger = logging.getLogger()
        previous_handlers = list(root_logger.handlers)
        previous_level = root_logger.level
        console = io.StringIO()
        temporary_directory = tempfile.TemporaryDirectory()

        try:
            log_file = Path(temporary_directory.name) / "application.log"
            LoggingConfigurator(console).configure("DEBUG", log_file)

            logging.getLogger("elderly_care_agent.test").info("Pipeline event | stage=download")
            for handler in logging.getLogger().handlers:
                handler.flush()

            self.assertIn("Pipeline event | stage=download", console.getvalue())
            self.assertIn(
                "Pipeline event | stage=download",
                log_file.read_text(encoding="utf-8"),
            )
        finally:
            for handler in logging.getLogger().handlers:
                if handler not in previous_handlers:
                    handler.close()
            root_logger.handlers = previous_handlers
            root_logger.setLevel(previous_level)
            temporary_directory.cleanup()


if __name__ == "__main__":
    unittest.main()
