from enum import Enum, auto
from datetime import datetime
import enum
from typing import Optional

from numpy import log


class LogLevel(Enum):
    INFO = auto()
    DEBUG = auto()
    ERROR = auto()
    WARNING = auto()


class Verbosity(Enum):
    INFO = auto()
    DEBUG = auto()


log_level_map = ["INFO", "DEBUG", "ERROR", "WARNING"]


class Logging:
    def __init__(
        self,
        file_name,
        verbosity: Verbosity = Verbosity.INFO,
        user_name: Optional[str] = "myapp",
        show_in_console=False,
    ) -> None:
        self.name = user_name
        self.file_name = file_name
        self.verbos = verbosity
        self.show_in_console = show_in_console

    def _log(
        self,
        log_level: LogLevel,
        message: str,
        *args,
        **keywords,
    ):
        if self.verbos == Verbosity.INFO and log_level.value > self.verbos.value:
            return
        formatted_message = self._formatting_message(
            log_level, message, *args, **keywords
        )

        self._write_to_file(formatted_message)
        if self.show_in_console:
            self._write_to_console(formatted_message)

    def _formatting_message(
        self, log_level: LogLevel, message: str, *args, **keywords
    ) -> str:
        formatted_content = message.format(*args, **keywords)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}] [{log_level.name}] {self.name} - {formatted_content}"

    def _write_to_console(self, message: str) -> None:
        print(message)

    def _write_to_file(self, message: str) -> None:
        with open(self.file_name, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def info(
        self,
        message: str,
        *args,
        **keywords,
    ) -> None:
        self._log(LogLevel.INFO, message, *args, **keywords)

    def debug(self, message: str, *args, **keywords) -> None:
        self._log(LogLevel.DEBUG, message, *args, **keywords)

    def warning(self, message: str, *args, **keywords) -> None:
        self._log(LogLevel.WARNING, message, *args, **keywords)

    def error(self, message: str, *args, **keywords) -> None:
        self._log(LogLevel.ERROR, message, *args, **keywords)
