from __future__ import annotations

from enum import IntEnum, StrEnum


class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Category(StrEnum):
    BUG = "bug"
    SECURITY = "security"
    CORRECTNESS = "correctness"
    CONCURRENCY = "concurrency"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    DATABASE = "database"
    API = "api"
    MAINTAINABILITY = "maintainability"


SEVERITY_RANK: dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
}


class LogLevel(IntEnum):
    QUIET = 0
    INFO = 1
    DEBUG = 2
