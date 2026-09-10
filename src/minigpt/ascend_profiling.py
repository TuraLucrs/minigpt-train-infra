"""Compatibility imports for the frozen v0.7.1/v0.8 Ascend profiler API.

The production implementation lives in ``minigpt.backends.profile_ascend``.
Existing callers continue to read and write the original schema version 1.
"""

from .backends.profile_ascend import (
    PROFILE_SCHEMA_VERSION,
    REQUIRED_LEVEL1_ARTIFACTS,
    AscendProfileProtocol,
    AscendStepProfiler,
    analyze_profile_manifest,
    build_profile_manifest,
    load_profile_manifest,
    parse_profile_ranks,
    write_profile_manifest,
)

__all__ = [
    "PROFILE_SCHEMA_VERSION", "REQUIRED_LEVEL1_ARTIFACTS",
    "AscendProfileProtocol", "AscendStepProfiler", "analyze_profile_manifest",
    "build_profile_manifest", "load_profile_manifest", "parse_profile_ranks",
    "write_profile_manifest",
]
