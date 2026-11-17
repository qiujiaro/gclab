"""Shared result schema for simulated collective benchmarks."""

REQUIRED_COLUMNS = (
    "mode",
    "collective",
    "algorithm",
    "topology",
    "nranks",
    "count",
    "bytes",
    "chunk_bytes",
    "latency_us",
    "algbw_GBps",
    "busbw_GBps",
)


def validate_rows(rows):
    """Validate benchmark rows before plotting."""
    # TODO(M2): reject missing/extra columns, invalid units/types, non-positive
    # timings, and any mode other than SIMULATED for the M2 benchmark.
    del rows
    raise NotImplementedError("TODO(M2): validate_rows")
