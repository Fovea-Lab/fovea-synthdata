"""Generic plate-profile report: any object-level plate dataset in, a deterministic report out."""

from synthdata.report.contract import DEFAULT_FEATURES, FeatureSpec, ReportError, ReportInputs, validate


def build_report(*args, **kwargs):
    """Build a report; see `synthdata.report.build.build_report`."""
    from synthdata.report.build import build_report as _build

    return _build(*args, **kwargs)


def compute_metrics(inputs):
    """Metrics and effects without writing a report; see `synthdata.report.build.compute_metrics`."""
    from synthdata.report.build import compute_metrics as _cm

    return _cm(inputs)


__all__ = ["compute_metrics", "DEFAULT_FEATURES", "FeatureSpec", "ReportError", "ReportInputs", "validate", "build_report"]
