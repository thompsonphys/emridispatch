"""cli.main had no coverage; these pin the contract the --isolate subprocess
paths in pp.py and multichain.py depend on."""

import logging
import textwrap

import pytest

from conftest import MINIMAL_TOY_CONFIG
from emridispatch import cli


@pytest.fixture(autouse=True)
def _reset_package_logger():
    """main() calls setup_logging, which reconfigures the package logger.

    Its handlers and propagate=False outlive the test: a FileHandler keeps an
    open descriptor on a torn-down tmp_path, and propagate=False leaves every
    later test file relying on caplog attaching to the named logger rather
    than the root.
    """
    yield
    logger = logging.getLogger("emridispatch")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = True
    logger.setLevel(logging.NOTSET)


@pytest.fixture
def config_path(tmp_path):
    outdir = tmp_path / "chains"
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(
        MINIMAL_TOY_CONFIG.format(outdir=str(outdir))))
    return path


def _stub_run(monkeypatch, result):
    """Replace the pipeline entry point; returns the resume flags it saw."""
    seen = []

    def run_from_config(cfg, resume=True):
        seen.append(resume)
        return result

    monkeypatch.setattr("emridispatch.pipeline.run_from_config",
                        run_from_config)
    return seen


def test_a_failed_sampler_exits_nonzero(config_path, monkeypatch):
    """Both backends return None when sample() raises. The subprocess callers
    read the return code, and a shell caller has nothing else to read."""
    _stub_run(monkeypatch, None)
    with pytest.raises(SystemExit) as exc:
        cli.main([str(config_path)])
    assert exc.value.code                      # truthy -> nonzero exit status
    # The full path, not the bare filename: --isolate subprocesses run from a
    # cwd that is not the outdir.
    assert str(config_path.parent / "chains" / "run.log") in str(exc.value.code)
    assert "no run_summary.json written" in str(exc.value.code)


def test_a_failed_resumed_run_does_not_call_a_stale_summary_absent(config_path,
                                                                   monkeypatch):
    """Nothing deletes a previous run_summary.json and resume is the default,
    so a failed re-run leaves one behind that is_complete() reads as success."""
    outdir = config_path.parent / "chains"
    outdir.mkdir(exist_ok=True)
    (outdir / "run_summary.json").write_text("{}")
    _stub_run(monkeypatch, None)
    with pytest.raises(SystemExit) as exc:
        cli.main([str(config_path)])
    assert "earlier run" in str(exc.value.code)
    assert "no run_summary.json written" not in str(exc.value.code)


def test_a_failed_sampler_exits_nonzero_with_no_log_file(config_path,
                                                         monkeypatch):
    """`logging: {file: null}` is a valid way to say "no log file", so the
    failure message must not try to name one."""
    config_path.write_text(
        config_path.read_text() + "\nlogging:\n  file: null\n")
    _stub_run(monkeypatch, None)
    with pytest.raises(SystemExit) as exc:
        cli.main([str(config_path)])
    assert "run_summary.json" in str(exc.value.code)
    assert "See" not in str(exc.value.code)


def test_a_successful_run_exits_cleanly(config_path, monkeypatch):
    _stub_run(monkeypatch, {"config": {"backend": "impulse"}})
    assert cli.main([str(config_path)]) is None


def test_no_resume_reaches_the_pipeline(config_path, monkeypatch):
    seen = _stub_run(monkeypatch, {"config": {}})
    cli.main([str(config_path), "--no-resume"])
    cli.main([str(config_path)])
    assert seen == [False, True]


def test_the_config_is_copied_into_the_outdir(config_path, monkeypatch):
    """postprocess falls back to this copy when a run has no summary, and
    multichain relies on the same per-seed convention."""
    _stub_run(monkeypatch, {"config": {}})
    cli.main([str(config_path)])
    copied = config_path.parent / "chains" / "config.yaml"
    assert copied.exists()
    assert copied.read_text() == config_path.read_text()


def test_a_config_already_in_the_outdir_is_not_copied_onto_itself(tmp_path,
                                                                  monkeypatch):
    """pp --isolate and multichain --isolate both write <outdir>/config.yaml
    and then pass that path as argv, so source and destination are one file."""
    outdir = tmp_path / "chains"
    outdir.mkdir()
    path = outdir / "config.yaml"
    path.write_text(textwrap.dedent(MINIMAL_TOY_CONFIG.format(outdir=str(outdir))))
    _stub_run(monkeypatch, {"config": {}})
    cli.main([str(path)])
    assert f"outdir: {outdir}" in path.read_text()
