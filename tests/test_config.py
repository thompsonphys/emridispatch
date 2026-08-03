import pathlib

import pytest
import yaml

from emridispatch.config import INJECTION_KEYS, REQUIRED_SECTIONS, load_config

_INJECTION = {"mass_1": "1.0e6", "mass_2": 10.0, "a": 0.0, "p": 10.0,
              "e": 0.1, "x": 1.0, "q_k": 1.0, "phi_k": 1.5, "q_s": 1.0,
              "phi_s": 1.5, "luminosity_distance": 1.0, "phi_phi": 1.5,
              "phi_theta": 1.5, "phi_r": 1.5}

BASE = {
    # Placeholder for any key added to the schema later, so these tests stay
    # about config parsing rather than about the parameter list.
    "injection": {name: _INJECTION.get(name, 0.5) for name in INJECTION_KEYS},
    "data": {"response": "toy", "duration": 0.3, "delta_t": 10.0,
             "inj_snr": 30.0, "channels": ["A", "E"]},
    "sampler": {"nsamples": 100,
                "impulse": {"threads": 1, "cov_update": 50, "save_freq": 50,
                            "ladder": {"max_temp": 100.0, "t_split": 10.0,
                                       "ntemps_low": 3, "ntemps_high": 2},
                            "mode_jump": {"method": "none", "weight": 25.0}}},
    "prior": {"angle_sigma": 0.05, "fisher_use_gpu": False},
    "reparam": {"mode": "auto", "idx": [0, 1, 2, 3, 4, 5]},
    "run": {"seed": 1, "outdir": "./out"},
}


def write_cfg(tmp_path, extra=None, **section_updates):
    raw = yaml.safe_load(yaml.safe_dump(BASE))  # deep copy
    for key, val in section_updates.items():
        raw.setdefault(key, {}).update(val)
    if extra:
        raw.update(extra)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return str(path)


def test_load_and_coercions(tmp_path):
    cfg = load_config(write_cfg(tmp_path))
    # String "1.0e6" coerced to float.
    assert cfg.injection["mass_1"] == 1e6
    assert isinstance(cfg.injection["mass_1"], float)
    assert cfg.data.response == "toy"
    # Provided keys.
    assert cfg.sampler.nsamples == 100
    assert cfg.sampler.impulse.ladder.max_temp == 100.0
    # Optional-key defaults.
    assert cfg.sampler.backend == "impulse"
    assert cfg.sampler.start_mode == "truth"
    assert cfg.sampler.start_jitter == 5.0
    assert cfg.prior.box_scale == 3.0
    assert cfg.prior.fisher == "auto"
    assert cfg.prior.periodic_2pi_indices == [7, 9, 10, 11]
    assert cfg.priors == {}
    assert cfg.logging.level == "INFO"


@pytest.mark.parametrize("bad", ["foo", None, [1.0]])
def test_a_bad_injection_value_names_its_key(tmp_path, bad):
    """float() alone reports the value and not which of the fourteen keys
    carried it, and for None or a list it reports neither."""
    with pytest.raises(ValueError, match="injection.mass_2"):
        load_config(write_cfg(tmp_path, injection={"mass_2": bad}))


def test_reparam_off_coercion(tmp_path):
    # Bare `off` parses as YAML boolean False -> normalized to "off".
    cfg = load_config(write_cfg(tmp_path, reparam={"mode": False}))
    assert cfg.reparam.mode == "off"


def test_optional_section_overrides(tmp_path):
    cfg = load_config(write_cfg(
        tmp_path,
        extra={"sampler": {"backend": "eryn"},
               "priors": {"dist": {"type": "loguniform"}},
               "logging": {"level": "DEBUG"}}))
    assert cfg.sampler.backend == "eryn"
    assert cfg.priors["dist"]["type"] == "loguniform"
    assert cfg.logging.level == "DEBUG"
    # Shared sampler defaults survive a partial sampler section.
    assert cfg.sampler.nsamples == 10000
    assert cfg.sampler.impulse.ladder.max_temp == 1000.0


def test_backend_subsections_optional(tmp_path):
    # Configs can omit the backend subsections (or the whole sampler section);
    # every knob materializes with its default.
    raw = yaml.safe_load(yaml.safe_dump(BASE))
    del raw["sampler"]
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(str(path))
    assert cfg.sampler.backend == "impulse"
    assert cfg.sampler.nsamples == 10000
    assert cfg.sampler.impulse.threads == 1
    assert cfg.sampler.impulse.ladder.max_temp == 1000.0
    assert cfg.sampler.impulse.mode_jump.method == "none"
    assert cfg.sampler.eryn.nwalkers == 32
    assert cfg.sampler.eryn.move == "stretch"
    # Provided subsections still override the defaults.
    cfg2 = load_config(write_cfg(tmp_path))
    assert cfg2.sampler.impulse.ladder.max_temp == 100.0


def test_partial_subsection_merges_defaults(tmp_path):
    cfg = load_config(write_cfg(
        tmp_path, sampler={"backend": "eryn", "eryn": {"nwalkers": 8}}))
    assert cfg.sampler.eryn.nwalkers == 8
    assert cfg.sampler.eryn.ntemps == 1
    # Sibling subsection untouched.
    assert cfg.sampler.impulse.cov_update == 50


def test_tdi_default(tmp_path):
    cfg = load_config(write_cfg(tmp_path))
    assert cfg.data.tdi == "2nd generation"


def test_tdi_coercions(tmp_path):
    cases = [("off", "off"), (False, "off"),
             ("1st", "1st generation"), ("1st generation", "1st generation"),
             ("2nd", "2nd generation"), ("2nd generation", "2nd generation"),
             (True, "2nd generation")]
    for raw, want in cases:
        cfg = load_config(write_cfg(tmp_path, data={"tdi": raw}))
        assert cfg.data.tdi == want, raw


def test_tdi_invalid(tmp_path):
    with pytest.raises(ValueError, match="data.tdi"):
        load_config(write_cfg(tmp_path, data={"tdi": "3rd"}))


def test_foreground_default(tmp_path):
    cfg = load_config(write_cfg(tmp_path))
    assert cfg.data.foreground is True


def test_foreground_coercions(tmp_path):
    cases = [(True, True), (False, False), ("", False), (1, True), (0, False)]
    for raw, want in cases:
        cfg = load_config(write_cfg(tmp_path, data={"foreground": raw}))
        assert cfg.data.foreground is want, raw


def _write_raw(tmp_path, raw):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return str(path)


@pytest.mark.parametrize("section", sorted(REQUIRED_SECTIONS))
def test_missing_required_section_names_itself(tmp_path, section):
    raw = yaml.safe_load(yaml.safe_dump(BASE))
    del raw[section]
    with pytest.raises(ValueError, match=f"missing config section.*{section}"):
        load_config(_write_raw(tmp_path, raw))


@pytest.mark.parametrize("section", sorted(REQUIRED_SECTIONS))
def test_empty_required_section_is_treated_as_missing(tmp_path, section):
    # `reparam:` with nothing under it parses as None, not {}.
    raw = yaml.safe_load(yaml.safe_dump(BASE))
    raw[section] = None
    with pytest.raises(ValueError, match=f"missing config section.*{section}"):
        load_config(_write_raw(tmp_path, raw))


@pytest.mark.parametrize("section,key", [
    ("reparam", "mode"), ("reparam", "idx"),
    ("run", "outdir"), ("run", "seed"), ("injection", "phi_theta"),
])
def test_missing_required_key_names_itself(tmp_path, section, key):
    raw = yaml.safe_load(yaml.safe_dump(BASE))
    del raw[section][key]
    with pytest.raises(ValueError,
                       match=f"missing key.*{section}.*{key}"):
        load_config(_write_raw(tmp_path, raw))


def test_unknown_top_level_section_is_rejected(tmp_path):
    raw = yaml.safe_load(yaml.safe_dump(BASE))
    raw["sampeler"] = {"nsamples": 5}
    with pytest.raises(ValueError, match="did you mean 'sampler'"):
        load_config(_write_raw(tmp_path, raw))


@pytest.mark.parametrize("section", ["priors", "pp", "logging", "prior", "sampler"])
def test_optional_section_with_an_empty_body_falls_back_to_defaults(
        tmp_path, section):
    # `priors:` with every entry commented out parses as None, not {}.
    raw = yaml.safe_load(yaml.safe_dump(BASE))
    raw[section] = None
    cfg = load_config(_write_raw(tmp_path, raw))
    assert cfg.priors == {}
    assert cfg.logging.level == "INFO"
    assert cfg.sampler.backend == "impulse"
    assert cfg.prior.box_scale == 3.0


def test_empty_config_is_rejected(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("")
    with pytest.raises(ValueError, match="empty or not a YAML mapping"):
        load_config(str(path))


def test_every_example_config_loads():
    """The shipped examples must satisfy the required-section checks."""
    root = pathlib.Path(__file__).resolve().parents[1] / "examples"
    paths = sorted(root.glob("*.yaml"))
    assert paths
    for path in paths:
        load_config(str(path))
