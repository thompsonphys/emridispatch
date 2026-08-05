NOISE_MODEL = "scirdv1"

MIN_FREQ = 1.0e-5

DEFAULT_TDI_CHANNELS = ["A", "E"]

DIRECT_CHANNELS = ["I", "II"]


def load_sensitivity_table():
    from lisatools.sensitivity import (
        LISASens,
        A1TDISens, E1TDISens, T1TDISens,
        A2TDISens, E2TDISens, T2TDISens,
    )

    return {
        "off": {"I": LISASens, "II": LISASens},
        "1st generation": {"A": A1TDISens, "E": E1TDISens, "T": T1TDISens},
        "2nd generation": {"A": A2TDISens, "E": E2TDISens, "T": T2TDISens},
    }


def sensitivity_spec(tdi, channels=None, sens_table=None):
    if sens_table is None:
        sens_table = load_sensitivity_table()

    gen_sens = sens_table.get(tdi)
    if gen_sens is None:
        raise ValueError(
            f"unknown TDI generation {tdi!r}; choose from "
            f"{sorted(k for k in sens_table if k != 'off')}")

    if tdi == "off":
        names = list(DIRECT_CHANNELS)
    else:
        names = list(DEFAULT_TDI_CHANNELS if channels is None else channels)

    try:
        sens = [gen_sens[c] for c in names]
    except KeyError as err:
        raise ValueError(
            f"unknown TDI channel {err.args[0]!r}; choose from "
            f"{sorted(gen_sens)}") from err
    return sens, names


def noise_sens_kwargs(duration, foreground):
    kwargs = {"model": NOISE_MODEL}
    if foreground:
        from lisatools.utils.constants import YRSID_SI

        kwargs["stochastic_params"] = (duration * YRSID_SI,)
    return kwargs


def channel_noise_psd(f, sens_cls, **kwargs):
    return sens_cls.get_Sn(f, **kwargs)


def per_channel_noise_kwargs(duration, foreground, sens_classes):
    shared = noise_sens_kwargs(duration, foreground)
    return [dict(shared, sens_cls=cls) for cls in sens_classes]
