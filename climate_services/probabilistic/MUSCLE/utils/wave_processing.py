"""Post-process simulated waves from the climate emulator.

Simulated wave heights and peak periods can reach unrealistic values in the
tail of the fitted distributions. limit_hs and limit_tp cap them at a factor
of the historical maximum.
"""
import numpy as np


def limit_max(simulated_waves, historical, var, factor=1.8, method="clip"):
    """Return a copy of simulated_waves with var not above factor * max(historical).

    method="clip" sets exceeding values to the limit; method="nan" masks them.
    The limit and the number of modified values are stored in the variable attrs.
    """
    if method not in ("clip", "nan"):
        raise ValueError(f"method must be 'clip' or 'nan', got {method!r}")

    limit = factor * float(np.nanmax(historical))
    exceed = simulated_waves[var] > limit

    processed = simulated_waves.copy()
    if method == "clip":
        processed[var] = processed[var].where(~exceed, limit)
    else:
        processed[var] = processed[var].where(~exceed)

    processed[var].attrs.update(
        simulated_waves[var].attrs,
        limit=limit,
        limit_factor=factor,
        n_limited=int(exceed.sum()),
    )
    return processed


def limit_tp(simulated_waves, historical_tp, factor=1.8, var="bulk_Tp", method="clip"):
    """Cap the simulated peak period at factor * max(historical_tp)."""
    return limit_max(simulated_waves, historical_tp, var, factor=factor, method=method)


def limit_hs(simulated_waves, historical_hs, factor=1.8, var="bulk_Hs", method="clip"):
    """Cap the simulated significant wave height at factor * max(historical_hs)."""
    return limit_max(simulated_waves, historical_hs, var, factor=factor, method=method)
