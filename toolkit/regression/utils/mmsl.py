"""Monthly sea-level preprocessing and PC-modulated harmonic regression."""
import numpy as np
import pandas as pd
import xarray as xr


def monthly_sea_level(path, start_year, end_year, minimum_coverage=0.8, trend_fit_before=None):
    """Remove a linear hourly trend and average valid residuals by month.

    Retains seasonality. Coverage is valid unique hourly samples / calendar hours.
    Trend uses elapsed time, so gaps do not compress the time axis.
    """
    with xr.open_dataset(path) as ds:
        if ds.sea_level.attrs.get('units') != 'm':
            raise ValueError('Expected sea level in metres.')
        selected = ds.sel(time=slice(str(start_year), str(end_year))).load()
        level = selected.sea_level.where(selected.flag2 == 1).to_series()
    level = level.replace([np.inf, -np.inf], np.nan).sort_index()
    if level.index.has_duplicates:
        raise ValueError('Duplicate sea-level timestamps.')
    if not (level.index == level.index.floor('h')).all():
        raise ValueError('Expected hourly timestamps for coverage calculation.')
    origin = pd.Timestamp(f'{start_year}-01-01')
    elapsed_years = (level.index - origin).total_seconds().to_numpy() / (365.2425 * 86400)
    valid = level.notna().to_numpy()
    if trend_fit_before is not None:
        valid = valid & (level.index < pd.Timestamp(trend_fit_before))
    if valid.sum() < 2:
        raise ValueError('Insufficient valid sea-level samples.')
    slope, intercept = np.polyfit(elapsed_years[valid], level.to_numpy()[valid], 1)
    trend = pd.Series(intercept + slope * elapsed_years, index=level.index)
    residual = level - trend
    monthly = pd.DataFrame({
        'sea_level_m': level.resample('MS').mean(),
        'trend_m': trend.where(level.notna()).resample('MS').mean(),
        'mmsla_m': residual.resample('MS').mean(),
        'valid_hours': level.resample('MS').count(),
    })
    monthly['coverage'] = monthly.valid_hours / (monthly.index.days_in_month * 24)
    monthly.loc[monthly.coverage < minimum_coverage, 'mmsla_m'] = np.nan
    return monthly, {'slope_m_per_year': float(slope), 'intercept_m': float(intercept),
                     'origin': str(origin), 'minimum_coverage': minimum_coverage,
                     'trend_fit_before': str(trend_fit_before) if trend_fit_before is not None else 'full record'}


def harmonic_design(t, *pcs, n_harmonics=2):
    """Mean block plus sine/cosine pairs, each linearly modulated by the PCs."""
    if not isinstance(n_harmonics, (int, np.integer)) or isinstance(n_harmonics, bool) or n_harmonics < 0:
        raise ValueError('n_harmonics must be a non-negative integer.')
    arrays = np.broadcast_arrays(t, *pcs)
    t = arrays[0]
    base = np.column_stack([np.ones(t.size)] + [pc.ravel() for pc in arrays[1:]])
    harmonics = [np.ones(t.size)]
    for k in range(1, n_harmonics + 1):
        harmonics.extend([np.cos(2*np.pi*k*t).ravel(), np.sin(2*np.pi*k*t).ravel()])
    return np.column_stack([base * h[:, None] for h in harmonics])


def modelfun(x, t, *pcs, y, n_harmonics=2):
    """Residual of the PC-modulated harmonic model, with any number of PCs."""
    return harmonic_design(t, *pcs, n_harmonics=n_harmonics) @ np.asarray(x) - np.asarray(y)


def fit_mmsl(monthly, annual_pcs, fit_before=None, n_harmonics=2):
    """Align calendar-year PCs and solve a model linear in its coefficients."""
    annual = annual_pcs.copy()
    pc_columns = list(annual.columns)
    if not pc_columns:
        raise ValueError('Select at least one PC.')
    annual = annual.replace([np.inf, -np.inf], np.nan)
    if annual.index.has_duplicates:
        raise ValueError('Duplicate PC years.')
    table = monthly.join(annual, on=monthly.index.year)
    table = table.dropna(subset=['mmsla_m', *pc_columns]).copy()
    # Fractional calendar year at the month-start label, matching the reference.
    starts = pd.to_datetime(table.index.year.astype(str) + '-01-01')
    ends = pd.to_datetime((table.index.year + 1).astype(str) + '-01-01')
    table['t'] = np.asarray((table.index - starts) / (ends - starts))
    design = harmonic_design(table.t, *[table[col] for col in pc_columns], n_harmonics=n_harmonics)
    fit_mask = np.ones(len(table), dtype=bool) if fit_before is None else table.index < pd.Timestamp(fit_before)
    coefficients, _, rank, singular = np.linalg.lstsq(design[fit_mask], table.mmsla_m.to_numpy()[fit_mask], rcond=None)
    if fit_mask.sum() <= design.shape[1] or rank != design.shape[1]:
        raise ValueError(f'Insufficient data or rank-deficient design: {len(table)} months, rank {rank}.')
    table['fitted_m'] = design @ coefficients
    table['residual_m'] = table.mmsla_m - table.fitted_m
    fit_table = table.loc[fit_mask]
    y = fit_table.mmsla_m.to_numpy()
    mse = np.mean(fit_table.residual_m**2)
    metrics = {'n_months': len(fit_table), 'n_pcs': len(pc_columns), 'n_harmonics': n_harmonics, 'n_parameters': design.shape[1], 'rank': int(rank),
               'RMSE_m': float(np.sqrt(mse)), 'MAE_m': float(np.mean(np.abs(fit_table.residual_m))),
               'R2': float(1 - np.sum(fit_table.residual_m**2) / np.sum((y-y.mean())**2)),
               'correlation': float(np.corrcoef(y, fit_table.fitted_m)[0, 1]),
               'condition_number': float(singular[0] / singular[-1])}
    return table, coefficients, metrics
