
from hydromt_sfincs import SfincsModel, utils
import xarray as xr

def postprocess_case(case_dir, dep_hr, mhhw):
    sf = SfincsModel(case_dir, mode="r")
    sf.read_results()

    zsmax = sf.results["zsmax"].max(dim="timemax")

    dep = dep_hr["band_data"].squeeze()

    zsmax_hr = utils.downscale_floodmap(
        zsmax=zsmax,
        dep=dep,
    )

    # Mask below MHHW, set dry land above MHHW to 0
    zsmax_hr = zsmax_hr.where(dep >= mhhw)
    zsmax_hr = zsmax_hr.fillna(0).where(dep >= mhhw)

    # Crop
    zsmax_hr = zsmax_hr.sel(
        x=slice(zsmax.x.min(), zsmax.x.max()),
        y=slice(zsmax.y.max(), zsmax.y.min()),
    )

    return zsmax_hr.rename("zsmax").to_dataset()


def postprocess_cases(cases_dir, dep_path, mhhw, n_cases=5):
    dep_hr = xr.open_dataset(dep_path)

    return [
        postprocess_case(f"{cases_dir}/{i:04d}", dep_hr, mhhw)
        for i in range(n_cases)
    ]
