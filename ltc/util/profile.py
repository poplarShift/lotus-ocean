import numpy as np
import gsw

try:
    import pandas as pd
except ImportError:
    pd = None


def interpolate_df_on_column(df, on_values, col='zz'):
    """
    Interpolate a dataframe `df` on values `on_values` of column `col`.
    """

    df = (
        df
        .set_index(col)
        .merge(pd.DataFrame(None, pd.Index(on_values, name=col), dtype=float), left_index=True, right_index=True, how='outer', sort=True)
        .interpolate(method='index') # interpolate on index values
        .fillna(method='bfill') # pad uppermost values
        .reindex(on_values) # only retain the new indices
        .reset_index()
    )
    return df


def convert_ctd(S, T, depth, hice, lon, lat, ts_units, depth_unit):
    """
    TBD docstring
    Convert CTD profile to suitable form

    Arguments
    ---------
        S: Salinity, units as below
        T: Temperature, units as below
        depth: Depth, positive downward, units as below
        lon: Longitude (degrees East)
        lat: Latitude (degrees North)
        hice: Ice draft (m)
        ts_units: {'SP/PT', 'SA/CT'} PSU/potential temperature or absolute salinity/conservative temperature
        depth_unit: {'pres', 'depth', 'distance'} pressure (dbar), depth (m), or distance from ice (m)

    Returns
    -------
        SA: Absolute salinity
        CT: Conservative temperature
        z: Depth below sea ice, positive downward
        p: Pressure
    """

    if depth_unit == 'pres':
        p = depth
        z = -gsw.z_from_p(depth, lat)
    elif depth_unit == 'depth':
        p = gsw.p_from_z(-depth, lat)
        z = depth - hice
    elif depth_unit == 'distance':
        p = gsw.p_from_z(-hice-depth, lat)
        z = depth

    if ts_units == 'SP/PT':
        SA = gsw.SA_from_SP(S, p, lon, lat)
        CT = gsw.CT_from_pt(SA, T)
    elif ts_units == 'SA/CT':
        SA = S
        CT = T

    return SA, CT, z, p


def make_staggered_grid(deltaz, depth):
    """
    Create staggered grid (z, zz) where z[0]=0, and zz are halfway between z.

    Arguments
    ---------
        deltaz: Vertical grid step (m)
        depth: Maximum depth (m)

    Returns
    -------
        z: grid for gradients and fluxes
        zz: mean quantity grid
    """

    z = np.arange(0, depth, deltaz)
    if z[-1]+ (z[-1]-z[-2]) == depth:
        z = np.r_[z, depth]

    dz = np.diff(z)
    zz = np.r_[
        z[0] + 0.5*dz[0],
        z[1:] + 0.5*dz,
    ]

    return z, zz
