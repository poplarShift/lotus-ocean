import os
import numpy as np
import xarray as xr
import pandas as pd

from ..util import decimal_day_of_year

cwd = os.path.dirname(__file__)

def load_wind():
    df = xr.load_dataset(cwd+'/CCMP/CCMP_2017-19.nc').to_dataframe()
    df['windspeed10'] = np.sqrt(df.uwnd**2 + df.vwnd**2)
    df = df.reset_index()
    df = df.rename(columns={'time': 'Date'})

    return df[['Date', 'windspeed10']]

wind = load_wind()

def load_ice():
    df = pd.read_csv(cwd+'/AMSR2_timeseries.csv', parse_dates=['Date'])
    return df

ice = load_ice()

def load_heatflux():
    df = pd.read_csv(cwd+'/ERA-Interim-heatflux-timeseries.csv', parse_dates=['Date'])
    df = df.set_index('Date')
    df = df.drop(columns=['dparcs_dt', 'parcs'])

    rhocp = 4e6 # Seawater density times specific heat capacity
    df /= rhocp # from W/m2 to K m/s

    return df.reset_index()

heatflux = load_heatflux()




dataframes = [wind, ice, heatflux]
for df in dataframes:
    df['doy'] = decimal_day_of_year(df.Date, wrap=False)


df = (
    wind.merge(ice, on='doy', how='outer')
    .merge(heatflux, on='doy', how='outer')
)
df = df.merge(
    pd.Series(np.arange(df.doy.min(), df.doy.max(), 0.25), name='doy'),
    how='outer',
)

forcing = (
    df
    .drop(columns=df.filter(like='Date').columns)
    .set_index('doy').sort_index()
    .interpolate().interpolate(limit_area='outside', limit_direction='backward')
    .reset_index()
)

# CTD data

def load_ctd():
    df = pd.read_csv(cwd+'/CTD/argo_floats.csv', parse_dates=['Date'])
    df['time'] = decimal_day_of_year(df.Date, wrap=False)
    ds = df.drop(columns=['Date']).set_index(['time', 'Depth']).to_xarray()
    return ds

ctd = load_ctd()
