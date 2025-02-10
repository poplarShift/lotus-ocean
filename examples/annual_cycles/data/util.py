import pandas as pd

def _get_epoch(times):
    tmin = times.min()
    return pd.Timestamp(year=tmin.year, month=1, day=1)

def decimal_day_of_year(times, wrap=True):
    """
    Converts pandas datetime series into decimal days of the year.

    January 1 at noon corresponds to ddoy=0.5.

    wrap=False counts days from the earliest date in the series. Pay attention to
    leap years if you do so.
    """
    datatype = type(times)

    if wrap:
        ddoys = []
        for year in times.dt.year.unique():
            times_this_year = [t for t in times if t.year == year]
            ddoys += list(decimal_day_of_year(datatype(times_this_year), wrap=False))
        ddoys = datatype(ddoys)
    else:
        epoch = _get_epoch(times)
        ddoys = (times - epoch)/pd.Timedelta(days=1)

    return ddoys

def interpolate_on_column(df, on_values, col='zz'):
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
