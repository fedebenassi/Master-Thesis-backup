import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta


def bathymetry_filter(data, bath, h = -1000):
    data_filt = xr.where(bath.elevation < h, data, np.nan)
    return data_filt.to_numpy()

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def find_mean(data, start_date, end_date):
    d = []
    for date in daterange(start_date, end_date):
        try:
            d.append(data.sel(time = date.strftime("%Y-%m-%d")).to_numpy())
        except:
            pass
        
    return np.nanmean(np.array(d), axis = 0)

def clima_std(data, start_date, end_date):
    start_month = start_date.month
    start_day = start_date.day

    end_month = end_date.month
    end_day = end_date.day
    d = []
    for year in range(2003,2022):
        start_date = date(year, start_month, start_day)
        end_date = date(year, end_month, end_day)
        d.append(find_mean(data, start_date, end_date))

    return np.nanmean(np.array(d), axis = 0), np.nanstd(np.array(d), axis = 0)
