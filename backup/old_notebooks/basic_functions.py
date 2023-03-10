import numpy as np
import xarray as xr
from cartopy.feature import NaturalEarthFeature
from cartopy.crs import PlateCarree
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def call_projection():
    """Calls the PlateCarree projection for plots"""
    global states, cart_proj
    states = NaturalEarthFeature(category="cultural", scale="50m",
                             facecolor="none",
                             name="admin_0_boundary_lines_land")
    cart_proj = PlateCarree()
    return states, cart_proj

def time_preprocess(data):
    data["time"] = data["time"].dt.strftime("%Y-%m-%d").astype(np.datetime64)
    return data

def import_data(fname):
    """.nc file importation as a XArray
    ...
    Args:
    - fname = (str) path of the .nc file
    ...
    Returns:
    - data = imported data
    """
    data = xr.open_dataset(fname)
    return data

def extract_lonlats(data):
    """Grid extraction
    ...
    Args:
    - data = XArray file
    ...
    Returns:
    - lons, lats: longitude and latitude XArrays"""

    lons = data.lon
    lats = data.lat
    return lons, lats

def interpolate_data(data, lons, lats, method = "nearest"):
    """Data interpolation on the chosen grid
    ...
    Args:
    - data = Xarray file to interpolate
    - lons = Xarray file with longitudes
    - lats = Xarray file with latitudes
    - method = (str) nearest by default
    ...
    Returns:
    - data_interp = interpolated data"""

    data_interp = data.interp(lon = lons, lat = lats, method = method)
    return data_interp

def bathymetry_filter(data, bath, h = -1000):
    """ Removes coastal data (depth > h)
    ...
    Args:
    - a = NumPy array with the same grid of bathymetry
    - bath = NumPy array of bathymetry
    - h = reference depth (must be < 0) 
    ...
    Returns:
    - a_filt = NumPy array with coastal data removed"""

    data_filt = xr.where(bath.elevation < h, data, np.nan)
    return data_filt

def plot_data(lons, lats, data, cmap, fig = None, ax = None, cart_proj = PlateCarree()):

    if ax == None:
        fig = plt.figure()
        ax = plt.axes(projection = cart_proj)

    ax.add_feature(states, linewidth=.5, edgecolor="black")
    ax.coastlines('50m', linewidth=0.8)
    
    ax.set_xticks(np.arange(108,115,2), crs=cart_proj)
    ax.set_yticks(np.arange(10,20,2), crs=cart_proj)
    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)

    cmap = get_cmap(cmap, 10)
    c = ax.contourf(lons, lats, data, cmap = cmap)
    # fig.colorbar(c)

    return ax

def climatology(data):
    data_clima = data.groupby("time.dayofyear").mean("time").rename({"dayofyear" : "time"})
    return data_clima

