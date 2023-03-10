from scipy.stats import skewnorm, weibull_min, exponweib, skew
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cartopy.crs import PlateCarree
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator, LongitudeLocator)
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes


def make_grid(ax, font_size):
        gl = ax.gridlines(linewidth=1, color='gray', alpha=0.5,
                    linestyle='--',draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False

        gl.xlabel_style = {'fontsize': font_size}
        gl.ylabel_style = {'fontsize': font_size}

        return gl

def define_coast(ax, lons, lats, bath):
    if type(ax) != np.ndarray:
        ax.contourf(lons, lats, bath, colors = "darkgray", zorder = -1)
    else:
        for a in ax:
            a.contourf(lons, lats, bath, colors = "darkgray", zorder = -1)
    return ax


def set_plot(figsize = [7,7], nrows = 1, ncols = 1):
    fig, ax = plt.subplots(subplot_kw = dict(projection = PlateCarree()), figsize = figsize, nrows = nrows, ncols = ncols)
    
    font_size = 12.5
    if nrows == 1 and ncols == 1:
        ax.coastlines(resolution="10m", linewidths=0.5)
        ax.add_feature(cfeature.LAND.with_scale("10m"),
                edgecolor='lightgray',facecolor='lightgray',
                zorder=0)

        ax.tick_params(axis = "both", labelsize = font_size)
        gl = make_grid(ax, font_size)

    else:
        ax = ax.ravel()
        for a in ax:
            a.coastlines(resolution="10m", linewidths=0.5)
            a.add_feature(cfeature.LAND.with_scale("10m"),
                edgecolor='lightgray',facecolor='lightgray',
                zorder=0)

            a.tick_params(axis = "both", labelsize = font_size)
            gl = make_grid(a, font_size)
    
    return fig, ax


def plot_histogram(data, lon_points, lat_points, fit_func = weibull_min, point = False, region = False):
    lons = data.lon
    lats = data.lat
    if point:

        time_series = data.sel(lon = lon_points, lat = lat_points)
        ts = time_series.to_numpy()[~np.isnan(time_series)]

    if region: 
        mask_lon = (data.lon >= lon_points[0]) & (data.lon <= lon_points[1])
        mask_lat = (data.lat >= lat_points[0]) & (data.lat <= lat_points[1])
        time_series = data.where(mask_lon & mask_lat, drop = True)
        ts = time_series.to_numpy().flatten()[~np.isnan(time_series.to_numpy().flatten())]
        
    
    fig, ax = plt.subplots()

    ax.hist(ts, density=True, bins="auto", alpha=0.7, label = "Entries")

    xmin, xmax = ax.set_xlim()

    x = np.linspace(xmin, xmax, 300)
    fit = fit_func.fit(ts)
    
    if fit_func == weibull_min:
        name = "Weibull fit"
        textstr = '\n'.join((
        r'$\mathrm{shape} =%.2f$' % (fit[0], ),
        r'$\mathrm{loc}=%.2f$' % (fit[1], ),
        r'$\mathrm{scale}=%.2f$' % (fit[2], )))
        x_text = 0.75

    if fit_func == skewnorm:
        name = "Skew normal fit"
        skewness = skew(ts)
        textstr = '\n'.join((
        r'$\mathrm{skewness} =%.2f$' % (skewness, ),
        r'$\mathrm{loc}=%.2f$' % (fit[1], ),
        r'$\mathrm{scale}=%.2f$' % (fit[2], )))
        x_text = 0.65

    ax.plot(x, fit_func.pdf(x, *fit), color = "red", linewidth = 2, label = name)
    ax.grid()
    ax.set_ylabel("Density")
    ax.legend()

    # place a text box in upper left in axes coords

    ax.text(x_text, 0.8, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=1))


    axins = inset_axes(ax, width="50%", height="50%", loc="right", borderpad = -14, 
                        axes_class=GeoAxes, 
                        axes_kwargs=dict(map_projection=PlateCarree()))
    axins.add_feature(cfeature.COASTLINE)
    axins.add_feature(cfeature.LAND, color = "white")
    axins.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])

    axins.patch.set_facecolor(color='blue')
    axins.patch.set_alpha(0.5)

    if point:
        x, y = lon_points, lat_points
        axins.scatter(lon_points,lat_points,color='black', transform = PlateCarree(), marker = "x")


    if region:
        x=[lon_points[0],lon_points[1],lon_points[1],lon_points[0],lon_points[0]]
        y=[lat_points[1],lat_points[1],lat_points[0],lat_points[0],lat_points[1]]
        axins.plot(x,y,color='black', transform = PlateCarree())

    return fig, ax

def plot_time_series(data, lon_points, lat_points, point = False, region = False):
    lons = data.lon
    lats = data.lat
    if point:

        time_series = data.sel(lon = lon_points, lat = lat_points)
        ts = time_series.to_numpy()[~np.isnan(time_series)]

    if region: 
        mask_lon = (data.lon >= lon_points[0]) & (data.lon <= lon_points[1])
        mask_lat = (data.lat >= lat_points[0]) & (data.lat <= lat_points[1])
        time_series = data.where(mask_lon & mask_lat, drop = True).mean(dim = ["lon", "lat"])
        ts = time_series.to_numpy().flatten()[~np.isnan(time_series.to_numpy().flatten())]


    fig, ax = plt.subplots()

    ax.scatter(time_series.time, time_series, s = 2)
    ax.grid()
    ax.set_xlabel("Time")

    axins = inset_axes(ax, width="50%", height="50%", loc="right", borderpad = -14, 
                            axes_class=GeoAxes, 
                            axes_kwargs=dict(map_projection=PlateCarree()))
    axins.add_feature(cfeature.COASTLINE)
    axins.add_feature(cfeature.LAND, color = "white")
    axins.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])

    axins.patch.set_facecolor(color='blue')
    axins.patch.set_alpha(0.5)

    if point:
        x, y = lon_points, lat_points
        axins.scatter(lon_points,lat_points,color='black', transform = PlateCarree(), marker = "x")

    if region:
        x=[lon_points[0],lon_points[1],lon_points[1],lon_points[0],lon_points[0]]
        y=[lat_points[1],lat_points[1],lat_points[0],lat_points[0],lat_points[1]]
        axins.plot(x,y,color='black', transform = PlateCarree())

    return fig, ax