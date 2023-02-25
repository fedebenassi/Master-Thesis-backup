import numpy as np
import xarray as xr
from sklearn.cluster import KMeans

def data_importer(folder = "SCS", datatype = "CMEMS", ext = False):
    if ext:
        
        fname = "../data/" + folder + "/" + datatype + "_chl_extended.nc"
        chl_data = xr.open_dataset(fname)

        fname = "../data/" + folder + "/" + datatype + "_temp_extended.nc"
        temp_data = xr.open_dataset(fname)
    else:
    
        fname = "../data/" + folder + "/" + datatype + "_chl.nc"
        chl_data = xr.open_dataset(fname)

        fname = "../data/" + folder + "/" + datatype + "_temp.nc"
        temp_data = xr.open_dataset(fname)
    
    fname = "../data/" + folder + "/bathymetry.nc"
    bath = xr.open_dataset(fname)
    
    
    return chl_data, temp_data, bath, datatype

def data_extractor(date="all"):
    """ Returns chlorophyll, sst and bathymetry interpolated to the
    larger grid (for this dataset, temperature) """
    
    chl_data, temp_data, bath, datatype = data_importer()
    if datatype == "CMEMS":
        temp_data["time"] = np.datetime_as_string(temp_data["time"], unit='D')

        if date == "all":
            chl = chl_data.CHL
            temp = temp_data.analysed_sst

        else: 
            chl = chl_data.sel(time=date).CHL
            temp = temp_data.sel(time=date).analysed_sst    

        lons = temp.lon.to_numpy()
        lats = temp.lat.to_numpy()
        temp = temp.to_numpy()
        chl = chl.interp(lon = lons, lat = lats, method = "nearest").to_numpy()
        bath = bath.elevation.interp(lon = lons, lat = lats, method = "nearest").to_numpy()
        bath[np.isnan(bath)] = 0

        if len(date)==1:
            temp = temp[0]
            chl = chl[0]

    return lons, lats, chl, temp, bath

def bathymetry_filter(a, bath, h = -1000):
    """ Removes coastal data (depth > -1000 m)"""
    if len(a.shape) == 2:
        a_filt = np.copy(a)
        a_filt[bath >= h] = np.nan
    else: 
        a_filt = np.copy(a)
        a_filt[:, bath >= h] = np.nan
    return a_filt

def anomalies(a):
    return (a - np.nanmean(a))/np.nanstd(a)

def cluster_data(chl, temp):
    data = np.array(list(zip(chl[~np.isnan(chl)], temp[~np.isnan(chl)])))
    kmeans = KMeans(n_clusters = 2, init = "random")
    labels = kmeans.fit_predict(data)
    labels_matrix = np.empty(np.shape(chl))
    labels_matrix[:,:] = np.nan
    mask = ~np.isnan(chl)
    labels_matrix[mask] = labels.astype(int)
    return labels_matrix

