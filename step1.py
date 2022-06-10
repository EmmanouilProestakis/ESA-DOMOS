# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 17:29:13 2021
This script uses as input the ERA5 u, v, w, lat, lon, and geopotential and performs two tasks:
    (a) converts geopotential to geometric height a.m.s.l.
    (b) regrids u, v, w wind components into a regular 1x1 deg2 grid.
Here the script is applied for DOMOS domain.
@author: proestakis
"""

import netCDF4  as nc
import numpy    as np
import metpy.calc
from   metpy.units import units
from   datetime    import datetime, timedelta
from   os          import walk
import math

# Initializing timer:
startTime = datetime.now()

# setting up input and output path of ERA5 files.
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels-monthly-means?tab=overview
# last access: 03/01/2022.
mypath      = r"C:\Users\proes\Desktop\PC-backup\Projects\DOMOS\Datasets\Input\ERA\temp"
output_path = r"C:\Users\proes\Desktop\PC-backup\Projects\DOMOS\Datasets\Input\ERA\processed"

# checking if input files exist in input dir.
filenames   = next(walk(mypath), (None, None, []))[2]  # [] if no file

# ESA-DOMOS: "... the full coverage of the Atlantic Ocean (including dust emission sources of Africa and S. America, 
# the broader Atlantic Ocean, Caribbean Sea and Gulf of Mexico, confined between latitudes 40°N to 60°S), and of 
# temporal coverage at least between 2010 and 2020".
# Therefore: 
# (I)  DOMOS lon: -100E:20E
# (II) DOMOS lat:  -60N:40N
# (a wider domain is used here to account for (1) all fluxes and (2) the broader domain, and N.Atlandic Dust)
lon_array = np.arange(-100.5,31.5)
lat_array = np.arange(-60.5,71.5)

# function for weighted average and weighted std, for computation of the ERA surface weighted u, v, w, z
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))

for filename in filenames:
    
    file                 = mypath + '\\' + filename
    dataset              = nc.Dataset(file) 
    
    # ERA spatial reference:
    # https://confluence.ecmwf.int/display/CKB/ERA5%3A+What+is+the+spatial+reference
    # last visit: 03/02/2022.
    # ERA longitude from 0->360 deg to -180->180 deg.    
    dataset_latitude     = dataset['latitude'][:]
    idx_lat              = np.where((dataset_latitude >= lat_array[0]-1) & (dataset_latitude <= lat_array[-1]+1))
    idx_lat              = np.ravel(idx_lat)
    dataset_longitude    = dataset['longitude'][:] 
    for lon in dataset_longitude:
        if lon >= 180:
            dataset_longitude[np.where(lon == dataset_longitude)] = lon - 360
    idx_lon              = np.where((dataset_longitude >= lon_array[0]-1) & (dataset_longitude <= lon_array[-1]+1))
    idx_lon              = np.ravel(idx_lon)
    dataset_latitude     = dataset['latitude'][idx_lat]
    dataset_longitude    = dataset['longitude'][idx_lon] 
    for lon in dataset_longitude:
        if lon >= 180:
            dataset_longitude[np.where(lon == dataset_longitude)] = lon - 360    
    # example: 
    # for lon_array: [98.5 99.5] 
    #     dataset_longitude: [ 97.5 97.75 98. 98.25 98.5 98.75 99. 99.25 99.5 99.75 100. 100.25 100.5 ]
    # for lat_array: [28.5 29.5]
    #     dataset_latitide: [ 30.5 30.25 30. 29.75 29.5 29.25 29. 28.75 28.5 28.25 28. 27.75 27.5 ]
            
    dataset_time         = dataset['time'][:]
    dataset_level        = dataset['level'][:]
    dataset_u            = dataset['u'][:,:,idx_lat,idx_lon]
    dataset_v            = dataset['v'][:,:,idx_lat,idx_lon]
    dataset_w            = dataset['w'][:,:,idx_lat,idx_lon]    
    dataset_geopotential = dataset['z']
    
    # ERA convert Geopotenial to geometric height (a.m.s.l.):
    # https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.geopotential_to_height.html
    # last access: 03/02/2022. 
    dataset_geo_units    = dataset_geopotential.units
    dataset_geopotential = dataset['z'][:,:,idx_lat,idx_lon]
    dataset_geopotential = units.Quantity(dataset_geopotential,dataset_geo_units)  
    dataset_height       = metpy.calc.geopotential_to_height(dataset_geopotential)
    dataset_height       = np.array(dataset_height)
    
    # ERA time:
    # units     = hours since 1900-01-01 00:00:00.0 
    # long_name = time 
    # calendar  = gregorian     
    # loop to produce montly-mean files
    for time in dataset_time:
        
        idx_time = np.where(time == dataset_time)
        idx_time = np.ravel(idx_time)      
                
        # keeping time info "yyyy/mm" for output filename.
        date_1          = '1/1/1900 00:00:00.0'
        date_format_str = '%d/%m/%Y %H:%M:%S.%f'
        start           = datetime.strptime(date_1, date_format_str)
        temp_time       = start + timedelta(hours = int(time))
        temp_time       = temp_time.strftime("%Y%m")
        
        # initializing: u-mean,SD / v-mean,SD / w-mean,SD / z-mean for 1x1 deg2 grid resolution.
        u_total_mean  = np.empty((len(lon_array),len(lat_array),len(dataset_level)))
        u_total_SD    = np.empty((len(lon_array),len(lat_array),len(dataset_level)))
        v_total_mean  = np.empty((len(lon_array),len(lat_array),len(dataset_level)))
        v_total_SD    = np.empty((len(lon_array),len(lat_array),len(dataset_level)))
        w_total_mean  = np.empty((len(lon_array),len(lat_array),len(dataset_level)))
        w_total_SD    = np.empty((len(lon_array),len(lat_array),len(dataset_level)))        
        z_total_mean  = np.empty((len(lon_array),len(lat_array),len(dataset_level)))
        
        counter_lon_lat = 0.0
        
        # computing and saving: u-mean,SD / v-mean,SD / w-mean,SD / z-mean for 1x1 deg2 grid resolution.
        for lon in lon_array:
            idx_lon = np.where((lon-0.5 <= dataset_longitude) & (dataset_longitude <= lon+0.5))
            idx_lon = np.ravel(idx_lon)
            temp_u = dataset_u[:,:,:,idx_lon]
            temp_v = dataset_v[:,:,:,idx_lon]
            temp_w = dataset_w[:,:,:,idx_lon]
            temp_height = dataset_height[:,:,:,idx_lon]
            
            for lat in lat_array:
                idx_lat = np.where((lat-0.5 <= dataset_latitude) & (dataset_latitude <= lat+0.5))
                idx_lat = np.ravel(idx_lat)
                u       = temp_u[idx_time,:,idx_lat,:]
                v       = temp_v[idx_time,:,idx_lat,:]
                w       = temp_w[idx_time,:,idx_lat,:]
                z       = temp_height[idx_time,:,idx_lat,:]
                
                for lev in dataset_level:
                    idx_level = np.where(lev == dataset_level)
                    idx_level = np.ravel(idx_level)
                    
                    u_temp = u[:,idx_level,:]
                    v_temp = v[:,idx_level,:]
                    w_temp = w[:,idx_level,:]
                    z_temp = z[:,idx_level,:]

                    # remove level dimension from u_temp, v_temp, w_temp, z_temp
                    u_temp = np.squeeze(u_temp)
                    v_temp = np.squeeze(v_temp)
                    w_temp = np.squeeze(w_temp)
                    z_temp = np.squeeze(z_temp)
                    
                    # hardcoded..
                    # According to ERA documentation, each 1x1 deg2 grid is composed by 25 points 
                    #  4 points at corners -> weight 0.25
                    # 12 points at rectangle sides -> weight 0.5
                    #  9 points internally -> weight 1.0
                    # thus:
                    # ERA weight to account for the surface were the u, v, w, z are representative within a 1x1 deg2 grid resolution.
                    ERA_weights = [[1, 2, 2, 2, 1], [2, 4, 4, 4, 2], [2, 4, 4, 4, 2], [2, 4, 4, 4, 2], [1, 2, 2, 2, 1] ] 
                                        
                    u_total_mean[np.where(lon == lon_array),np.where(lat == lat_array),idx_level] = weighted_avg_and_std(u_temp, ERA_weights)[0]
                    u_total_SD[np.where(lon == lon_array),np.where(lat == lat_array),idx_level]   = weighted_avg_and_std(u_temp, ERA_weights)[1]
                    v_total_mean[np.where(lon == lon_array),np.where(lat == lat_array),idx_level] = weighted_avg_and_std(v_temp, ERA_weights)[0]
                    v_total_SD[np.where(lon == lon_array),np.where(lat == lat_array),idx_level]   = weighted_avg_and_std(v_temp, ERA_weights)[1]
                    w_total_mean[np.where(lon == lon_array),np.where(lat == lat_array),idx_level] = weighted_avg_and_std(w_temp, ERA_weights)[0]
                    w_total_SD[np.where(lon == lon_array),np.where(lat == lat_array),idx_level]   = weighted_avg_and_std(w_temp, ERA_weights)[1] 
                    z_total_mean[np.where(lon == lon_array),np.where(lat == lat_array),idx_level] = weighted_avg_and_std(z_temp, ERA_weights)[0]
                    
        ######################################################################            
        #  --- Saving ERA u, v, w, height monthly mean dataset as NetCDF --- #
        ######################################################################

        # creating nc. filename and initiallizing:                 
        fn           = output_path + '\\' + temp_time + '-ERA_uvw-MONTHLY.nc'
        ds           = nc.Dataset(fn, 'w', format='NETCDF4')
        
        # create nc. dimensions:
        longitude = lon_array
        latitude  = lat_array        
        lev          = ds.createDimension('lev',  len(dataset_level))
        lat          = ds.createDimension('lat',  len(latitude)) 
        lon          = ds.createDimension('lon',  len(longitude))
      
        # create nc. variables:
        lats         = ds.createVariable('Latitude', 'f4', ('lat',))
        lons         = ds.createVariable('Longitude','f4', ('lon',))
        Height       = ds.createVariable('Height',   'f4', ('lon','lat','lev',))
        U            = ds.createVariable('U',    np.float64, ('lon','lat','lev',))
        U_SD         = ds.createVariable('U_SD', np.float64, ('lon','lat','lev',))
        V            = ds.createVariable('V',    np.float64, ('lon','lat','lev',))
        V_SD         = ds.createVariable('V_SD', np.float64, ('lon','lat','lev',))
        W            = ds.createVariable('W',    np.float64, ('lon','lat','lev',))
        W_SD         = ds.createVariable('W_SD', np.float64, ('lon','lat','lev',))        
        
        # nc. variables' units
        lats.units   = 'degrees_north'
        lons.units   = 'degrees_east'
        Height.units = 'm'     
        U.units      = 'm s**-1'
        U_SD.units   = 'm s**-1'
        V.units      = 'm s**-1'
        V_SD.units   = 'm s**-1'
        W.units      = 'Pa s**-1'
        W_SD.units   = 'Pa s**-1'   
        
        # nc. variables' "long names":
        lats.long_name   = 'Latitude'
        lons.long_name   = 'Longitude'
        Height.long_name = 'Height'
        U.long_name      = 'U component of wind'
        U_SD.long_name   = 'U component of wind SD'
        V.long_name      = 'V component of wind'
        V_SD.long_name   = 'V component of wind SD'
        W.long_name      = 'Vertical velocity'
        W_SD.long_name   = 'Vertical velocity SD'

        # nc. variables' "standard names":
        Height.standard_name = 'height'
        U.standard_name      = 'eastward_wind'
        U_SD.standard_name   = 'eastward_wind_SD'
        V.standard_name      = 'northward_wind'
        V_SD.standard_name   = 'northward_wind_SD'
        W.standard_name      = 'lagrangian_tendency_of_air_pressure'
        W_SD.standard_name   = 'lagrangian_tendency_of_air_pressure_SD'         

        # nc. saving datasets           
        lats[:]      = latitude
        lons[:]      = longitude
        Height[:]    = z_total_mean
        U[:]         = u_total_mean
        U_SD[:]      = u_total_SD        
        V[:]         = v_total_mean
        V_SD[:]      = v_total_SD
        W[:]         = w_total_mean 
        W_SD[:]      = w_total_SD

        ds.close()                
        
        print("End of: " + temp_time + '-ERA_uvw-MONTHLY.nc')
        
# How much time it took to execute the script in minutes:
print(datetime.now() - startTime)                               