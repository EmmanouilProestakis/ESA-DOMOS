# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 08:54:57 2022
Establishes the required EO-and-ERA5 DOMOS dataset in the same L3 1x1 grid resolution and monthly-mean.
@author: proes
"""

import netCDF4  as nc
import numpy    as np
import os
from   os.path import exists
from   datetime    import datetime, timedelta
from   os          import walk
import glob
import math
import warnings
import metpy
import metpy.calc
from   metpy.units import units
from   pathlib import Path
import numpy.ma as ma

# directory paths of input datasets
ERA_path               = r"S:\DOMOS\ERA5\processed"
ERA_path               = r"S:\DOMOS\ERA5\test"
MetopA_IASI_ENS_path   = r"S:\DOMOS\WP2100-EO_satellite_observations-products_and_LIVAS\Metop-IASI\METOPA\L3_MM\ENS" 
MetopA_IASI_IMARS_path = r"S:\DOMOS\WP2100-EO_satellite_observations-products_and_LIVAS\Metop-IASI\METOPA\L3_MM\IMARS"
MetopA_IASI_LMD_path   = r"S:\DOMOS\WP2100-EO_satellite_observations-products_and_LIVAS\Metop-IASI\METOPA\L3_MM\LMD"
MetopA_IASI_MARIP_path = r"S:\DOMOS\WP2100-EO_satellite_observations-products_and_LIVAS\Metop-IASI\METOPA\L3_MM\MARIP"
MetopA_IASI_ULB_path   = r"S:\DOMOS\WP2100-EO_satellite_observations-products_and_LIVAS\Metop-IASI\METOPA\L3_MM\ULB"
MetopC_IASI_ENS_path   = r"S:\DOMOS\WP2100-EO_satellite_observations-products_and_LIVAS\Metop-IASI\METOPC\L3_MM\ENS" 
MetopC_IASI_IMARS_path = r"S:\DOMOS\WP2100-EO_satellite_observations-products_and_LIVAS\Metop-IASI\METOPC\L3_MM\IMARS"
MetopC_IASI_LMD_path   = r"S:\DOMOS\WP2100-EO_satellite_observations-products_and_LIVAS\Metop-IASI\METOPC\L3_MM\LMD"
MetopC_IASI_MARIP_path = r"S:\DOMOS\WP2100-EO_satellite_observations-products_and_LIVAS\Metop-IASI\METOPC\L3_MM\MARIP"
MetopC_IASI_ULB_path   = r"S:\DOMOS\WP2100-EO_satellite_observations-products_and_LIVAS\Metop-IASI\METOPC\L3_MM\ULB"
MIDAS_path             = r"S:\DOMOS\WP2100-EO_satellite_observations-products_and_LIVAS\MONTHLY-MIDAS-FILES\MODIS-AQUA"
Land_Mask_path         = r"S:\DOMOS\Land_Ocean_Map"
LIVAS_path             = r"T:\LIVAS\2022-01_grid"

# directory path of output datasets
output_path           = r"S:\DOMOS\Step_II"

##########################################################################################
####                     functions to locate files                                    ####
##########################################################################################

def find_IASI_file(path, sufix):
    IASI_file = path + "\\" + sufix + "*.nc" 
    IASI_file = glob.glob(IASI_file)
    if len(IASI_file) == 0:
        Metop_IASI_file = 'non_existing_file'
    else:       
        Metop_IASI_file = path + "\\" + IASI_file[0].split("\\")[-1]
    return(Metop_IASI_file)

def find_MIDAS_file(path, sufix):
    MIDAS_file = path + "\\*" + sufix + ".nc"    
    MIDAS_file = glob.glob(MIDAS_file)
    if len(MIDAS_file) == 0: 
        MIDAS_file = 'non_existing_file'
    else:       
        MIDAS_file = MIDAS_file[0].split("\\")[-1]
        MIDAS_file = path + "\\" + MIDAS_file
    return(MIDAS_file)     

def CALIPSO_Aerosol_Subtypes( AVD_Aerosol_Subtype, L2_CF_profiles ):
    dim_I  = AVD_Aerosol_Subtype.shape[0]
    dim_II = AVD_Aerosol_Subtype.shape[1] 
    Marine                     = np.array(np.empty((dim_I,dim_II))) 
    Dust                       = np.array(np.empty((dim_I,dim_II))) 
    Polluted_Continental_Smoke = np.array(np.empty((dim_I,dim_II))) 
    Clean_Continental          = np.array(np.empty((dim_I,dim_II))) 
    Polluted_Dust              = np.array(np.empty((dim_I,dim_II))) 
    Elevated_Smoke             = np.array(np.empty((dim_I,dim_II))) 
    Dusty_Marine               = np.array(np.empty((dim_I,dim_II))) 
    Marine[:]                     = 0
    Dust[:]                       = 0
    Polluted_Continental_Smoke[:] = 0
    Clean_Continental[:]          = 0
    Polluted_Dust[:]              = 0
    Elevated_Smoke[:]             = 0
    Dusty_Marine[:]               = 0
    if L2_CF_profiles > 0:
        idx    = np.where(AVD_Aerosol_Subtype == 1)
        Marine[idx] = 1 
        Marine = np.sum(Marine, axis=0)
        idx    = np.where(AVD_Aerosol_Subtype == 2)
        Dust[idx] = 1 
        Dust = np.sum(Dust, axis=0)                     
        idx    = np.where(AVD_Aerosol_Subtype == 3)
        Polluted_Continental_Smoke[idx] = 1 
        Polluted_Continental_Smoke = np.sum(Polluted_Continental_Smoke, axis=0)      
        idx    = np.where(AVD_Aerosol_Subtype == 4)
        Clean_Continental[idx] = 1 
        Clean_Continental = np.sum(Clean_Continental, axis=0)                          
        idx    = np.where(AVD_Aerosol_Subtype == 5)
        Polluted_Dust[idx] = 1 
        Polluted_Dust = np.sum(Polluted_Dust, axis=0)      
        idx    = np.where(AVD_Aerosol_Subtype == 6)
        Elevated_Smoke[idx] = 1 
        Elevated_Smoke = np.sum(Elevated_Smoke, axis=0)                          
        idx    = np.where(AVD_Aerosol_Subtype == 7)
        Dusty_Marine[idx] = 1 
        Dusty_Marine = np.sum(Dusty_Marine, axis=0)  
    return( Marine, Dust, Polluted_Continental_Smoke, Clean_Continental, Polluted_Dust, Elevated_Smoke, Dusty_Marine )
    
##########################################################################################
####                                main                                              ####
##########################################################################################

# Initializing timer:
startTime = datetime.now()

# checking if input files exist in input dir.
ERA_filenames = next(walk(ERA_path), (None, None, []))[2]  # [] if no file

for ERA_file_count,ERA_filename in enumerate(ERA_filenames):
    
    ######################
    ###       ERA5     ###
    ######################    
    
    # ERA5 file opening and reading of ERA5 varibales.
    ERA_file      = ERA_path + '\\' + ERA_filename
    
    # reading variables of interest.    

    ERA_dataset   = nc.Dataset(ERA_file) 
    
    ERA_Latitude  = ERA_dataset['Latitude'][:]
    ERA_Longitude = ERA_dataset['Longitude'][:]    
    idx           = np.ravel(np.where((ERA_Latitude >= -60.5) & (ERA_Latitude <= 41.5)))   
    ERA_Height    = ERA_dataset['Height'][:,idx,:]
    ERA_Latitude  = ERA_dataset['Latitude'][idx]
    ERA_Longitude = ERA_dataset['Longitude'][:]
    ERA_U         = ERA_dataset['U'][:,idx,:]
    ERA_U_SD      = ERA_dataset['U_SD'][:,idx,:]
    ERA_V         = ERA_dataset['V'][:,idx,:]
    ERA_V_SD      = ERA_dataset['V_SD'][:,idx,:]
    ERA_W         = ERA_dataset['W'][:,idx,:]
    ERA_W_SD      = ERA_dataset['W_SD'][:,idx,:]
    
    # extracting "yyyymm" sufix from ERA5 filename, for finding the satellite-based MM files.  
    ERA_substrings = ERA_file.split("\\")
    ERA_substrings = ERA_substrings[len(ERA_substrings)-1]
    ERA_substrings = ERA_substrings.split("-")
    file_sufix     = ERA_substrings[0]
    ERA_month      = int(file_sufix[4:6])
    ERA_year       = int(file_sufix[0:4])               
                
    #######################
    ####### LandMask ######
    #######################

    # Land_Mask file opening and reading of varibales.
    Land_Mask_filename  = 'Land_Ocean_Mask_1x1.nc'
    Land_Mask_file      = Land_Mask_path + '\\' + Land_Mask_filename
    
    # reading variables of interest.    
    Land_Mask_dataset   = nc.Dataset(Land_Mask_file)    
    Land_Mask_Latitude  = Land_Mask_dataset['Latitude'][:]
    Land_Mask_Longitude = Land_Mask_dataset['Longitude'][:]
    Land_Ocean_Mask     = Land_Mask_dataset['Land_Ocean_Mask'][:]

    idx_lat         = np.where((Land_Mask_Latitude >= min(ERA_Latitude))   & (Land_Mask_Latitude <= max(ERA_Latitude) ))
    idx_lon         = np.where((Land_Mask_Longitude >= min(ERA_Longitude)) & (Land_Mask_Longitude <= max(ERA_Longitude)))   
    idx_lat         = np.squeeze(np.asarray(idx_lat))
    idx_lon         = np.squeeze(np.asarray(idx_lon))  
    Land_Mask_Latitude  = Land_Mask_dataset['Latitude'][idx_lat]
    Land_Mask_Longitude = Land_Mask_dataset['Longitude'][idx_lon]
    Land_Ocean_Mask = np.fliplr(Land_Mask_dataset['Land_Ocean_Mask'][idx_lon,idx_lat])

    ########################
    ### Aqua MODIS MIDAS ###
    ########################    

    # finding Aqua MODIS MIDAS file.
    MIDAS_file      = find_MIDAS_file(MIDAS_path, file_sufix)

    if (MIDAS_file == 'non_existing_file'):
    # reading variables of interest.    
        empty_matrix    = np.empty((len(ERA_Longitude),len(ERA_Latitude)))
        empty_matrix[:] = np.nan
        MIDAS_DOD_MEAN  = empty_matrix
        MIDAS_DOD_STD   = empty_matrix  
        MIDAS_DOD_MEAN[:] = np.nan
        MIDAS_DOD_STD[:]  = np.nan
    else:
        # reading variables of interest.    
        MIDAS_dataset   = nc.Dataset(MIDAS_file)
        MIDAS_Latitude  = MIDAS_dataset['Latitude'][:]
        MIDAS_Longitude = MIDAS_dataset['Longitude'][:]
        MIDAS_DOD_MEAN  = MIDAS_dataset['MIDAS-DOD-MEAN'][:]
        MIDAS_DOD_STD   = MIDAS_dataset['MIDAS-DOD-STD'][:]
        Lon             = MIDAS_Longitude[1,:]
        Lat             = MIDAS_Latitude[:,1]

        idx_lat         = np.where((Lat >= min(ERA_Latitude))  & (Lat <= max(ERA_Latitude) ))
        idx_lon         = np.where((Lon >= min(ERA_Longitude)) & (Lon <= max(ERA_Longitude)))   
        idx_lat         = np.squeeze(np.asarray(idx_lat))
        idx_lon         = np.squeeze(np.asarray(idx_lon))  
        MIDAS_DOD_MEAN  = MIDAS_dataset['MIDAS-DOD-MEAN'][idx_lat,idx_lon]
        MIDAS_DOD_STD   = MIDAS_dataset['MIDAS-DOD-STD'][idx_lat,idx_lon]
    
        MIDAS_DOD_MEAN  = np.transpose(MIDAS_DOD_MEAN)
        MIDAS_DOD_STD   = np.transpose(MIDAS_DOD_STD)
        MIDAS_DOD_MEAN[MIDAS_DOD_MEAN.mask == True ] = np.nan
        MIDAS_DOD_STD[MIDAS_DOD_STD.mask == True ]   = np.nan

    ########################
    #### MetopA IASI ENS ###
    ########################
    
    # finding Metop IASI ENS file.
    Metop_IASI_ENS_file    =  find_IASI_file(MetopA_IASI_ENS_path, file_sufix)   
    
    if (Metop_IASI_ENS_file == 'non_existing_file'):
    # reading variables of interest.    
        empty_matrix              = np.empty((len(ERA_Longitude),len(ERA_Latitude)))
        empty_matrix[:]           = np.nan
        MetopA_ENS_DAOD550        = empty_matrix
        MetopA_ENS_DAOD550_unc    = empty_matrix
        MetopA_ENS_DAOD550[:]     = np.nan
        MetopA_ENS_DAOD550_unc[:] = np.nan
    else:
        Metop_IASI_ENS_dataset = nc.Dataset(Metop_IASI_ENS_file)
        ENS_Latitude           = Metop_IASI_ENS_dataset['latitude'][:]
        ENS_Longitude          = Metop_IASI_ENS_dataset['longitude'][:]    
        idx_lat                = np.where((ENS_Latitude  >= min(ERA_Latitude))  & (ENS_Latitude  <= max(ERA_Latitude) ))
        idx_lon                = np.where((ENS_Longitude >= min(ERA_Longitude)) & (ENS_Longitude <= max(ERA_Longitude)))   
        idx_lat                = np.squeeze(np.asarray(idx_lat))
        idx_lon                = np.squeeze(np.asarray(idx_lon))
        ENS_DAOD550            = Metop_IASI_ENS_dataset['DAOD550'][idx_lat,idx_lon]
        ENS_DAOD550_unc        = Metop_IASI_ENS_dataset['DAOD550_UNCERTAINTY_ENSEMBLE'][idx_lat,idx_lon]
        MetopA_ENS_DAOD550     = np.transpose(ENS_DAOD550)
        MetopA_ENS_DAOD550_unc = np.transpose(ENS_DAOD550_unc)
        if ma.isMaskedArray(MetopA_ENS_DAOD550) == True:        
            MetopA_ENS_DAOD550[MetopA_ENS_DAOD550.mask == True ] = np.nan
        if ma.isMaskedArray(MetopA_ENS_DAOD550_unc) == True:
            MetopA_ENS_DAOD550_unc[MetopA_ENS_DAOD550_unc.mask == True ] = np.nan

    #########################
    ### MetopA IASI IMARS ###
    #########################
    
    # finding Metop IASI IMARS file.
    Metop_IASI_IMARS_file    =  find_IASI_file(MetopA_IASI_IMARS_path, file_sufix)

    if (Metop_IASI_IMARS_file == 'non_existing_file'):
    # reading variables of interest.    
        empty_matrix             = np.empty((len(ERA_Longitude),len(ERA_Latitude)))
        empty_matrix[:]          = np.nan
        MetopA_IMARS_DAOD550     = empty_matrix
        MetopA_IMARS_DAOD550_unc = empty_matrix 
        MetopA_IMARS_DAOD550[:]     = np.nan
        MetopA_IMARS_DAOD550_unc[:] = np.nan        
    else:
    # reading variables of interest.    
        Metop_IASI_IMARS_dataset = nc.Dataset(Metop_IASI_IMARS_file)
        IMARS_Latitude           = Metop_IASI_IMARS_dataset['latitude'][:]
        IMARS_Longitude          = Metop_IASI_IMARS_dataset['longitude'][:]
        idx_lat                  = np.where((IMARS_Latitude  >= min(ERA_Latitude))  & (IMARS_Latitude  <= max(ERA_Latitude) ))
        idx_lon                  = np.where((IMARS_Longitude >= min(ERA_Longitude)) & (IMARS_Longitude <= max(ERA_Longitude)))   
        idx_lat                  = np.squeeze(np.asarray(idx_lat))
        idx_lon                  = np.squeeze(np.asarray(idx_lon))    
        MetopA_IMARS_DAOD550     = Metop_IASI_IMARS_dataset['D_AOD550'][idx_lon,idx_lat]
        MetopA_IMARS_DAOD550_unc = Metop_IASI_IMARS_dataset['D_AOD550_uncertainty'][idx_lon,idx_lat]   
        if ma.isMaskedArray(MetopA_IMARS_DAOD550) == True:        
            MetopA_IMARS_DAOD550[MetopA_IMARS_DAOD550.mask == True ] = np.nan
        if ma.isMaskedArray(MetopA_IMARS_DAOD550_unc) == True:
            MetopA_IMARS_DAOD550_unc[MetopA_IMARS_DAOD550_unc.mask == True ] = np.nan
        
    #######################
    ### MetopA IASI ULB ###
    #######################

    # finding Metop IASI ULB file.
    Metop_IASI_ULB_file    =  find_IASI_file(MetopA_IASI_ULB_path, file_sufix)
    
    if (Metop_IASI_ULB_file == 'non_existing_file'):
    # reading variables of interest.    
        empty_matrix          = np.empty((len(ERA_Longitude),len(ERA_Latitude)))
        empty_matrix[:]       = np.nan
        MetopA_ULB_DAOD550    = empty_matrix  
        MetopA_ULB_DAOD550[:] = np.nan
    else:
    # reading variables of interest.
       Metop_IASI_ULB_dataset = nc.Dataset(Metop_IASI_ULB_file)
       ULB_Latitude           = Metop_IASI_ULB_dataset['latitude'][:]
       ULB_Longitude          = Metop_IASI_ULB_dataset['longitude'][:]       
       idx_lat                = np.where((ULB_Latitude  >= min(ERA_Latitude))  & (ULB_Latitude  <= max(ERA_Latitude) ))
       idx_lon                = np.where((ULB_Longitude >= min(ERA_Longitude)) & (ULB_Longitude <= max(ERA_Longitude)))   
       idx_lat                = np.squeeze(np.asarray(idx_lat))
       idx_lon                = np.squeeze(np.asarray(idx_lon))  
       ULB_DAOD550            = Metop_IASI_ULB_dataset['D_AOD550_mean'][idx_lat,idx_lon]
       MetopA_ULB_DAOD550     = np.transpose(ULB_DAOD550)
       if ma.isMaskedArray(MetopA_ULB_DAOD550) == True:
           MetopA_ULB_DAOD550[MetopA_ULB_DAOD550.mask == True ] = np.nan

    #######################
    ### MetopA IASI LMD ###
    #######################

    # finding Metop IASI LMD file.
    Metop_IASI_LMD_file    =  find_IASI_file(MetopA_IASI_LMD_path, file_sufix)

    if (Metop_IASI_LMD_file == 'non_existing_file'):
    # reading variables of interest.    
        empty_matrix         = np.empty((len(ERA_Longitude),len(ERA_Latitude)))
        empty_matrix[:]      = np.nan
        MetopA_LMD_DAOD550   = empty_matrix  
        MetopA_LMD_DAOD550[:] = np.nan
    else:
    # reading variables of interest.    
        Metop_IASI_LMD_dataset = nc.Dataset(Metop_IASI_LMD_file)
        LMD_Latitude           = Metop_IASI_LMD_dataset['Latitude'][:]  + 0.5
        LMD_Longitude          = Metop_IASI_LMD_dataset['Longitude'][:] + 0.5
        idx_lat                = np.where((LMD_Latitude  >= min(ERA_Latitude))  & (LMD_Latitude  <= max(ERA_Latitude) ))
        idx_lon                = np.where((LMD_Longitude >= min(ERA_Longitude)) & (LMD_Longitude <= max(ERA_Longitude)))   
        idx_lat                = np.squeeze(np.asarray(idx_lat))
        idx_lon                = np.squeeze(np.asarray(idx_lon))      
        LMD_DAOD550            = Metop_IASI_LMD_dataset['Daod550'][idx_lat,idx_lon]
        MetopA_LMD_DAOD550     = np.transpose(LMD_DAOD550)
        if ma.isMaskedArray(MetopA_LMD_DAOD550) == True:             
            MetopA_LMD_DAOD550[MetopA_LMD_DAOD550.mask == True ] = np.nan
       
    #########################
    ### MetopA IASI MARIP ###
    #########################    

    # finding Metop IASI MARIP file.
    Metop_IASI_MARIP_file    =  find_IASI_file(MetopA_IASI_MARIP_path, file_sufix)

    if (Metop_IASI_MARIP_file == 'non_existing_file'):
    # reading variables of interest.    
        empty_matrix         = np.empty((len(ERA_Longitude),len(ERA_Latitude)))
        empty_matrix[:]      = np.nan
        MetopA_MARIP_DAOD550 = empty_matrix  
        MetopA_MARIP_DAOD550[:] = np.nan
    else:
    # reading variables of interest.    
        Metop_IASI_MARIP_dataset = nc.Dataset(Metop_IASI_MARIP_file)
        MARIP_DAOD550            = Metop_IASI_MARIP_dataset['D_AOD550_mean'][:]
        MARIP_Latitude           = Metop_IASI_MARIP_dataset['latitude'][:]
        MARIP_Longitude          = Metop_IASI_MARIP_dataset['longitude'][:]
        
        DOD_temp    = np.empty((len(ERA_Longitude),len(ERA_Latitude)))
        DOD_temp[:] = np.nan
        for count_lon,lon in enumerate(ERA_Longitude):
            for count_lat,lat in enumerate(ERA_Latitude):
                idx_lat = np.where(lat == MARIP_Latitude)
                idx_lon = np.where(lon == MARIP_Longitude)
                idx_lat = np.squeeze(np.asarray(idx_lat))
                idx_lon = np.squeeze(np.asarray(idx_lon))             
                if (idx_lon.size == 0) or (idx_lat.size == 0):
                    DOD_temp[count_lon,count_lat] = np.nan
                else:
                    DOD_temp[count_lon,count_lat] = MARIP_DAOD550[idx_lat,idx_lon]
        MetopA_MARIP_DAOD550 = DOD_temp
        
    ########################
    #### MetopC IASI ENS ###
    ########################
    
    # finding Metop IASI ENS file.
    Metop_IASI_ENS_file    =  find_IASI_file(MetopC_IASI_ENS_path, file_sufix)   
    
    if (Metop_IASI_ENS_file == 'non_existing_file'):
    # reading variables of interest.    
        empty_matrix           = np.empty((len(ERA_Longitude),len(ERA_Latitude)))
        empty_matrix[:]        = np.nan
        MetopC_ENS_DAOD550     = empty_matrix
        MetopC_ENS_DAOD550_unc = empty_matrix     
        MetopC_ENS_DAOD550[:]     = np.nan
        MetopC_ENS_DAOD550_unc[:] = np.nan        
    else:
        Metop_IASI_ENS_dataset = nc.Dataset(Metop_IASI_ENS_file)
        ENS_Latitude           = Metop_IASI_ENS_dataset['latitude'][:]
        ENS_Longitude          = Metop_IASI_ENS_dataset['longitude'][:]    
        idx_lat                = np.where((ENS_Latitude  >= min(ERA_Latitude))  & (ENS_Latitude  <= max(ERA_Latitude) ))
        idx_lon                = np.where((ENS_Longitude >= min(ERA_Longitude)) & (ENS_Longitude <= max(ERA_Longitude)))   
        idx_lat                = np.squeeze(np.asarray(idx_lat))
        idx_lon                = np.squeeze(np.asarray(idx_lon))
        ENS_DAOD550            = Metop_IASI_ENS_dataset['DAOD550'][idx_lat,idx_lon]
        ENS_DAOD550_unc        = Metop_IASI_ENS_dataset['DAOD550_UNCERTAINTY_ENSEMBLE'][idx_lat,idx_lon]
        MetopC_ENS_DAOD550     = np.transpose(ENS_DAOD550)
        MetopC_ENS_DAOD550_unc = np.transpose(ENS_DAOD550_unc)   
        if ma.isMaskedArray(MetopC_ENS_DAOD550) == True:        
            MetopC_ENS_DAOD550[MetopC_ENS_DAOD550.mask == True ] = np.nan
        if ma.isMaskedArray(MetopC_ENS_DAOD550_unc) == True:
            MetopC_ENS_DAOD550_unc[MetopC_ENS_DAOD550_unc.mask == True ] = np.nan      

    #########################
    ### MetopC IASI IMARS ###
    #########################
    
    # finding Metop IASI IMARS file.
    Metop_IASI_IMARS_file    =  find_IASI_file(MetopC_IASI_IMARS_path, file_sufix)

    if (Metop_IASI_IMARS_file == 'non_existing_file'):
    # reading variables of interest.    
        empty_matrix             = np.empty((len(ERA_Longitude),len(ERA_Latitude)))
        empty_matrix[:]          = np.nan
        MetopC_IMARS_DAOD550     =  empty_matrix
        MetopC_IMARS_DAOD550_unc =  empty_matrix  
        MetopC_IMARS_DAOD550[:]     = np.nan
        MetopC_IMARS_DAOD550_unc[:] = np.nan           
    else:
    # reading variables of interest.    
        Metop_IASI_IMARS_dataset = nc.Dataset(Metop_IASI_IMARS_file)
        IMARS_Latitude           = Metop_IASI_IMARS_dataset['latitude'][:]
        IMARS_Longitude          = Metop_IASI_IMARS_dataset['longitude'][:]
        idx_lat                  = np.where((IMARS_Latitude  >= min(ERA_Latitude))  & (IMARS_Latitude  <= max(ERA_Latitude) ))
        idx_lon                  = np.where((IMARS_Longitude >= min(ERA_Longitude)) & (IMARS_Longitude <= max(ERA_Longitude)))   
        idx_lat                  = np.squeeze(np.asarray(idx_lat))
        idx_lon                  = np.squeeze(np.asarray(idx_lon))    
        MetopC_IMARS_DAOD550     = Metop_IASI_IMARS_dataset['D_AOD550'][idx_lon,idx_lat]
        MetopC_IMARS_DAOD550_unc = Metop_IASI_IMARS_dataset['D_AOD550_uncertainty'][idx_lon,idx_lat] 
        if ma.isMaskedArray(MetopC_IMARS_DAOD550) == True:        
            MetopC_IMARS_DAOD550[MetopC_IMARS_DAOD550.mask == True ] = np.nan
        if ma.isMaskedArray(MetopC_IMARS_DAOD550_unc) == True:
            MetopC_IMARS_DAOD550_unc[MetopC_IMARS_DAOD550_unc.mask == True ] = np.nan

    #######################
    ### MetopC IASI ULB ###
    #######################

    # finding Metop IASI ULB file.
    Metop_IASI_ULB_file    =  find_IASI_file(MetopC_IASI_ULB_path, file_sufix)
    
    if (Metop_IASI_ULB_file == 'non_existing_file'):
    # reading variables of interest.    
        empty_matrix          = np.empty((len(ERA_Longitude),len(ERA_Latitude)))
        empty_matrix[:]       = np.nan
        MetopC_ULB_DAOD550    = empty_matrix 
        MetopC_ULB_DAOD550[:] = np.nan        
    else:
    # reading variables of interest.
       Metop_IASI_ULB_dataset = nc.Dataset(Metop_IASI_ULB_file)
       ULB_Latitude           = Metop_IASI_ULB_dataset['latitude'][:]
       ULB_Longitude          = Metop_IASI_ULB_dataset['longitude'][:]       
       idx_lat                = np.where((ULB_Latitude  >= min(ERA_Latitude))  & (ULB_Latitude  <= max(ERA_Latitude) ))
       idx_lon                = np.where((ULB_Longitude >= min(ERA_Longitude)) & (ULB_Longitude <= max(ERA_Longitude)))   
       idx_lat                = np.squeeze(np.asarray(idx_lat))
       idx_lon                = np.squeeze(np.asarray(idx_lon))  
       ULB_DAOD550            = Metop_IASI_ULB_dataset['D_AOD550_mean'][idx_lat,idx_lon]
       MetopC_ULB_DAOD550     = np.transpose(ULB_DAOD550)
       MetopC_ULB_DAOD550     = np.transpose(ULB_DAOD550)
       if ma.isMaskedArray(MetopC_ULB_DAOD550) == True:
           MetopC_ULB_DAOD550[MetopC_ULB_DAOD550.mask == True ] = np.nan       

    #######################
    ### MetopC IASI LMD ###
    #######################

    # finding Metop IASI LMD file.
    Metop_IASI_LMD_file    =  find_IASI_file(MetopC_IASI_LMD_path, file_sufix)

    if (Metop_IASI_LMD_file == 'non_existing_file'):
    # reading variables of interest.    
        empty_matrix         = np.empty((len(ERA_Longitude),len(ERA_Latitude)))
        empty_matrix[:]      = np.nan
        MetopC_LMD_DAOD550   = empty_matrix 
        MetopC_LMD_DAOD550[:] = np.nan        
    else:
    # reading variables of interest.    
        Metop_IASI_LMD_dataset = nc.Dataset(Metop_IASI_LMD_file)
        LMD_Latitude           = Metop_IASI_LMD_dataset['Latitude'][:]  + 0.5
        LMD_Longitude          = Metop_IASI_LMD_dataset['Longitude'][:] + 0.5
        idx_lat                = np.where((LMD_Latitude  >= min(ERA_Latitude))  & (LMD_Latitude  <= max(ERA_Latitude) ))
        idx_lon                = np.where((LMD_Longitude >= min(ERA_Longitude)) & (LMD_Longitude <= max(ERA_Longitude)))   
        idx_lat                = np.squeeze(np.asarray(idx_lat))
        idx_lon                = np.squeeze(np.asarray(idx_lon))      
        LMD_DAOD550            = Metop_IASI_LMD_dataset['Daod550'][idx_lat,idx_lon]
        MetopC_LMD_DAOD550     = np.transpose(LMD_DAOD550)
        if ma.isMaskedArray(MetopC_LMD_DAOD550) == True:
           MetopC_LMD_DAOD550[MetopC_LMD_DAOD550.mask == True ] = np.nan       

    #########################
    ### MetopC IASI MARIP ###
    #########################    

    # finding Metop IASI MARIP file.
    Metop_IASI_MARIP_file    =  find_IASI_file(MetopC_IASI_MARIP_path, file_sufix)

    if (Metop_IASI_MARIP_file == 'non_existing_file'):
    # reading variables of interest.    
        empty_matrix         = np.empty((len(ERA_Longitude),len(ERA_Latitude)))
        empty_matrix[:]      = np.nan
        MetopC_MARIP_DAOD550 = empty_matrix        
    else:
    # reading variables of interest.    
        Metop_IASI_MARIP_dataset = nc.Dataset(Metop_IASI_MARIP_file)
        MARIP_DAOD550            = Metop_IASI_MARIP_dataset['D_AOD550_mean'][:]
        MARIP_Latitude           = Metop_IASI_MARIP_dataset['latitude'][:]
        MARIP_Longitude          = Metop_IASI_MARIP_dataset['longitude'][:]
        
        DOD_temp    = np.empty((len(ERA_Longitude),len(ERA_Latitude)))
        DOD_temp[:] = np.nan
        for count_lon,lon in enumerate(ERA_Longitude):
            for count_lat,lat in enumerate(ERA_Latitude):
                idx_lat = np.where(lat == MARIP_Latitude)
                idx_lon = np.where(lon == MARIP_Longitude)
                idx_lat = np.squeeze(np.asarray(idx_lat))
                idx_lon = np.squeeze(np.asarray(idx_lon))             
                if (idx_lon.size == 0) or (idx_lat.size == 0):
                    DOD_temp[count_lon,count_lat] = np.nan
                else:
                    DOD_temp[count_lon,count_lat] = MARIP_DAOD550[idx_lat,idx_lon]
        MetopC_MARIP_DAOD550 = DOD_temp                

    #######################
    ### LIVAS pure-dust ###
    #######################    

    Final_Number_of_Overpasses       = np.empty(( len(ERA_Longitude), len(ERA_Latitude)))
    Final_Number_of_L2Profiles       = np.empty(( len(ERA_Longitude), len(ERA_Latitude)))
    Final_DOD_532nm_mean             = np.empty(( len(ERA_Longitude), len(ERA_Latitude)))
    Final_DOD_532nm_SD               = np.empty(( len(ERA_Longitude), len(ERA_Latitude)))  
    Final_DOD_532nm_unc              = np.empty(( len(ERA_Longitude), len(ERA_Latitude)))      
    Final_PD_b532nm                  = np.empty(( len(ERA_Longitude), len(ERA_Latitude), 399))
    Final_PD_a532nm                  = np.empty(( len(ERA_Longitude), len(ERA_Latitude), 399))
    Final_PD_MC                      = np.empty(( len(ERA_Longitude), len(ERA_Latitude), 399))
    Final_PD_b532nm_unc              = np.empty(( len(ERA_Longitude), len(ERA_Latitude), 399))
    Final_PD_a532nm_unc              = np.empty(( len(ERA_Longitude), len(ERA_Latitude), 399))
    Final_PD_MC_unc                  = np.empty(( len(ERA_Longitude), len(ERA_Latitude), 399))
    Final_PD_b532nm_SD               = np.empty(( len(ERA_Longitude), len(ERA_Latitude), 399))
    Final_PD_a532nm_SD               = np.empty(( len(ERA_Longitude), len(ERA_Latitude), 399))
    Final_PD_MC_SD                   = np.empty(( len(ERA_Longitude), len(ERA_Latitude), 399))  
    Final_Marine                     = np.empty(( len(ERA_Longitude), len(ERA_Latitude), 399))
    Final_Dust                       = np.empty(( len(ERA_Longitude), len(ERA_Latitude), 399))
    Final_Polluted_Continental_Smoke = np.empty(( len(ERA_Longitude), len(ERA_Latitude), 399))
    Final_Clean_Continental          = np.empty(( len(ERA_Longitude), len(ERA_Latitude), 399))
    Final_Polluted_Dust              = np.empty(( len(ERA_Longitude), len(ERA_Latitude), 399))
    Final_Elevated_Smoke             = np.empty(( len(ERA_Longitude), len(ERA_Latitude), 399))
    Final_Dusty_Marine               = np.empty(( len(ERA_Longitude), len(ERA_Latitude), 399))      

    Final_Number_of_Overpasses[:]       = np.nan
    Final_Number_of_L2Profiles[:]       = np.nan
    Final_DOD_532nm_mean[:]             = np.nan
    Final_DOD_532nm_SD[:]               = np.nan      
    Final_PD_b532nm[:]                  = np.nan
    Final_PD_a532nm[:]                  = np.nan    
    Final_PD_MC[:]                      = np.nan
    Final_PD_b532nm_unc[:]              = np.nan
    Final_PD_a532nm_unc[:]              = np.nan    
    Final_PD_MC_unc[:]                  = np.nan
    Final_PD_b532nm_SD[:]               = np.nan
    Final_PD_a532nm_SD[:]               = np.nan    
    Final_PD_MC_SD[:]                   = np.nan
    Final_Marine[:]                     = 0
    Final_Dust[:]                       = 0
    Final_Polluted_Continental_Smoke[:] = 0
    Final_Clean_Continental[:]          = 0
    Final_Polluted_Dust[:]              = 0
    Final_Elevated_Smoke[:]             = 0
    Final_Dusty_Marine[:]               = 0      

    Empty_Vertical_array    = np.empty(399)
    Empty_Vertical_array[:] = np.nan  

    for count_lon,lon in enumerate(ERA_Longitude): 
        print(lon)
        for count_lat,lat in enumerate(ERA_Latitude):
                        
#            if ((lon < -25) | (lon > -15) | (lat > 25) | (lat < 15)):
#                continue
                        
            LIVAS_filename  = 'LIVAS_CALIPSO_L2_Grid_lon_c_' + str(lon) + '_lat_c_' + str(lat) + '.nc'
            
            LIVAS_file      = LIVAS_path + '\\' + LIVAS_filename           
            file_existing   = exists(LIVAS_file)
            
            if not file_existing:
                DOD_532nm_mean                = np.nan          
                DOD_532nm_SD                  = np.nan  
                PD_b_532nm                    = Empty_Vertical_array
                PD_a_532nm                    = Empty_Vertical_array
                PD_MC                         = Empty_Vertical_array
                PD_b_532nm_SD                 = Empty_Vertical_array
                PD_a_532nm_SD                 = Empty_Vertical_array
                PD_MC_SD                      = Empty_Vertical_array                    
                PD_b_532nm_unc                = Empty_Vertical_array
                PD_a_532nm_unc                = Empty_Vertical_array
                PD_MC_unc                     = Empty_Vertical_array              
                Number_of_Overpasses          = 0             
                L2_CF_profiles                = 0
                Marine                        = np.array(np.empty((399))) 
                Dust                          = np.array(np.empty((399))) 
                Polluted_Continental_Smoke    = np.array(np.empty((399))) 
                Clean_Continental             = np.array(np.empty((399))) 
                Polluted_Dust                 = np.array(np.empty((399))) 
                Elevated_Smoke                = np.array(np.empty((399))) 
                Dusty_Marine                  = np.array(np.empty((399))) 
                Marine[:]                     = 0
                Dust[:]                       = 0
                Polluted_Continental_Smoke[:] = 0
                Clean_Continental[:]          = 0
                Polluted_Dust[:]              = 0
                Elevated_Smoke[:]             = 0
                Dusty_Marine[:]               = 0                
            else:
           
                LIVAS_dataset       = nc.Dataset(LIVAS_file)
                
                Profile_Time_Parsed = LIVAS_dataset['/Profile_Time_Parsed'][:]
                Days      = np.empty((len(Profile_Time_Parsed)))
                Months    = np.empty((len(Profile_Time_Parsed)))
                Years     = np.empty((len(Profile_Time_Parsed)))
                Days[:]   = np.nan
                Months[:] = np.nan
                Years[:]  = np.nan
                for count_time,time in enumerate(Profile_Time_Parsed):
                    time = time.split(' ')[0]
                    Days[count_time]   = int(time.split('/')[2])
                    Months[count_time] = int(time.split('/')[1])
                    Years[count_time]  = int(time.split('/')[0])
                idx = np.where((ERA_month == Months) & (ERA_year == Years))
                idx = np.ravel(idx)
                Day   = Days[idx]
                Month = Months[idx]
                Year  = Years[idx]
                                
                if len(idx) == 0:				
                    DOD_532nm_mean       = np.nan          
                    DOD_532nm_SD         = np.nan
                    DOD_532nm_unc        = np.nan                    
                    PD_b_532nm           = Empty_Vertical_array
                    PD_a_532nm           = Empty_Vertical_array
                    PD_MC                = Empty_Vertical_array
                    PD_b_532nm_SD        = Empty_Vertical_array
                    PD_a_532nm_SD        = Empty_Vertical_array
                    PD_MC_SD             = Empty_Vertical_array                    
                    PD_b_532nm_unc       = Empty_Vertical_array
                    PD_a_532nm_unc       = Empty_Vertical_array
                    PD_MC_unc            = Empty_Vertical_array
                    Number_of_Overpasses = 0             
                    L2_CF_profiles       = 0
                    Marine                     = np.array(np.empty((399))) 
                    Dust                       = np.array(np.empty((399))) 
                    Polluted_Continental_Smoke = np.array(np.empty((399))) 
                    Clean_Continental          = np.array(np.empty((399))) 
                    Polluted_Dust              = np.array(np.empty((399))) 
                    Elevated_Smoke             = np.array(np.empty((399))) 
                    Dusty_Marine               = np.array(np.empty((399))) 
                    Marine[:]                     = 0
                    Dust[:]                       = 0
                    Polluted_Continental_Smoke[:] = 0
                    Clean_Continental[:]          = 0
                    Polluted_Dust[:]              = 0
                    Elevated_Smoke[:]             = 0
                    Dusty_Marine[:]               = 0                       
                else:                           
                    Altitude             = LIVAS_dataset['/Altitude'][:]
                    LIVAS_PD_b532nm      = LIVAS_dataset['/LIVAS/Cloud_Free/Pure_Dust_and_Fine_Coarse/Optical_Products/Pure_Dust_Backscatter_Coefficient_532'][idx,:]
                    LIVAS_PD_a532nm      = LIVAS_dataset['/LIVAS/Cloud_Free/Pure_Dust_and_Fine_Coarse/Optical_Products/Pure_Dust_Extinction_Coefficient_532'][idx,:]  
                    LIVAS_PD_MC          = LIVAS_dataset['/LIVAS/Cloud_Free/Pure_Dust_and_Fine_Coarse/Mass_Concentrations/Pure_Dust_Mass_Concentration'][idx,:]     
                    LIVAS_PD_b532nm_unc  = LIVAS_dataset['/LIVAS/Cloud_Free/Pure_Dust_and_Fine_Coarse/Optical_Products/Pure_Dust_Backscatter_Coefficient_532_Uncertainty'][idx,:]
                    LIVAS_PD_a532nm_unc  = LIVAS_dataset['/LIVAS/Cloud_Free/Pure_Dust_and_Fine_Coarse/Optical_Products/Pure_Dust_Extinction_Coefficient_532_Uncertainty'][idx,:]  
                    LIVAS_PD_MC_unc      = LIVAS_dataset['/LIVAS/Cloud_Free/Pure_Dust_and_Fine_Coarse/Mass_Concentrations/Pure_Dust_Mass_Concentration_Uncertainty'][idx,:]
                    AVD_Aerosol_Subtype  = LIVAS_dataset['/CALIPSO_Flags_and_Auxiliary/Flags/AVD_Aerosol_Subtype'][idx,:]

                    idx = np.where(LIVAS_PD_b532nm == 0)
                    LIVAS_PD_MC[idx]         = 0
                    LIVAS_PD_b532nm_unc[idx] = 0
                    LIVAS_PD_a532nm_unc[idx] = 0
                    LIVAS_PD_MC_unc[idx]     = 0
                    
                    if ma.isMaskedArray(LIVAS_PD_b532nm) == True:
                        LIVAS_PD_a532nm[LIVAS_PD_b532nm.mask == True ]     = np.nan
                        LIVAS_PD_MC[LIVAS_PD_b532nm.mask == True ]         = np.nan
                        LIVAS_PD_b532nm_unc[LIVAS_PD_b532nm.mask == True ] = np.nan
                        LIVAS_PD_a532nm_unc[LIVAS_PD_b532nm.mask == True ] = np.nan
                        LIVAS_PD_MC_unc[LIVAS_PD_b532nm.mask == True ]     = np.nan                   
                        LIVAS_PD_b532nm[LIVAS_PD_b532nm.mask == True ]     = np.nan
                    
                    Number_of_Overpasses = len(np.unique(np.multiply(Month,Day)))
                    temp = np.ravel(np.count_nonzero(~np.isnan(LIVAS_PD_b532nm),axis=1))
                    L2_CF_profiles  = len(temp) - len(np.ravel(np.where(temp == 0))) 
                    
                    if (L2_CF_profiles == 0):
                        DOD_532nm_mean       = np.nan          
                        DOD_532nm_SD         = np.nan  
                        DOD_532nm_unc        = np.nan                        
                        PD_b_532nm           = Empty_Vertical_array
                        PD_a_532nm           = Empty_Vertical_array
                        PD_MC                = Empty_Vertical_array
                        PD_b_532nm_SD        = Empty_Vertical_array
                        PD_a_532nm_SD        = Empty_Vertical_array
                        PD_MC_SD             = Empty_Vertical_array                    
                        PD_b_532nm_unc       = Empty_Vertical_array
                        PD_a_532nm_unc       = Empty_Vertical_array
                        PD_MC_unc            = Empty_Vertical_array
                        Marine                     = np.array(np.empty((399))) 
                        Dust                       = np.array(np.empty((399))) 
                        Polluted_Continental_Smoke = np.array(np.empty((399))) 
                        Clean_Continental          = np.array(np.empty((399))) 
                        Polluted_Dust              = np.array(np.empty((399))) 
                        Elevated_Smoke             = np.array(np.empty((399))) 
                        Dusty_Marine               = np.array(np.empty((399))) 
                        Marine[:]                     = 0
                        Dust[:]                       = 0
                        Polluted_Continental_Smoke[:] = 0
                        Clean_Continental[:]          = 0
                        Polluted_Dust[:]              = 0
                        Elevated_Smoke[:]             = 0
                        Dusty_Marine[:]               = 0                          
                    if (L2_CF_profiles == 1):
                        PD_b_532nm           = np.nanmean(LIVAS_PD_b532nm, axis = 0)
                        PD_a_532nm           = np.nanmean(LIVAS_PD_a532nm, axis = 0)
                        PD_MC                = np.nanmean(LIVAS_PD_MC,     axis = 0)                                
                        PD_b_532nm_SD        = Empty_Vertical_array
                        PD_a_532nm_SD        = Empty_Vertical_array
                        PD_MC_SD             = Empty_Vertical_array                    
                        PD_b_532nm_unc       = np.nanmean(LIVAS_PD_b532nm_unc, axis = 0)
                        PD_a_532nm_unc       = np.nanmean(LIVAS_PD_a532nm_unc, axis = 0)
                        PD_MC_unc            = np.nanmean(LIVAS_PD_MC_unc,     axis = 0)   
                        arr = np.copy(PD_a_532nm)
                        arr[np.isnan(arr)] = 0                     
                        DOD_532nm_mean       = np.trapz(Altitude,arr) 
                        arr = np.copy(PD_a_532nm_SD)
                        arr[np.isnan(arr)] = 0                            
                        DOD_532nm_SD         = np.trapz(Altitude,arr) 
                        arr = np.copy(PD_a_532nm_unc)
                        arr[np.isnan(arr)] = 0                     
                        DOD_532nm_unc        = np.trapz(Altitude,arr)
                        [ Marine, Dust, Polluted_Continental_Smoke, Clean_Continental, Polluted_Dust, Elevated_Smoke, Dusty_Marine ] = CALIPSO_Aerosol_Subtypes( AVD_Aerosol_Subtype, L2_CF_profiles )
                    if (L2_CF_profiles > 1):                    
                        PD_b_532nm           = np.nanmean(LIVAS_PD_b532nm, axis = 0)
                        PD_a_532nm           = np.nanmean(LIVAS_PD_a532nm, axis = 0)
                        PD_MC                = np.nanmean(LIVAS_PD_MC,     axis = 0)
                        PD_b_532nm_SD        = np.nanstd(LIVAS_PD_b532nm,  axis = 0, ddof = 1)
                        PD_a_532nm_SD        = np.nanstd(LIVAS_PD_a532nm,  axis = 0, ddof = 1)
                        PD_MC_SD             = np.nanstd(LIVAS_PD_MC,      axis = 0, ddof = 1)                                            
                        LIVAS_PD_b532nm_unc[np.where(LIVAS_PD_b532nm == 0)] = 0.0
                        PD_b_532nm_unc = np.divide(np.sqrt(np.nansum(np.square(LIVAS_PD_b532nm_unc),axis=0)), np.count_nonzero(~np.isnan(LIVAS_PD_b532nm_unc),axis=0))
                        LIVAS_PD_a532nm_unc[np.where(LIVAS_PD_b532nm == 0)] = 0.0    
                        PD_a_532nm_unc = np.divide(np.sqrt(np.nansum(np.square(LIVAS_PD_a532nm_unc),axis=0)), np.count_nonzero(~np.isnan(LIVAS_PD_a532nm_unc),axis=0))
                        LIVAS_PD_MC_unc[np.where(LIVAS_PD_b532nm == 0)]     = 0.0     
                        PD_MC_unc = np.divide(np.sqrt(np.nansum(np.square(LIVAS_PD_MC_unc),axis=0)), np.count_nonzero(~np.isnan(LIVAS_PD_MC_unc),axis=0))
                        arr = np.copy(PD_a_532nm)
                        arr[np.isnan(arr)] = 0                     
                        DOD_532nm_mean       = np.trapz(Altitude,arr) 
                        arr = np.copy(PD_a_532nm_SD)
                        arr[np.isnan(arr)] = 0                            
                        DOD_532nm_SD         = np.trapz(Altitude,arr) 
                        arr = np.copy(PD_a_532nm_unc)
                        arr[np.isnan(arr)] = 0                     
                        DOD_532nm_unc        = np.trapz(Altitude,arr) 
                        [ Marine, Dust, Polluted_Continental_Smoke, Clean_Continental, Polluted_Dust, Elevated_Smoke, Dusty_Marine ] = CALIPSO_Aerosol_Subtypes( AVD_Aerosol_Subtype, L2_CF_profiles )
    
            Final_Number_of_L2Profiles[count_lon,count_lat] = L2_CF_profiles
            Final_Number_of_Overpasses[count_lon,count_lat] = Number_of_Overpasses            
            Final_DOD_532nm_mean[count_lon,count_lat]       = DOD_532nm_mean
            Final_DOD_532nm_SD[count_lon,count_lat]         = DOD_532nm_SD
            Final_DOD_532nm_unc[count_lon,count_lat]        = DOD_532nm_unc
            for count_alt in range(399):   
                Final_PD_b532nm[count_lon,count_lat,count_alt]     = PD_b_532nm[count_alt]
                Final_PD_a532nm[count_lon,count_lat,count_alt]     = PD_a_532nm[count_alt]    
                Final_PD_MC[count_lon,count_lat,count_alt]         = PD_MC[count_alt]
                Final_PD_b532nm_unc[count_lon,count_lat,count_alt] = PD_b_532nm_unc[count_alt]
                Final_PD_a532nm_unc[count_lon,count_lat,count_alt] = PD_a_532nm_unc[count_alt]   
                Final_PD_MC_unc[count_lon,count_lat,count_alt]     = PD_MC_unc[count_alt] 
                Final_PD_b532nm_SD[count_lon,count_lat,count_alt]  = PD_b_532nm_SD[count_alt] 
                Final_PD_a532nm_SD[count_lon,count_lat,count_alt]  = PD_a_532nm_SD[count_alt]    
                Final_PD_MC_SD[count_lon,count_lat,count_alt]      = PD_MC_SD[count_alt]
                Final_Marine[count_lon,count_lat,count_alt]                     = Marine[count_alt]
                Final_Dust[count_lon,count_lat,count_alt]                       = Dust[count_alt]
                Final_Polluted_Continental_Smoke[count_lon,count_lat,count_alt] = Polluted_Continental_Smoke[count_alt]
                Final_Clean_Continental[count_lon,count_lat,count_alt]          = Clean_Continental[count_alt]
                Final_Polluted_Dust[count_lon,count_lat,count_alt]              = Polluted_Dust[count_alt]
                Final_Elevated_Smoke[count_lon,count_lat,count_alt]             = Elevated_Smoke[count_alt]
                Final_Dusty_Marine[count_lon,count_lat,count_alt]               = Dusty_Marine[count_alt]   

    #####################################            
    #  --- Saving dataset as NetCDF --- #
    #####################################

    # creating nc. filename and initiallizing:                 
    fn           = output_path + '\\' + "DOMOS_Datasets_" + file_sufix + ".nc"
    ds           = nc.Dataset(fn, 'w', format='NETCDF4')
    
    # create nc. dimensions:
    longitude    = ERA_Longitude
    latitude     = ERA_Latitude        
    ERA_lev      = ds.createDimension('ERA_lev', ERA_Height.shape[2])
    lat          = ds.createDimension('lat',     len(ERA_Latitude)) 
    lon          = ds.createDimension('lon',     len(ERA_Longitude))
    
    Geolocation_group     = ds.createGroup("Geolocation")
    ERA5_group            = ds.createGroup("ERA5")
    LIVAS_group           = ds.createGroup("EO_pure-dust_products/LIVAS")
    MIDAS_group           = ds.createGroup("EO_pure-dust_products/MIDAS")
    MetopA_group          = ds.createGroup("EO_pure-dust_products/METOP/MetopA-IASI")
    MetopC_group          = ds.createGroup("EO_pure-dust_products/METOP/MetopC-IASI")
    Land_Ocean_Mask_group = ds.createGroup("Land_Ocean_Mask")
    
    # create nc. variables:
    lats                         = ds.createVariable('Geolocation/Latitude', 'f4', ('lat',))
    lons                         = ds.createVariable('Geolocation/Longitude','f4', ('lon',))
    
    Height                       = ds.createVariable('ERA5/Height', 'f4',     ('lon','lat','ERA_lev',))
    U                            = ds.createVariable('ERA5/U',    np.float64, ('lon','lat','ERA_lev',))
    U_SD                         = ds.createVariable('ERA5/U_SD', np.float64, ('lon','lat','ERA_lev',))
    V                            = ds.createVariable('ERA5/V',    np.float64, ('lon','lat','ERA_lev',))
    V_SD                         = ds.createVariable('ERA5/V_SD', np.float64, ('lon','lat','ERA_lev',))      
   
    MIDAS_DOD550_mean            = ds.createVariable('EO_pure-dust_products/MIDAS/MIDAS_DAOD550_mean',       np.float64, ('lon','lat',))
    MIDAS_DOD550_std             = ds.createVariable('EO_pure-dust_products/MIDAS/MIDAS_DAOD550_std',        np.float64, ('lon','lat',))

    lev_dim                      = ds.createDimension('lev_dim', len(Altitude))
    L2A_dim                      = ds.createDimension('L2A_dim', Number_of_Overpasses)
    Scalar_dim                   = ds.createDimension('Scalar_dim', 1)
    
    LIVAS_Altitude               = ds.createVariable('EO_pure-dust_products/LIVAS/Altitude',                                             'f4',        ('lev_dim',))
    LIVAS_PD_b_532nm             = ds.createVariable('EO_pure-dust_products/LIVAS/Pure_Dust/LIVAS_PD_b_532nm',             np.float64,  ('lon','lat','lev_dim',))
    LIVAS_PD_a_532nm             = ds.createVariable('EO_pure-dust_products/LIVAS/Pure_Dust/LIVAS_PD_a_532nm',             np.float64,  ('lon','lat','lev_dim',))
    LIVAS_PD_MC                  = ds.createVariable('EO_pure-dust_products/LIVAS/Pure_Dust/LIVAS_PD_MC',                  np.float64,  ('lon','lat','lev_dim',))
    LIVAS_PD_b_532nm_unc         = ds.createVariable('EO_pure-dust_products/LIVAS/Pure_Dust/LIVAS_PD_b_532nm_unc',         np.float64,  ('lon','lat','lev_dim',))
    LIVAS_PD_a_532nm_unc         = ds.createVariable('EO_pure-dust_products/LIVAS/Pure_Dust/LIVAS_PD_a_532nm_unc',         np.float64,  ('lon','lat','lev_dim',))
    LIVAS_PD_MC_unc              = ds.createVariable('EO_pure-dust_products/LIVAS/Pure_Dust/LIVAS_PD_MC_unc',              np.float64,  ('lon','lat','lev_dim',))
    LIVAS_PD_b_532nm_SD          = ds.createVariable('EO_pure-dust_products/LIVAS/Pure_Dust/LIVAS_PD_b_532nm_STD',         np.float64,  ('lon','lat','lev_dim',))
    LIVAS_PD_a_532nm_SD          = ds.createVariable('EO_pure-dust_products/LIVAS/Pure_Dust/LIVAS_PD_a_532nm_STD',         np.float64,  ('lon','lat','lev_dim',))
    LIVAS_PD_MC_SD               = ds.createVariable('EO_pure-dust_products/LIVAS/Pure_Dust/LIVAS_PD_MC_STD',              np.float64,  ('lon','lat','lev_dim',))    
    LIVAS_N_of_CF_Pro            = ds.createVariable('EO_pure-dust_products/LIVAS/Flags/Number_of_L2A_CF_Profiles',        np.float64,  ('lon','lat',))
    LIVAS_N_of_Ocerpasses        = ds.createVariable('EO_pure-dust_products/LIVAS/Flags/Number_of_CALIPSO_Overpasses',     np.float64,  ('lon','lat',))
    LIVAS_DOD_532nm_mean         = ds.createVariable('EO_pure-dust_products/LIVAS/Pure_Dust/DOD_532nm_mean',               np.float64,  ('lon','lat',))
    LIVAS_DOD_532nm_unc          = ds.createVariable('EO_pure-dust_products/LIVAS/Pure_Dust/DOD_532nm_unc',                np.float64,  ('lon','lat',))
    LIVAS_DOD_532nm_SD           = ds.createVariable('EO_pure-dust_products/LIVAS/Pure_Dust/DOD_532nm_STD',                np.float64,  ('lon','lat',))
    LIVAS_Final_Marine                     = ds.createVariable('EO_pure-dust_products/LIVAS/AVD_Aerosol_Subtype/Marine',             np.float64,  ('lon','lat','lev_dim',)) 
    LIVAS_Final_Dust                       = ds.createVariable('EO_pure-dust_products/LIVAS/AVD_Aerosol_Subtype/Dust',               np.float64,  ('lon','lat','lev_dim',)) 
    LIVAS_Final_Polluted_Continental_Smoke = ds.createVariable('EO_pure-dust_products/LIVAS/AVD_Aerosol_Subtype/Polluted_Continental_Smoke', np.float64,  ('lon','lat','lev_dim',)) 
    LIVAS_Final_Clean_Continental          = ds.createVariable('EO_pure-dust_products/LIVAS/AVD_Aerosol_Subtype/Clean_Continental',          np.float64,  ('lon','lat','lev_dim',)) 
    LIVAS_Final_Polluted_Dust              = ds.createVariable('EO_pure-dust_products/LIVAS/AVD_Aerosol_Subtype/Polluted_Dust',              np.float64,  ('lon','lat','lev_dim',)) 
    LIVAS_Final_Elevated_Smoke             = ds.createVariable('EO_pure-dust_products/LIVAS/AVD_Aerosol_Subtype/Elevated_Smoke',             np.float64,  ('lon','lat','lev_dim',)) 
    LIVAS_Final_Dusty_Marine               = ds.createVariable('EO_pure-dust_products/LIVAS/AVD_Aerosol_Subtype/Dusty_Marine',               np.float64,  ('lon','lat','lev_dim',))       

    MetopA_IASI_ENS_DAOD550      = ds.createVariable('EO_pure-dust_products/METOP/MetopA-IASI/MetopA_IASI_ENS_DAOD550',       np.float64, ('lon','lat',))
    MetopA_IASI_ENS_DAOD550_unc  = ds.createVariable('EO_pure-dust_products/METOP/MetopA-IASI/MetopA_IASI_ENS_DAOD550_unc',   np.float64, ('lon','lat',))
    MetopA_IASI_IMARS_DAOD550    = ds.createVariable('EO_pure-dust_products/METOP/MetopA-IASI/MetopA_IASI_IMARS_DAOD550',     np.float64, ('lon','lat',))
    MetopA_IASI_IMARS_DAOD550_unc= ds.createVariable('EO_pure-dust_products/METOP/MetopA-IASI/MetopA_IASI_IMARS_DAOD550_unc', np.float64, ('lon','lat',))
    MetopA_IASI_ULB_DAOD550      = ds.createVariable('EO_pure-dust_products/METOP/MetopA-IASI/MetopA_IASI_ULB_DAOD550',       np.float64, ('lon','lat',))
    MetopA_IASI_LMD_DAOD550      = ds.createVariable('EO_pure-dust_products/METOP/MetopA-IASI/MetopA_IASI_LMD_DAOD550',       np.float64, ('lon','lat',))
    MetopA_IASI_MARIP_DAOD550    = ds.createVariable('EO_pure-dust_products/METOP/MetopA-IASI/MetopA_IASI_MARIP_DAOD550',     np.float64, ('lon','lat',))

    MetopC_IASI_ENS_DAOD550      = ds.createVariable('EO_pure-dust_products/METOP/MetopC-IASI/MetopC_IASI_ENS_DAOD550',       np.float64, ('lon','lat',))
    MetopC_IASI_ENS_DAOD550_unc  = ds.createVariable('EO_pure-dust_products/METOP/MetopC-IASI/MetopC_IASI_ENS_DAOD550_unc',   np.float64, ('lon','lat',))
    MetopC_IASI_IMARS_DAOD550    = ds.createVariable('EO_pure-dust_products/METOP/MetopC-IASI/MetopC_IASI_IMARS_DAOD550',     np.float64, ('lon','lat',))
    MetopC_IASI_IMARS_DAOD550_unc= ds.createVariable('EO_pure-dust_products/METOP/MetopC-IASI/MetopC_IASI_IMARS_DAOD550_unc', np.float64, ('lon','lat',))
    MetopC_IASI_ULB_DAOD550      = ds.createVariable('EO_pure-dust_products/METOP/MetopC-IASI/MetopC_IASI_ULB_DAOD550',       np.float64, ('lon','lat',))
    MetopC_IASI_LMD_DAOD550      = ds.createVariable('EO_pure-dust_products/METOP/MetopC-IASI/MetopC_IASI_LMD_DAOD550',       np.float64, ('lon','lat',))
    MetopC_IASI_MARIP_DAOD550    = ds.createVariable('EO_pure-dust_products/METOP/MetopC-IASI/MetopC_IASI_MARIP_DAOD550',     np.float64, ('lon','lat',))
    
    LOM_lat                      = ds.createDimension('LOM_lat', len(Land_Mask_Latitude)) 
    LOM_lon                      = ds.createDimension('LOM_lon', len(Land_Mask_Longitude))    
    Land_Ocean_Mask_id           = ds.createVariable('Land_Ocean_Mask/DOMOS_cells',     np.float64, ('lon','lat',))
    
    # nc. variables' units
    lats.units                        = 'degrees_north'
    lons.units                        = 'degrees_east'
    Height.units                      = 'm'     
    U.units                           = 'm s**-1'
    U_SD.units                        = 'm s**-1'
    V.units                           = 'm s**-1'
    V_SD.units                        = 'm s**-1'
    LIVAS_Altitude.units              = 'km'
    LIVAS_N_of_CF_Pro.units           = 'none'
    LIVAS_DOD_532nm_mean.units        = 'none'
    LIVAS_DOD_532nm_SD.units          = 'none'
    LIVAS_DOD_532nm_unc.units         = 'none'
    LIVAS_PD_b_532nm.units            = 'km-1sr-1'
    LIVAS_PD_a_532nm.units            = 'km-1'
    LIVAS_PD_MC.units                 = 'micrograms/m^3'
    LIVAS_PD_b_532nm_unc.units        = 'km-1sr-1'
    LIVAS_PD_a_532nm_unc.units        = 'km-1'
    LIVAS_PD_MC_unc.units             = 'micrograms/m^3'
    LIVAS_PD_b_532nm_SD.units         = 'km-1sr-1'
    LIVAS_PD_a_532nm_SD.units         = 'km-1'
    LIVAS_PD_MC_SD.units              = 'micrograms/m^3'    
    LIVAS_Final_Marine.units          = 'none'
    LIVAS_Final_Dust.units            = 'none'
    LIVAS_Final_Polluted_Continental_Smoke.units = 'none'
    LIVAS_Final_Clean_Continental.units          = 'none'
    LIVAS_Final_Polluted_Dust.units              = 'none'
    LIVAS_Final_Elevated_Smoke.units             = 'none'
    LIVAS_Final_Dusty_Marine.units               = 'none'     

    MetopA_IASI_ENS_DAOD550.units       = 'none'
    MetopA_IASI_ENS_DAOD550_unc.units   = 'none'
    MetopA_IASI_IMARS_DAOD550.units     = 'none'
    MetopA_IASI_IMARS_DAOD550_unc.units = 'none'   
    MetopA_IASI_ULB_DAOD550.units       = 'none'
    MetopA_IASI_LMD_DAOD550.units       = 'none'
    MetopA_IASI_MARIP_DAOD550.units     = 'none'    
    
    MetopC_IASI_ENS_DAOD550.units       = 'none'
    MetopC_IASI_ENS_DAOD550_unc.units   = 'none'
    MetopC_IASI_IMARS_DAOD550.units     = 'none'
    MetopC_IASI_IMARS_DAOD550_unc.units = 'none'   
    MetopC_IASI_ULB_DAOD550.units       = 'none'
    MetopC_IASI_LMD_DAOD550.units       = 'none'
    MetopC_IASI_MARIP_DAOD550.units     = 'none'     
    
    MIDAS_DOD550_mean.units         = 'none'
    MIDAS_DOD550_std.units          = 'none'
    
    # nc. variables' "long names":
    lats.long_name                      = 'Latitude'
    lons.long_name                      = 'Longitude'
    Height.long_name                    = 'Height'
    U.long_name                         = 'U component of wind'
    U_SD.long_name                      = 'U component of wind SD'
    V.long_name                         = 'V component of wind'
    V_SD.long_name                      = 'V component of wind SD'
    LIVAS_Altitude.long_name            = 'Height'
    LIVAS_N_of_CF_Pro.long_name         = 'Number of L2 Cloud-Free profiles'
    LIVAS_DOD_532nm_mean.long_name      = 'Dust Optical Depth 532nm - mean'
    LIVAS_DOD_532nm_unc.long_name       = 'Dust Optical Depth 532nm - unc'
    LIVAS_DOD_532nm_SD.long_name        = 'Dust Optical Depth 532nm - STD'
    LIVAS_PD_b_532nm.long_name          = 'Pure-Dust Backscatter Coefficient 532nm'
    LIVAS_PD_a_532nm.long_name          = 'Pure-Dust Extinction Coefficient 532nm'
    LIVAS_PD_MC.long_name               = 'Pure-Dust Mass Concentration'
    LIVAS_PD_b_532nm_unc.long_name      = 'Pure-Dust Backscatter Coefficient 532nm - unc'
    LIVAS_PD_a_532nm_unc.long_name      = 'Pure-Dust Extinction Coefficient 532nm - unc'
    LIVAS_PD_MC_unc.long_name           = 'Pure-Dust Mass Concentration - unc'   
    LIVAS_PD_b_532nm_SD.long_name       = 'Pure-Dust Backscatter Coefficient 532nm - STD'
    LIVAS_PD_a_532nm_SD.long_name       = 'Pure-Dust Extinction Coefficient 532nm - STD'
    LIVAS_PD_MC_SD.long_name            = 'Pure-Dust Mass Concentration - STD'   
    LIVAS_Final_Marine.long_name        = 'Marine'
    LIVAS_Final_Dust.long_name          = 'Dust'
    LIVAS_Final_Polluted_Continental_Smoke.long_name = 'Polluted Continental and Smoke'
    LIVAS_Final_Clean_Continental.long_name          = 'Clean Continental'
    LIVAS_Final_Polluted_Dust.long_name              = 'Polluted Dust'
    LIVAS_Final_Elevated_Smoke.long_name             = 'Elevated Smoke'
    LIVAS_Final_Dusty_Marine.long_name               = 'Dusty Marine'
     
    MetopA_IASI_ENS_DAOD550.long_name       = 'Dust Optical Depth at 0.55 microns'
    MetopA_IASI_ENS_DAOD550_unc.long_name   = 'Dust Optical Depth at 0.55 microns - unc'
    MetopA_IASI_IMARS_DAOD550.long_name     = 'Dust Optical Depth at 0.55 microns'
    MetopA_IASI_IMARS_DAOD550_unc.long_name = 'Dust Optical Depth at 0.55 microns - unc'
    MetopA_IASI_ULB_DAOD550.long_name       = 'Dust Optical Depth at 0.55 microns'
    MetopA_IASI_LMD_DAOD550.long_name       = 'Dust Optical Depth at 0.55 microns'
    MetopA_IASI_MARIP_DAOD550.long_name     = 'Dust Optical Depth at 0.55 microns'     
    MetopC_IASI_ENS_DAOD550.long_name       = 'Dust Optical Depth at 0.55 microns'
    MetopC_IASI_ENS_DAOD550_unc.long_name   = 'Dust Optical Depth at 0.55 microns - unc'
    MetopC_IASI_IMARS_DAOD550.long_name     = 'Dust Optical Depth at 0.55 microns'
    MetopC_IASI_IMARS_DAOD550_unc.long_name = 'Dust Optical Depth at 0.55 microns - unc'
    MetopC_IASI_ULB_DAOD550.long_name   = 'Dust Optical Depth at 0.55 microns'
    MetopC_IASI_LMD_DAOD550.long_name   = 'Dust Optical Depth at 0.55 microns'
    MetopC_IASI_MARIP_DAOD550.long_name = 'Dust Optical Depth at 0.55 microns'    
    MIDAS_DOD550_mean.long_name         = 'Dust Optical Depth at 0.55 microns - mean'
    MIDAS_DOD550_std.long_name          = 'Dust Optical Depth at 0.55 microns - STD'
    
    # nc. variables' "standard names":
    Height.standard_name                    = 'height'
    U.standard_name                         = 'eastward_wind'
    U_SD.standard_name                      = 'eastward_wind_SD'
    V.standard_name                         = 'northward_wind'
    V_SD.standard_name                      = 'northward_wind_SD'        
    LIVAS_Altitude.standard_name            = 'height'
    LIVAS_N_of_CF_Pro.standard_name         = 'Number_of_L2_Cloud-Free_profiles'
    LIVAS_DOD_532nm_mean.standard_name      = 'atmosphere_optical_thickness_due_to_dust_aerosol_mean'
    LIVAS_DOD_532nm_SD.standard_name        = 'atmosphere_optical_thickness_due_to_dust_aerosol_std'
    LIVAS_PD_b_532nm.standard_name          = 'Backscatter_Coefficient_532'
    LIVAS_PD_a_532nm.standard_name          = 'Extinction_Coefficient_532'
    LIVAS_PD_MC.standard_name               = 'Mass_Concentration'
    LIVAS_PD_b_532nm_unc.standard_name      = 'Backscatter_Coefficient_532nm_unc'
    LIVAS_PD_a_532nm_unc.standard_name      = 'Extinction_Coefficient_532_unc'
    LIVAS_PD_MC_unc.standard_name           = 'Mass_Concentration_unc'
    MetopA_IASI_ENS_DAOD550.standard_name   = 'atmosphere_optical_thickness_due_to_dust_aerosol'
    MetopA_IASI_ENS_DAOD550_unc.standard_name   = 'atmosphere_optical_thickness_due_to_dust_aerosol_unc'
    MetopA_IASI_IMARS_DAOD550.standard_name     = 'atmosphere_optical_thickness_due_to_dust_aerosol'
    MetopA_IASI_IMARS_DAOD550_unc.standard_name = 'atmosphere_optical_thickness_due_to_dust_aerosol_unc'
    MetopA_IASI_ULB_DAOD550.standard_name       = 'atmosphere_optical_thickness_due_to_dust_aerosol'
    MetopA_IASI_LMD_DAOD550.standard_name       = 'atmosphere_optical_thickness_due_to_dust_aerosol'
    MetopA_IASI_MARIP_DAOD550.standard_name     = 'atmosphere_optical_thickness_due_to_dust_aerosol'  
    MetopC_IASI_ENS_DAOD550.standard_name       = 'atmosphere_optical_thickness_due_to_dust_aerosol'
    MetopC_IASI_ENS_DAOD550_unc.standard_name   = 'atmosphere_optical_thickness_due_to_dust_aerosol_unc'
    MetopC_IASI_IMARS_DAOD550.standard_name     = 'atmosphere_optical_thickness_due_to_dust_aerosol'
    MetopC_IASI_IMARS_DAOD550_unc.standard_name = 'atmosphere_optical_thickness_due_to_dust_aerosol_unc'
    MetopC_IASI_ULB_DAOD550.standard_name       = 'atmosphere_optical_thickness_due_to_dust_aerosol'
    MetopC_IASI_LMD_DAOD550.standard_name       = 'atmosphere_optical_thickness_due_to_dust_aerosol'
    MetopC_IASI_MARIP_DAOD550.standard_name     = 'atmosphere_optical_thickness_due_to_dust_aerosol'      
    MIDAS_DOD550_mean.standard_name             = 'atmosphere_optical_thickness_due_to_dust_aerosol_mean'
    MIDAS_DOD550_std.standard_name              = 'atmosphere_optical_thickness_due_to_dust_aerosol_std'

    # nc. variables' "fill values":
    Height.fill_value                        = np.nan
    U.fill_value                             = np.nan
    U_SD.fill_value                          = np.nan
    V.fill_value                             = np.nan
    V_SD.fill_value                          = np.nan
    LIVAS_Altitude.fill_value                = np.nan
    LIVAS_N_of_CF_Pro.fill_value             = np.nan
    LIVAS_DOD_532nm_mean.fill_value          = np.nan
    LIVAS_DOD_532nm_SD.fill_value            = np.nan
    LIVAS_PD_b_532nm.fill_value              = np.nan
    LIVAS_PD_a_532nm.fill_value              = np.nan
    LIVAS_PD_MC.fill_value                   = np.nan
    LIVAS_PD_b_532nm_unc.fill_value          = np.nan
    LIVAS_PD_a_532nm_unc.fill_value          = np.nan
    LIVAS_PD_MC_unc.fill_value               = np.nan
    MetopA_IASI_ENS_DAOD550.fill_value       = np.nan
    MetopA_IASI_ENS_DAOD550_unc.fill_value   = np.nan
    MetopA_IASI_IMARS_DAOD550.fill_value     = np.nan
    MetopA_IASI_IMARS_DAOD550_unc.fill_value = np.nan
    MetopA_IASI_ULB_DAOD550.fill_value       = np.nan
    MetopA_IASI_LMD_DAOD550.fill_value       = np.nan
    MetopA_IASI_MARIP_DAOD550.fill_value     = np.nan
    MetopC_IASI_ENS_DAOD550.fill_value       = np.nan
    MetopC_IASI_ENS_DAOD550_unc.fill_value   = np.nan
    MetopC_IASI_IMARS_DAOD550.fill_value     = np.nan
    MetopC_IASI_IMARS_DAOD550_unc.fill_value = np.nan
    MetopC_IASI_ULB_DAOD550.fill_value       = np.nan
    MetopC_IASI_LMD_DAOD550.fill_value       = np.nan
    MetopC_IASI_MARIP_DAOD550.fill_value     = np.nan 
    MIDAS_DOD550_mean.fill_value             = np.nan
    MIDAS_DOD550_std.fill_value              = np.nan
    
    # nc. saving datasets           
    lats[:]                      = latitude
    lons[:]                      = longitude
    Height[:]                    = ERA_Height
    U[:]                         = ERA_U
    U_SD[:]                      = ERA_U_SD        
    V[:]                         = ERA_V
    V_SD[:]                      = ERA_V_SD
    
    MIDAS_DOD550_mean[:]         = MIDAS_DOD_MEAN
    MIDAS_DOD550_std[:]          = MIDAS_DOD_STD
    
    LIVAS_Altitude[:]            = Altitude
    LIVAS_PD_b_532nm[:]          = Final_PD_b532nm       
    LIVAS_PD_a_532nm[:]          = Final_PD_a532nm      
    LIVAS_PD_MC[:]               = Final_PD_MC          
    LIVAS_PD_b_532nm_unc[:]      = Final_PD_b532nm_unc   
    LIVAS_PD_a_532nm_unc[:]      = Final_PD_a532nm_unc   
    LIVAS_PD_MC_unc[:]           = Final_PD_MC_unc       
    LIVAS_PD_b_532nm_SD[:]       = Final_PD_b532nm_SD   
    LIVAS_PD_a_532nm_SD[:]       = Final_PD_a532nm_SD   
    LIVAS_PD_MC_SD[:]            = Final_PD_MC_SD       
    LIVAS_N_of_CF_Pro[:]         = Final_Number_of_L2Profiles  
    LIVAS_N_of_Ocerpasses[:]     = Final_Number_of_Overpasses  
    LIVAS_DOD_532nm_mean[:]      = Final_DOD_532nm_mean   
    LIVAS_DOD_532nm_SD[:]        = Final_DOD_532nm_SD    
    LIVAS_DOD_532nm_unc[:]       = Final_DOD_532nm_unc
    LIVAS_Final_Marine[:]        = Final_Marine
    LIVAS_Final_Dust[:]          = Final_Dust
    LIVAS_Final_Polluted_Continental_Smoke[:] = Final_Polluted_Continental_Smoke
    LIVAS_Final_Clean_Continental[:] = Final_Clean_Continental
    LIVAS_Final_Polluted_Dust[:]     = Final_Polluted_Dust
    LIVAS_Final_Elevated_Smoke[:]    = Final_Elevated_Smoke
    LIVAS_Final_Dusty_Marine[:]      = Final_Dusty_Marine
    
    MetopA_IASI_ENS_DAOD550[:]       = MetopA_ENS_DAOD550
    MetopA_IASI_ENS_DAOD550_unc[:]   = MetopA_ENS_DAOD550_unc
    MetopA_IASI_IMARS_DAOD550[:]     = MetopA_IMARS_DAOD550
    MetopA_IASI_IMARS_DAOD550_unc[:] = MetopA_IMARS_DAOD550_unc
    MetopA_IASI_ULB_DAOD550[:]       = MetopA_ULB_DAOD550
    MetopA_IASI_LMD_DAOD550[:]       = MetopA_LMD_DAOD550
    MetopA_IASI_MARIP_DAOD550[:]     = MetopA_MARIP_DAOD550

    MetopC_IASI_ENS_DAOD550[:]       = MetopC_ENS_DAOD550
    MetopC_IASI_ENS_DAOD550_unc[:]   = MetopC_ENS_DAOD550_unc
    MetopC_IASI_IMARS_DAOD550[:]     = MetopC_IMARS_DAOD550
    MetopC_IASI_IMARS_DAOD550_unc[:] = MetopC_IMARS_DAOD550_unc
    MetopC_IASI_ULB_DAOD550[:]       = MetopC_ULB_DAOD550
    MetopC_IASI_LMD_DAOD550[:]       = MetopC_LMD_DAOD550
    MetopC_IASI_MARIP_DAOD550[:]     = MetopC_MARIP_DAOD550    

    Land_Ocean_Mask_id[:]         = Land_Ocean_Mask

    ds.close()               
    
    # print("End of: " + temp_time + '-ERA_uvw-MONTHLY.nc')
        
# How much time it took to execute the script in minutes:
print(datetime.now() - startTime)  
            
            