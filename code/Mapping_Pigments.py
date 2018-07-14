
# coding: utf-8

# In[1]:


import h5py
import csv
import numpy as np
import gdal
import matplotlib.pyplot as plt
from math import floor
from math import ceil


# In[2]:


def plot_aop_refl(band_array,refl_extent,colorlimit=(0,1),ax=plt.gca(),title='',cbar ='on',cmap_title='',colormap='Greys'):
    
    ''' read in and plot a single band or 3 stacked bands of a reflectance array
    --------
    Parameters
    --------
        band_array: ndarray
            Array of reflectance values, created from aop_h5refl2array
            If 'band_array' is a 2-D array, plots intensity of values
            If 'band_array' is a 3-D array (3 bands), plots RGB image, set cbar to 'off' and don't need to specify colormap 
        refl_extent: tuple
            Extent of reflectance data to be plotted (xMin, xMax, yMin, yMax) 
            Stored in metadata['spatial extent'] from aop_h5refl2array function
        colorlimit: tuple, optional
            Range of values to plot (min,max). 
            Look at the histogram of reflectance values before plotting to determine colorlimit.
        ax: axis handle, optional
            Axis to plot on; specify if making figure with subplots. Default is current axis, plt.gca()
        title: string, optional
            plot title 
        cbar: string, optional
            Use cbar = 'on' (default), shows colorbar; use if plotting 1-band array
            If cbar = 'off' (or not 'on'), does no
        cmap_title: string, optional
            colorbar title (eg. 'reflectance', etc.)
        colormap: string, optional
            Matplotlib colormap to plot 
            see https://matplotlib.org/examples/color/colormaps_reference.html

    Returns 
    --------
        plots flightline array of single band of reflectance data
    --------

    Examples:
    --------
    >>> plot_aop_refl(sercb56,
              sercMetadata['spatial extent'],
              colorlimit=(0,0.3),
              title='SERC Band 56 Reflectance',
              cmap_title='Reflectance',
              colormap='Greys_r') '''
    
    plot = plt.imshow(band_array,extent=refl_extent,clim=colorlimit); 
    if cbar == 'on':
        cbar = plt.colorbar(plot,aspect=40); plt.set_cmap(colormap); 
        cbar.set_label(cmap_title,rotation=90,labelpad=20)
    plt.title(title); ax = plt.gca(); 
    ax.ticklabel_format(useOffset=False, style='plain'); #do not use scientific notation for ticklabels
    rotatexlabels = plt.setp(ax.get_xticklabels(),rotation=90); #rotate x tick labels 90 degrees


# In[3]:


def h5refl2array(h5_filename):
    hdf5_file = h5py.File(h5_filename,'r')

    #Get the site name
    file_attrs_string = str(list(hdf5_file.items()))
    file_attrs_string_split = file_attrs_string.split("'")
    sitename = file_attrs_string_split[1]
    refl = hdf5_file[sitename]['Reflectance']
    reflArray = refl['Reflectance_Data']
    refl_shape = reflArray.shape
    wavelengths = refl['Metadata']['Spectral_Data']['Wavelength']
    #Create dictionary containing relevant metadata information
    metadata = {}
    metadata['shape'] = reflArray.shape
    metadata['mapInfo'] = refl['Metadata']['Coordinate_System']['Map_Info']
    #Extract no data value & set no data value to NaN\n",
    metadata['scaleFactor'] = float(reflArray.attrs['Scale_Factor'])
    metadata['noDataVal'] = float(reflArray.attrs['Data_Ignore_Value'])
    metadata['bad_band_window1'] = (refl.attrs['Band_Window_1_Nanometers'])
    metadata['bad_band_window2'] = (refl.attrs['Band_Window_2_Nanometers'])
    metadata['projection'] = refl['Metadata']['Coordinate_System']['Proj4'].value
    metadata['EPSG'] = int(refl['Metadata']['Coordinate_System']['EPSG Code'].value)
    mapInfo = refl['Metadata']['Coordinate_System']['Map_Info'].value
    mapInfo_string = str(mapInfo); #print('Map Info:',mapInfo_string)\n",
    mapInfo_split = mapInfo_string.split(",")
    #Extract the resolution & convert to floating decimal number
    metadata['res'] = {}
    metadata['res']['pixelWidth'] = mapInfo_split[5]
    metadata['res']['pixelHeight'] = mapInfo_split[6]
    #Extract the upper left-hand corner coordinates from mapInfo\n",
    xMin = float(mapInfo_split[3]) #convert from string to floating point number\n",
    yMax = float(mapInfo_split[4])
    #Calculate the xMax and yMin values from the dimensions\n",
    xMax = xMin + (refl_shape[1]*float(metadata['res']['pixelWidth'])) #xMax = left edge + (# of columns * resolution)\n",
    yMin = yMax - (refl_shape[0]*float(metadata['res']['pixelHeight'])) #yMin = top edge - (# of rows * resolution)\n",
    metadata['extent'] = (xMin,xMax,yMin,yMax),
    metadata['ext_dict'] = {}
    metadata['ext_dict']['xMin'] = xMin
    metadata['ext_dict']['xMax'] = xMax
    metadata['ext_dict']['yMin'] = yMin
    metadata['ext_dict']['yMax'] = yMax
    hdf5_file.close        
    return reflArray, metadata, wavelengths


# In[4]:


def fixed_refl (refl_array, waves, meta, choose_wave):
    
    index = np.where(np.abs(waves.value - choose_wave) < 2.5)[0]
    choose_refl = refl_array[:,:,index[0]].astype(float)/meta['scaleFactor']
    bad_cells = np.where(choose_refl == meta['noDataVal'])
    choose_refl[bad_cells] = np.nan
    return choose_refl


# In[5]:


h5_filename = './dela_tiles/NEON_D08_DELA_DP3_425000_3600000_reflectance.h5' #input


# In[6]:


#Need h5reflarray from workshop
#Need plot_aop_refl
#Need fixed_refl 
def pigment_in_h2o (file_name,pig_type):
        
    #Read in h5 tile
    [reflArray,metadata,wavelengths] = h5refl2array(h5_filename) #inside
    
    #Extract and clean wavelenghts
    refl_num = fixed_refl(refl_array=reflArray, waves=wavelengths, meta=metadata, choose_wave=709)
    refl_chla = fixed_refl(refl_array=reflArray, waves=wavelengths, meta=metadata, choose_wave=665)
    refl_H2O = fixed_refl(refl_array=reflArray, waves=wavelengths, meta=metadata, choose_wave=778)
    
    #Calculating chlorophyll reflectance & concentration from Randolph et al. (2008)
    b_corr = np.divide(2.71*0.60*refl_H2O,0.082 - 0.60*refl_H2O) #Correction factor
    spectral_index_chla = np.divide(refl_num,refl_chla)
    abs_chla = np.divide(spectral_index_chla*(0.727 + b_corr) - b_corr - 0.401,0.68)
    
    #If chlorophyll  
    if pig_type == 'chla':
        refl_pig = refl_chla
        spectral_index_pig = spectral_index_chla
        abs_pig = abs_chla
        conc_pig = abs_pig/0.0153
        conc_pig[np.isinf(conc_pig)] = np.nan
        conc_pig[conc_pig < 0] = 0
        plot_aop_refl(band_array = conc_pig, 
              refl_extent = metadata['extent'][0], title='Chlorophyll a Concentration mg/sq.m',
              colorlimit = (0,200),
              cmap_title = '[Chlorophyll a]',        
              colormap = 'Greens')
    else:
        if  pig_type == 'phyco':
            refl_pig = fixed_refl(refl_array=reflArray, waves=wavelengths, meta=metadata, choose_wave=620)
            spectral_index_pig = np.divide(refl_num,refl_pig) 
            abs_pig = np.divide(spectral_index_pig*(0.720 + b_corr) - b_corr - 0.281,0.84) - 0.24*abs_chla
            conc_pig = abs_pig/0.0070
            conc_pig[np.isinf(conc_pig)] = np.nan
            conc_pig[conc_pig < 0] = 0
            plot_aop_refl(band_array = conc_pig, 
              refl_extent = metadata['extent'][0], title='Phycocyanin a Concentration mg/sq.m',
              colorlimit = (0,200),
              cmap_title = '[Phycocyanin]',        
              colormap = 'cool')
        else: 
            return print("Can't do that pigment!")
            
        
 


# In[7]:


pigment_in_h2o(h5_filename,'chla')


# In[8]:


pigment_in_h2o(h5_filename,'phyco')

