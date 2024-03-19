#!/usr/bin/env python
# coding: utf-8

# Testing results of 100 sources

from astropy.io import fits
import astropy.io.fits as pyfits
from astropy import table
from astropy.utils.data import get_pkg_data_filename

import glob as glob 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print('Generating pdfs for first 100 sources...')

path = '/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/'

with fits.open(path+'/Data/AGNs_with_ridges_and_curvature_info.fits') as data:
    success_cat = table.Table(data[1].data)
    
count = 0
for i in success_cat:
    
    name = i['Source_Name_1']
    quality = i['>=5_ridge_points']
    
    if quality==True:
        if count<=100:
            name = i['Source_Name_1']
            #gather images
            SB_profile = mpimg.imread(path+'SB_profiles/Profile-%s.png' %name)
            ridge_curve_plot = mpimg.imread(path+'ridge_curvature_plots/%s-curvature_and_ridge.png' %name)
            fits_cutout=glob.glob('/beegfs/lofar/mjh/rgz/*/ridgeline/*/fits_cutouts/%s-cutout.fits' %name)
            fits_data = get_pkg_data_filename(fits_cutout[0])
            fits_image=fits.getdata(fits_data, ext=0)
            
            #plot images onto same figure
            
            fig,axs = plt.subplots(nrows=3, ncols=1,figsize=(8.27,11.7))
            plt.suptitle(name)
            axs[0].imshow(fits_image,cmap='gray', interpolation='none')
            axs[0].axis('off')
            axs[1].imshow(SB_profile,interpolation='none')
            axs[1].axis('off')
            axs[2].imshow(ridge_curve_plot, interpolation='none')
            axs[2].axis('off')

            #save figure as pdf
            plt.savefig(path+'pdf/%s.pdf' %name)

            print(count, 'pdfs generated...', (count/100)*100,'% complete')
            count=count+1
            
print('100 pdfs generated successfully')
