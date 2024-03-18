#!/usr/bin/env python
# coding: utf-8

# Testing results of 100 sources

# In[16]:


from astropy.io import fits
import astropy.io.fits as pyfits
from astropy import table
from astropy.utils.data import get_pkg_data_filename

import glob as glob 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[21]:


path = '/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/'

with fits.open(path+'/Data/AGNs_with_ridges_and_curvature_info.fits') as data:
    success_cat = table.Table(data[1].data)
count = 0
for i in success_cat:
    quality = i['>=5_ridge_points']
    print(quality)
    if [x==True for x in quality]:
        if count<=100:
            name = i['Source_Name_1']
            #gather images
            SB_profile = mpimg.imread(path+'SB_profiles/Profile-%s.png' %name)
            ridge_curve_plot = mpimg.imread(path+'ridge_curvature_plots/%s-curvature_and_ridge.png' %name)
            fits_image = get_pkg_data_filename(glob.glob('/beegfs/lofar/mjh/rgz/*/ridgeline/*/fits_cutouts/%s-cutout.fits' %name))
            hdu = fits.open(fits_image)[0]

            #plot images onto same figure
            plt.figure()
            plt.imshow(hdu.data,cmap='gray')
            plt.imshow(SB_profile)
            plt.imshow(ridge_curve_plot)

            #save figure as pdf
            plt.savefig(path+'pdf/%s.pdf' %name)

            count=count+1


# In[ ]:




