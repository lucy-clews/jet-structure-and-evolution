#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import modules
from astropy.io import fits
from astropy import table
from astropy.table import QTable
from astropy.wcs import WCS, utils
from astropy.coordinates import SkyCoord
import astropy.units as u
import seaborn as sns
sns.set()

import numpy as np
import matplotlib.pyplot as plt
from heapq import nsmallest, nlargest
from statistics import median
from collections import Counter
import pandas as pd 
import glob as glob


# In[2]:


#access AGN catalogue

source_names_list = np.genfromtxt('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/source_with_ridge_names.txt', dtype='str')

with fits.open('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/fcoz_classifications_wiseclass_absmag_nosf.fits') as data:
    #data.info()
    all_AGN_table = table.Table(data[1].data)
    print(all_AGN_table.columns)
    


# In[ ]:


#create new table: only AGNs with a ridge. (use later for curvature/SB stuff - will be quicker to run)

redshift=[]

t_1 = table.Table(all_AGN_table[0:0])
name = all_AGN_table['Source_Name_1']

for i in source_names_list:
    for row in all_AGN_table[name==i]:
        t_1.add_row(row)
        print(i)
        
t_1.write('AGNs_with_ridge.fits')


# In[ ]:


with fits.open('AGNs_with_ridge.fits') as data:
    #data.info()
    t_final = table.Table(data[1].data)


#find redshifts for ridge sources

redshift=[]

for row in t_final:
    redshift.append(i['z_source'])


# In[ ]:


plt.figure()
plt.hist(redshift)
plt.xlabel('$z$')
plt.ylabel('Frequency')
plt.title('Sample Properties')
plt.savefig('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/z-properties.png')


# In[ ]:




