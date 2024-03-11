#!/usr/bin/env python
# coding: utf-8


from astropy.io import fits
from astropy import table
import numpy as np
import glob as glob
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

path= '/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/source_in_hp'
source_names_with_duplicates=[]
source_names=[]
tupels=[]

for file in glob.glob(path+'/*.txt'):
    txt_file = open(file)
    content = txt_file.read()
    lines= content.splitlines()
    tupels.append(lines)

for i in tupels:
    for x in i:
        source_names_with_duplicates.append(x)
i=0
while i< len(source_names_with_duplicates):
    if source_names_with_duplicates[i] not in source_names:
        source_names.append(source_names_with_duplicates[i])
    i +=1
        

print('Number of sources matched=',len(source_names))


#looking into failed/missing sources
completed=[]
missed=[]
full_list = open('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/Data/source_with_ridge_names.txt')
all_names = full_list.readlines()
cleaned_all=[elem.strip() for elem in all_names]

a = set(cleaned_all)
b = set(source_names)
missed = a.difference(b)
missed = list(missed)

common = a.intersection(b)
common= list(common)

np.savetxt('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/Data/failed_sources.txt',missed, fmt='%s', delimiter=' ')
np.savetxt('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/Data/names_w_ridge+curvature.txt', common, fmt='%s', delimiter=' ')


# generate table with sources that have output
with fits.open('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/Data/AGNs_with_ridge.fits') as data:
    catalogue = table.Table(data[1].data)
t_1 = table.Table(catalogue[0:0])
name = catalogue['Source_Name_1']

for i in common:
    for row in catalogue[name==i]:
        t_1.add_row(row)
        
t_1.write('AGNs_with_ridges_and_curvature_info.fits')


#generate table of failed sources
t_2 = table.Table(catalogue[0:0])
name = catalogue['Source_Name_1']

for x in missed:
    for row in catalogue[name==x]:
        t_2.add_row(row)
        
t_2.write('failed_curvature_sources.fits')

#exploring properties of failed sources
with fits.open('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/Data/failed_curvature_sources.fits') as data:
    failed_cat = table.Table(data[1].data)
    
    
redshift=[]
luminosity=[]
physical_size=[]
angular_size=[]

for row in failed_cat:
    luminosity.append(row['L_144']) #W/Hz
    redshift.append(row['z_best'])
    physical_size.append(row['Size']) #kpc
    angular_size.append(row['LAS']) #arcsec

plt.figure(figsize=(10,8))
plt.hist(luminosity)
plt.xlabel('$L_{144}$')
plt.ylabel('Count')
plt.title('Hist of Luminosity for failed')

plt.figure(figsize=(10,8))
plt.hist(luminosity, bins=3000)
plt.xlabel('$L_{144}$')
plt.xlim(0, 1.5e26)
plt.ylabel('Count')
plt.title('Hist of Luminosity for failed')

plt.figure(figsize=(10,8))
plt.hist(redshift, bins=100)
plt.xlabel('z')
plt.ylabel('Count')
plt.title('Hist of redshift for failed')

plt.figure(figsize=(10,8))
plt.hist(angular_size, bins=100)
plt.xlabel('LAS (arcsec)')
plt.ylabel('Count')
plt.title('Hist of angular size for failed')

plt.figure(figsize=(10,8))
plt.hist(physical_size, bins=100)
plt.xlim(0,2500)
plt.xlabel('Size (kpc)')
plt.ylabel('Count')
plt.title('Hist of source size for failed')

#exploring properties of successful sources
with fits.open('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/Data/AGNs_with_ridges_and_curvature_info.fits') as data:
    success_cat = table.Table(data[1].data)

redshift=[]
luminosity=[]
physical_size=[]
angular_size=[]

for row in success_cat:
    luminosity.append(row['L_144']) #W/Hz
    redshift.append(row['z_best'])
    physical_size.append(row['Size']) #kpc
    angular_size.append(row['LAS']) #arcsec
    
plt.figure(figsize=(10,8))
plt.hist(luminosity, bins=100)
plt.xlabel('$L_{144}$')
plt.ylabel('Count')
plt.title('Hist of Luminosity for success')


plt.figure(figsize=(10,8))
plt.hist(redshift, bins=100)
plt.xlabel('z')
plt.ylabel('Count')
plt.title('Hist of redshift for success')

plt.figure(figsize=(10,8))
plt.hist(angular_size, bins=100)
plt.xlabel('LAS (arcsec)')
plt.ylabel('Count')
plt.title('Hist of angular size for success')

plt.figure(figsize=(10,8))
plt.hist(physical_size, bins=100)
plt.xlim(0,2500)
plt.xlabel('Size (kpc)')
plt.ylabel('Count')
plt.title('Hist of source size for success')


# In[ ]:




