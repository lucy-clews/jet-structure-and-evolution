#!/usr/bin/env python
# coding: utf-8

from astropy.io import fits
import astropy.io.fits as pyfits
from astropy import table
import numpy as np
import glob as glob
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
import csv
import pandas as pd
path= '/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2'

source_names_with_duplicates=[]
source_names=[]
tupels=[]

for file in glob.glob(path+'/SB_csv/*-SB.csv'):
    source_names.append(os.path.basename(file).split('-')[0])
        

print('Number of sources matched=',len(source_names))



#missing sources
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
        
t_1.write('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/Data/AGNs_with_ridges_and_curvature_info.fits', overwrite=True)
print('Success catalogue made')


#generate table of failed sources
t_2 = table.Table(catalogue[0:0])
name = catalogue['Source_Name_1']

for x in missed:
    for row in catalogue[name==x]:
        t_2.add_row(row)
    
t_2.write('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/Data/failed_curvature_sources.fits', overwrite=True)
print('failed catalogue made')


#exploring properties of successful sources
with fits.open('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/Data/AGNs_with_ridges_and_curvature_info.fits') as data:
    success_cat = table.Table(data[1].data)

redshift=[]
luminosity=[]
physical_size=[]
angular_size=[]
points=[]
greater_than_5_points=[]

for row in success_cat:
    
    luminosity.append(row['L_144']) #W/Hz
    redshift.append(row['z_best'])
    physical_size.append(row['Size']) #kpc
    angular_size.append(row['LAS']) #arcsec
    name = row['Source_Name_1']
    
    SB_csv = pd.read_csv(path+'/SB_csv/%s-SB.csv' %name)
    number_points = SB_csv['Average SB (mJy/beam)']
    points.append(number_points)
    print(len(number_points))
    if len(number_points)>= 5:
        greater_than_5_points.append(True)
    else:
        greater_than_5_points.append(False)
        
success_cat.add_column(greater_than_5_points, name='>5_points')
success_cat.write('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/Data/AGNs_with_ridges_and_curvature_info.fits', overwrite=True)

print('Quality boolean added')

