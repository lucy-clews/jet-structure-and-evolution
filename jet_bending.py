#!/usr/bin/env python
# coding: utf-8

#import modules

from astropy.io import fits
from astropy import table
from astropy.table import QTable, Table
from astropy.wcs import WCS, utils
from astropy.coordinates import SkyCoord
import astropy.units as u

import pandas as pd

from scipy import interpolate
from scipy.interpolate import splrep, BSpline

import sys
import numpy as np
import matplotlib.pyplot as plt

import glob as glob
import os
import pathlib
import seaborn as sns
sns.set()

#import functions

from bending_setup import get_rms4cutout, get_ridge_txt_files,spline, get_fits_cutout, find_host_position, SB,spline_u, curvature, get_nearest_match
from SourceSearchDR2 import GetPointList
from ridge_toolkitDR2 import GetCutoutArray


#create empty arrays to append to later
fails = []
hp = []
names=[]
source_in_hp = []


#get data
with fits.open('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/AGNs_with_ridge.fits') as data:
    catalogue = table.Table(data[1].data)



#make list of all hp directories

for directory in glob.glob('/beegfs/lofar/mjh/rgz/*/ridgeline/**/'):
    hp.append(directory)

number_of_hp = len(hp)
print('There are', number_of_hp, 'healpix directories')
np.savetxt('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/hp_directories.txt', hp, fmt='%s',delimiter=' ')


batch_number = sys.argv[1]
path = hp[batch_number] #directory for this job

for x in glob.glob(path+'/ridges/*.txt'):
    names.append(os.path.basename(x).split('_')[0])
        

for name in names:
    
    for i in catalogue[0:30]:
        
        source_name = i['Source_Name_1']      
        
        if source_name==name:
            
            source_in_hp.append(source_name, dtype=object)
            
            LOFARx =float(i['RA_1']) 
            LOFARy = float(i['DEC_1'])
            hostx_deg= float(i['ID_RA'])
            hosty_deg=float(i['ID_DEC'])
            L_144 = float(i['L_144'])
            redshift = float(i['z_best'])
            rms = float(i['Isl_rms'])

            try:
                file1,file2 = get_ridge_txt_files(source_name, path)
                file1 = open(file1[0])
                file2 = open(file2[0])
            except:
                print('WARNING: No txt file for source', source_name)
                fails.append(source_name)

            else:
                points, lengths = GetPointList(file1, file2)
                ra=[]
                dec=[]

            pixels = np.round(points).astype(int)

            #get optical host position

            fits_cutout = get_fits_cutout(source_name, path)
            hdu = fits.open(fits_cutout[0])
            w = WCS(hdu[0].header)
            hostlength, ArcHost, Ix,Iy = find_host_position(source_name, hostx_deg, w,points,lengths, path)

            d={'Host_x (deg)':[hostx_deg], 'Host_y (deg)':[hosty_deg],'Ix (pix)':[Ix], 'Iy (pix)':[Iy], 'Hostlength (pix)':[hostlength], 'ArcHost (arcsec)':[ArcHost]}
            column_values=['Host_x (deg)','Host_y (deg)','Ix (pix)','Iy (pix)','Hostlength (pix)', 'ArcHost (arcsec)']
            df =pd.DataFrame(data=d, columns=column_values)
            df.to_csv('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/host_positions/%s-host_info.csv' %source_name)



            #SB


            distance, average, difference = SB(source_name, pixels,points,lengths) #SB profile

            d={'Distance along ridge (arcsec)':distance, 'Average SB (mJy/beam)':average, 'ArcHost (arcsec)':ArcHost}
            column_values=['Distance along ridge (arcsec)','Average SB (mJy/beam)','ArcHost (arcsec)']
            index_values = np.linspace(0, len(distance)-1, len(distance))
            df =pd.DataFrame(data=d, columns=column_values,index=index_values)
            df.to_csv('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/SB_csv/%s-SB.csv' %source_name)

            SB_splinex, SB_spliney = spline(distance,average) #SB spline

            d={'Spline_x (mJy/beam)':SB_splinex, 'Spline_y':SB_spliney}
            column_values=['Spline_x (mJy/beam)','Spline_y']
            index_values = np.linspace(0, len(SB_splinex)-1, len(SB_splinex))
            df =pd.DataFrame(data=d, columns=column_values,index=index_values)
            df.to_csv('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/SB_spline_csv/%s-SB_spline.csv' %source_name)

            plt.figure(figsize = (10,8))
            plt.plot(distance, average, ls = '-',color='b', label = 'Surface Brightness - Average Pixels')
            plt.plot(SB_splinex,SB_spliney, '--', color='k', label='Spline')
            plt.errorbar(distance, average, yerr=rms, xerr=None, color='b')
            plt.axvline(ArcHost, color = 'r', ls = ':', label = 'Host Position')

            plt.title(source_name)
            plt.xlabel('Distance Along Ridgeline (arcseconds)')
            plt.ylabel('Average Surface Brightness (mJy/beam)')
            plt.legend() 
            plt.savefig('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/SB_profiles/Profile-%s.png' %source_name)

            #ridge spline

            for index, item in enumerate(points):

                x = item[0]
                y=item[1]

                ra.append(x)
                dec.append(y)

            xnew_u, ynew_u,x_cord,y_cord, dx_du, dy_du, d2x_du2, d2y_du2, u_test,u_c= spline_u(ra,dec)
            host_cord= get_nearest_match(u_c, hostlength) #length along spline of host position 


            data={'Spline_x (pix)':xnew_u,'Spline_y (pix)':ynew_u,'dx_du':dx_du,'dy_du':dy_du,'d2x_du2':d2x_du2,'d2y_du2':d2y_du2,'u_c':u_c}
            column_values=['Spline_x (pix)', 'Spline_y (pix)','dx_du','dy_du','d2x_du2','d2y_du2','u_c']
            index_values = np.linspace(0, 199,200)
            df =pd.DataFrame(data=data,columns=column_values,index=index_values)
            df.to_csv('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/ridge_spline_csv/%s-ridge_spline.csv' %source_name, )


            # curvature 
            kappa = curvature(dx_du, dy_du, d2x_du2, d2y_du2)
            np.savetxt('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/curvature_txt/%s-curvature.txt' %source_name, kappa, delimiter=' ')

            plt.figure(figsize=(10,8))
            plt.plot(u_c[:-1],kappa,'-', color='k', label='Curvature')
            plt.axvline(host_cord, ls='--', color='b', label='Host Position')
            plt.title(source_name)
            plt.xlabel('Length along the ridgeline (pix)')
            plt.ylabel('$\kappa$')
            plt.savefig('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/curvature_plots/%s-curavture.png' %(source_name))

            #curvature/spline plot
            fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,8))
            ax2.set_xlabel('Length along the ridgeline (pix)')
            ax2.set_ylabel('$\kappa$')
            ax1.set_xlabel('RA (pix)')
            ax1.set_ylabel('DEC (pix)')
            ax1.plot(Ix,Iy, 'x', color='b', label='Optical Host')
            plt.suptitle(source_name)
            ax2.plot(u_c[:-1],kappa,'-', color='k', label='Curvature')
            ax2.axvline(host_cord, ls='--', color='b', label='Host Position')
            ax2.legend()
            ax1.plot(ra,dec, '-', color='r', label='Ridgeline') #plot ridge
            ax1.plot(xnew_u,ynew_u,'--',label='Spline', color='k') #plot spline
            ax1.legend()
            plt.savefig('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/ridge_curvature_plots(no_mod)/%s-curvature_and_ridge.png' %(source_name))

np.savetxt('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/source_in_hp/batch_%s-sources_with_ridge.txt' %batch_number, source_in_hp, fmt='%s', delimiter=' ')

