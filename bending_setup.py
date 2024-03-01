#!/usr/bin/env python
# coding: utf-8

# In[1]:


from astropy.io import fits
from astropy import table
from astropy.wcs import WCS,utils
from astropy.coordinates import SkyCoord
import astropy.units as u

import numpy as np
import math as math
import matplotlib.pyplot as plt
from heapq import nsmallest, nlargest
from statistics import median
from collections import Counter
from scipy import interpolate
from scipy.interpolate import splrep, BSpline,splprep,splev, make_interp_spline, CubicSpline
from scipy import interpolate as itp
from scipy.stats import chisquare

from ridge_toolkitDR2 import GetCutoutArray
from SourceSearchDR2 import GetPointList, ClosestPoints, PointOfIntersection, LengthOnLine
import RLConstantsDR2 as RLC

import glob as glob


def countlist(array): #function to count number of trues in an array
    return sum(bool(x) for x in array)


def get_nearest_match(array, value):
    
    array =  np.asarray(array)
    indx = (np.abs(array-value)).argmin()
    print('value=',value)
    print('match=',array[indx])
    
    return array[indx]

def spline_u(ra,dec): #parametric spline function
    
    u = np.arange(len(ra))
    
    #cord (a.k.a cumulative) length
    p = np.array([ra,dec])
    dp = p[:, 1:] - p[:, :-1]
    l = (dp**2).sum(axis=0)
    u_cord = np.sqrt(l).cumsum()
    u_cord = np.r_[0, u_cord] #np.r_ Translates slice objects to join arrays along the first axis.
    #u_c = np.r_[0, u_cord]
  
    u_c = np.linspace(0, u_cord.max(), 200)
   # print(u_cord)
    #print(u_c)
    u_test = np.linspace(0, len(ra)-1, 200) #parametric co-ords, evaluated on fine grid
   #fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    #parametrizations = ['uniform', 'cord length', 'centripetal']
   
    spline = make_interp_spline(u, p.T, bc_type="natural") #.T transposes result to unpack into a pair of x, y coord
    xnew_u, ynew_u = spline(u_test).T 
    
    cord_spline = make_interp_spline(u_cord, p.T,bc_type="natural")
    x_cord, y_cord = cord_spline(u_c).T
    
    #derivatives
    dx_du, dy_du = spline.derivative()(u_test).T
    d2x_du2, d2y_du2 = spline.derivative().derivative()(u_test).T
    
    return xnew_u, ynew_u,x_cord,y_cord,dx_du, dy_du, d2x_du2, d2y_du2, u_test,u_c


def curvature(dx_du, dy_du, d2x_du2, d2y_du2):
   
    kappa = []
    
    f_prime=(dy_du/dx_du)
    f_prime_prime=(d2y_du2/d2x_du2)

    for index_1, item in enumerate(f_prime[:-1]):
        
        curve = f_prime_prime[index_1]/((1+item**2)**1.5)
        kappa.append(curve)
    
    return kappa


def get_ridge_txt_files(source_name):
    
    file1=glob.glob('/beegfs/lofar/mjh/rgz/*/ridgeline/hp_*/ridges/%s_ridge1.txt' %source_name)
    file2=glob.glob('/beegfs/lofar/mjh/rgz/*/ridgeline/hp_*/ridges/%s_ridge2.txt' %source_name)
    
    return file1, file2

def get_rms4cutout(source_name):
    
    rms4cutout=glob.glob('/beegfs/lofar/mjh/rgz/*/ridgeline/hp_*/rms4_cutouts/%s-cutout.npy' %source_name)
    return rms4cutout
    
def get_fits_cutout(source_name):
    
    fits_cutout = glob.glob('/beegfs/lofar/mjh/rgz/*/ridgeline/hp_*/fits_cutouts/%s-cutout.fits' %source_name)
   
    return fits_cutout

def find_host_position(source_name, hostx_deg, w,points,lengths):
    
    L_dist = glob.glob('/beegfs/lofar/mjh/rgz/*/ridgeline/hp_*/Distances/Ldistances-%s.txt' %source_name)
    L_dist=open(L_dist[0])
    for row in L_dist: #determine optical host position in pixels
                
                    row_info =row.strip()
                    row_cols = row_info.split(',')
                    poss_RA = float(row_cols[3])
                    poss_DEC = float(row_cols[4])
                    
                    if poss_RA == hostx_deg:
                     
                        poss_coord= SkyCoord(poss_RA*u.degree, poss_DEC*u.degree, frame='fk5')
                        poss_pix = utils.skycoord_to_pixel(poss_coord, w,origin=0)
                        opt_poss=np.array([poss_pix[0], poss_pix[1]])
                        c1,c2, c1index, c2index = ClosestPoints(points,opt_poss)
                        posindex = min(c1index, c2index)
                        Ix,Iy= PointOfIntersection(c1,c2,opt_poss)
                        hostlength = LengthOnLine(lengths, points, posindex, Ix, Iy)[0] 
                        ArcHost = hostlength * RLC.ddel * 3600 #position of host rel to ridgeline
                       
                        
   
    return hostlength, ArcHost, Ix,Iy

def SB(source_name, pixels,points,lengths):

    values = []
    average = []
    difference = []
    x_pixels = []
    x_coords = []
    y_pixels = []
    y_coords = []
    distance = [0]
    dis = 0

    array = GetCutoutArray(source_name)
    arraynum = np.nan_to_num(array)

    for i in range(np.shape(pixels)[0]):
        x = pixels[i][0]
        y = pixels[i][1]
        xc = points[i][0]
        yc = points[i][1]
        val = array[y, x]
        total = 1000*(RLC.KerW * arraynum[y,x]) + arraynum[y, x - 1] + arraynum[y, x + 1] + arraynum[y - 1, x] + arraynum[y + 1, x]
        avg = (total/(RLC.KerW + 4))
        values.append(val)
        average.append(avg)
        x_pixels.append(x)
        y_pixels.append(y)
        x_coords.append(xc)
        y_coords.append(yc)
    for j in range(len(lengths)-1):
        dis += (abs(lengths[j+1] - lengths[j])) * RLC.ddel * 3600 #conv to arcsec
        distance.append(dis)

    for k in range(len(values)):
        diff = abs(values[k] - average[k])
        difference.append(diff)
        
    return distance, average, difference

def spline(x,y):
   # print(x)
    p = np.array([x,y])
    a= np.arange(len(x))
    z= np.linspace(0, len(x), 200)
    
    spl =make_interp_spline(a, p.T, bc_type="natural")
    x_new,y_new = spl(z).T
   
    #print(x_new,y_new)
    return x_new, y_new