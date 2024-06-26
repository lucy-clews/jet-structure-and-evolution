#!/usr/bin/env python
# coding: utf-8

from astropy.io import fits
from astropy import table
from astropy.wcs import WCS,utils
from astropy.coordinates import SkyCoord
import astropy.units as u

import numpy as np
import math as math
import matplotlib.pyplot as plt
import glob as glob

from scipy import interpolate
from scipy.interpolate import splrep, BSpline,splprep,splev, make_interp_spline, CubicSpline
from scipy import interpolate as itp
import scipy.integrate as integrate

from ridge_toolkitDR2 import GetCutoutArray
from SourceSearchDR2 import GetPointList, ClosestPoints, PointOfIntersection, LengthOnLine
import RLConstantsDR2 as RLC



def get_nearest_match(array, value):
    
    array =  np.asarray(array)
    indx = (np.abs(array-value)).argmin()
    
    return array[indx]

def spline(x,y): #spline function
  
    p = np.array([x,y])
    a= np.arange(len(x))
    z= np.linspace(0, len(x)-1, 200)
    
    spl =make_interp_spline(a, p.T, bc_type="natural")
    x_new,y_new = spl(z).T
    
    print('SB spline calculated successfully.')
    return x_new, y_new

def spline_u(ra,dec): #parametric spline function
    
    length_on_spline = [0]
    u = np.arange(len(ra))
  
    #cord (a.k.a cumulative) length
    p = np.array([ra,dec])
    dp = p[:, 1:] - p[:, :-1]
    l = (dp**2).sum(axis=0)
    u_cord = np.sqrt(l).cumsum()
    u_cord = np.r_[0, u_cord] #np.r_ Translates slice objects to join arrays along the first axis.
    
   
    u_c = np.linspace(0, u_cord.max(), 200)
    u_test = np.linspace(0, len(ra)-1, 200) #parametric co-ords, evaluated on fine grid
    
    spline = make_interp_spline(u, p.T, bc_type='natural') #.T transposes result to unpack into a pair of x, y coord
    xnew_u, ynew_u = spline(u_test).T
    
    #derivatives
    dx_du, dy_du = spline.derivative(nu=1)(u_test).T
    d2x_du2, d2y_du2 = spline.derivative(nu=2)(u_test).T
    
    print('Successfully calculated ridge spline.')
    
    #distance along spline
    
    distance=0
    
    for j in range(len(xnew_u)-1):
     
        distance += np.sqrt((xnew_u[j+1]-xnew_u[j])**2+(ynew_u[j+1]-ynew_u[j])**2)
        length_on_spline.append(distance)
        
    #host position on spline
    
    print('Calculated distance along spline')

    return xnew_u, ynew_u, dx_du, dy_du, d2x_du2, d2y_du2, length_on_spline

    
def curvature(dx_du, dy_du, d2x_du2, d2y_du2):
    
    fig, axs = plt.subplots(nrows=2,ncols=1, figsize=(10,8))
    axs[0].plot(dx_du,dy_du)
    axs[0].set_title('First derivative w.r.t u')
    axs[1].plot(d2x_du2,d2y_du2)
    axs[1].set_title('Second derivative w.r.t u')
                  
    x_dot = dx_du
    x_doubledot = d2x_du2
    
    y_dot = dy_du
    y_doubledot = d2y_du2
   
    # k= x'y''-y'x''/(x'^2+y'^2)^3/2
    kappa = (x_dot*y_doubledot - y_dot*x_doubledot)/((x_dot**2+y_dot**2)**1.5)
   
    print('Calculated curvature.')
    return kappa


def get_ridge_txt_files(source_name, path):
    
    file1=glob.glob(path+'/ridges/%s_ridge1.txt' %source_name)
    file2=glob.glob(path+'/ridges/%s_ridge2.txt' %source_name)
    
    return file1, file2

def get_rms4cutout(source_name, path):
    
    rms4cutout=glob.glob(path+'/rms4_cutouts/%s-cutout.npy' %source_name)
    print('Got rms4cutout.')
    return rms4cutout
    
def get_fits_cutout(source_name, path):
    
    fits_cutout = glob.glob(path+'/fits_cutouts/%s-cutout.fits' %source_name)
    print('Got fits cutout.')
    return fits_cutout

def find_host_position(source_name, hostx_deg, w,sorted_points,sorted_lengths, path):
    
    hostlength=0
    totallength=0
    ArcHost=0
    Ix=0
    Iy=0
    nearest=[]
    
    L_dist = glob.glob(path+'/Distances/Ldistances-%s.txt' %source_name)
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
            c1,c2, c1index, c2index = ClosestPoints(sorted_points,opt_poss)
            posindex = min(c1index, c2index)
            nearest= sorted_points[posindex]
            Ix,Iy= PointOfIntersection(c1,c2,opt_poss)
            hostlength, totallength = LengthOnLine(sorted_lengths, sorted_points, posindex, Ix, Iy)
            ArcHost = np.asarray(hostlength) * RLC.ddel * 3600 #position of host rel to ridgeline
            print('Calculated Host Position.')
            
    return hostlength, totallength, Ix,Iy

def host_spline(Ix,Iy,xnew_u,ynew_u,length_on_spline):
    
    spline_points=[]
    host_ridge = np.array([Ix,Iy])
   
    spline_points=[[float(xnew_u[i]),float(ynew_u[i])] for i in range(len(xnew_u))]
    c1,c2,c1index,c2index = ClosestPoints(spline_points,host_ridge)
    posindex = min(c1index,c2index)
    hostspline_x,hostspline_y = PointOfIntersection(c1,c2,host_ridge)
    
    hostlength_spline = LengthOnLine(length_on_spline,spline_points,posindex,hostspline_x,hostspline_y)[0]
    
    return hostspline_x, hostspline_y, hostlength_spline

def SB(source_name, pixels,sorted_points,sorted_lengths):

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
        xc = sorted_points[i][0]
        yc = sorted_points[i][1]
        val = array[y, x]
        total = 1000*(RLC.KerW * arraynum[y,x]) + arraynum[y, x - 1] + arraynum[y, x + 1] + arraynum[y - 1, x] + arraynum[y + 1, x]
        avg = (total/(RLC.KerW + 4))
        values.append(val)
        average.append(avg)
        x_pixels.append(x)
        y_pixels.append(y)
        x_coords.append(xc)
        y_coords.append(yc)
    for j in range(len(sorted_lengths)-1):
        dis += (abs(sorted_lengths[j+1] - sorted_lengths[j])) #* RLC.ddel * 3600 conv to arcsec  
        distance.append(dis)
        
    for k in range(len(values)):
        diff = abs(values[k] - average[k])
        difference.append(diff)
    
    print('Deduced SB profile.')
    return distance,average, difference
