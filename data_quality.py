#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import numpy as np

import glob as glob
import pandas as pd
import seaborn as sns
sns.set()


# In[2]:


path = '/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2'
SB_size = []
DR= [] 

for file in glob.glob(path + '/SB_csv/*.csv'):
    csv = pd.read_csv(file)
    SB = csv['Average SB (mJy/beam)']
    SB_size.append(len(SB)) #number of points in profile
    DR.append(np.max(SB)/np.mean(SB)) #dynamic range


# In[29]:


#plots to investigate data quality#

sns.color_palette('pastel')
plt.figure(figsize=(10,8))
plt.hist(SB_size, bins=50, color='m')
plt.xlabel('Number of points on ridgeline')
plt.ylabel('Count')
plt.savefig(path+'/Data/#_points_on_ridges.png')

plt.figure(figsize=(10,8))
plt.hist(DR, bins=50, color='b')
plt.xlabel('DR')
plt.axvline(2, color='r', linestyle='--',label='DR>2')
plt.ylabel('Count')
plt.legend()
plt.savefig(path+'/Data/DR_distribution.png')

good_DR = []
for x in DR:
    if x>2.0:
        good_DR.append(x)
        
print(len(good_DR), 'sources with DR>2')


# In[ ]:




