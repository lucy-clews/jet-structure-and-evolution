#!/usr/bin/env python
# coding: utf-8

# Code to check for missing sources

# In[1]:


import numpy as np
import glob as glob


# In[2]:


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
print('source_names format=', source_names[1])

# In[3]:


#missing sources
completed=[]
missed=[]
full_list = open('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/source_with_ridge_names.txt')
all_names = full_list.readlines()
cleaned_all=[elem.strip() for elem in all_names]

a = set(cleaned_all)
b = set(source_names)
print(a, b)
missed = a.difference(b)
missed = list(missed)
print(len(missed))

#print(cleaned_missed)
np.savetxt('/beegfs/lofar/lclews/DR2_ridgelines/full_sample_2/failed_sources.txt',missed, fmt='%s', delimiter=' ')



# In[ ]:




