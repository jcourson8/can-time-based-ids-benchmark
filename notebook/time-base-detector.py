#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import sys  

sys.path.insert(0, os.path.dirname(os.getcwd()) + "/code/")

import helper_functions
from importlib import reload

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.integrate import quad

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import json
import fnmatch
import scipy

from sklearn.metrics import auc
from sklearn.covariance import EllipticEnvelope


# ## Change Directory

# In[4]:


os.chdir(os.path.dirname(os.getcwd()) + "/data/ambient")
print(os.getcwd())


# In[5]:


# List the files in the 'data' folder
[file for file in os.listdir(".") if "dyno" in file]


# ## Load Training Data

# Aggregate ambient data for training in a single file. Here we considered only data generated in dyno conditions. 

# In[6]:


# df_aggregation = []

# for file_name in os.listdir("."):
    
#     if "dyno" in file_name:
        
#         print(file_name)
#         df = helper_functions.make_can_df(file_name)
#         df = helper_functions.add_time_diff_per_aid_col(df)
#         print(df.shape)
#         # print(df.dtypes)
             
#         df_aggregation.append(df)

# # Concatenate all training datasets on the dyno
# df_training = pd.concat(df_aggregation)
# df_training = df_training[["time", "aid", "time_diffs"]]
# display(df_training)


# ## Save Aggregated Training Set

# In[11]:


df_training.to_csv("aggregated_training_data.csv", index=False)


# ## Loading Aggregated Training Data

# In[13]:


df_training = pd.read_csv("aggregated_training_data.csv")
display(df_training)

# df_training = helper_functions.make_can_df("ambient_dyno_exercise_all_bits.log") 
# df_training = helper_functions.make_can_df("ambient_dyno_drive_basic_long.log") # Works well
# df_training = helper_functions.add_time_diff_per_aid_col(df_training)


# ## Train Model

# In[14]:


reload(helper_functions)

# Begin training on this dataset
# training_dict has aids as keys and dictionaries containing mu, std, kde, and gaussian as items
training_dict = {}
#cc = 0
for aid in tqdm(df_training.aid.unique()):
    #cc += 1
    #print (f'aid {cc} of 128')
    print("aid: ", aid)
    
    training_dict[aid] = helper_functions.train(df_training, aid)
    training_dict[aid]["y_thresholds_kde"] = {}
    training_dict[aid]["y_thresholds_gauss"] = {}
    
    #print("\n")


# In[16]:


#display(training_dict)


# ## Testing Data

# Here we exclude accelerator and masquerade attacks as well as the metadata file. Accelerator and masquerade attacjs are betond the scope of time-based detection methods.

# In[18]:


os.chdir(os.path.dirname(os.getcwd()) + "/attacks")
print(os.getcwd())


# In[21]:


df_aggregation = []

for file_name in os.listdir("."):
    
    if "masquerade" not in file_name and "accelerator" not in file_name and "metadata" not in file_name:
        print(file_name)
        
        df_attack = helper_functions.make_can_df(file_name)
        df_attack = helper_functions.add_time_diff_per_aid_col(df_attack)
        # print(df_attack.shape)
        # print(df.dtypes)
        df_aggregation.append(df_attack)
        
print(len(df_aggregation))


# In[23]:


display(df_aggregation[0])


# ## Find Injection Intervals

# Here we extract the injection interval of each attack.

# In[26]:


reload(helper_functions)

with open("capture_metadata.json", "r") as read_file:
    attack_dict = json.load(read_file)

attack_metadata = []

count = 0
for file_name in os.listdir("."):
    
    if "masquerade" not in file_name and "accelerator" not in file_name and "metadata" not in file_name:
        print(count, file_name)
        
        if attack_dict[file_name[:-4]]["injection_id"] != "XXX":
            print(attack_dict[file_name[:-4]]["injection_id"], int(attack_dict[file_name[:-4]]["injection_id"], 16))
        else:
            print(attack_dict[file_name[:-4]]["injection_id"])
        
        # print(attack_dict[file_name[:-4]]["injection_id"], int(attack_dict[file_name[:-4]]["injection_id"], 16))
        
        # attack_metadata.append(helper_functions.get_injection_interval(df_aggregation[count], int(attack_dict[file_name[:-4]]["injection_id"], 16),
        #                                                              attack_dict[file_name[:-4]]["injection_data_str"]))
        
        # From metadata file
        attack_metadata.append([tuple(attack_dict[file_name[:-4]]["injection_interval"])])
            
        count += 1


# ## Inspect Injection Intervals

# Check if injection intervals are consisten with the metadata file

# In[27]:


count = 0
for file_name in os.listdir("."):
    
    if "masquerade" not in file_name and "accelerator" not in file_name and "metadata" not in file_name:
    
        print(file_name)
        print(attack_metadata[count])
        print("\n")
        count += 1


# ## Annotate Attack DataFrames with Attacks

# Here we label each of the attack files based on the injection interval

# In[28]:


reload(helper_functions)

with open("capture_metadata.json", "r") as read_file:
    attack_dict = json.load(read_file)

count = 0
for file_name in os.listdir("."):
    
    if "masquerade" not in file_name and "accelerator" not in file_name and "metadata" not in file_name:
        print(count, file_name)
        
        print(attack_dict[file_name[:-4]]["injection_id"]) #, int(attack_dict[file_name[:-4]]["injection_id"], 16))#, attack_dict[file_name[:-4]]["injection_data_str"])
        
        # Add column to each attack dataframe to indicate attack (True) or non-attack (False) for each signal
        
        if attack_dict[file_name[:-4]]["injection_id"] != "XXX":
            df_aggregation[count] = helper_functions.add_actual_attack_col(df_aggregation[count], attack_metadata[count], int(attack_dict[file_name[:-4]]["injection_id"], 16), attack_dict[file_name[:-4]]["injection_data_str"])                                                      
        else:
            df_aggregation[count] = helper_functions.add_actual_attack_col(df_aggregation[count], attack_metadata[count], "XXX", attack_dict[file_name[:-4]]["injection_data_str"])  
            
        count += 1


# ## Time-Based Detector Methods Results

# To replicate the results of each of the methods presented in the paper, please run the respective method and save a `pickle` file with the corresponding results. Then you will have the option to reload the results file and replicate the figures of the paper. 

# ## Method 1: Mean

# In[29]:


os.path.dirname(os.getcwd())


# In[18]:


reload(helper_functions)

# # Generate results
# helper_functions.get_results_mean(df_aggregation, training_dict) # 35 seconds

# helper_functions.get_results_mean_various_p(df_aggregation, training_dict)

results_mean_final = helper_functions.unpickle(os.path.dirname(os.getcwd()) + "/results_mean_final.pkl") 


# ## Get Deeper on Mean

# In[17]:


df_aux = df_aggregation[0]

df_aux = df_aux[df_aux.alert_window == True]
print(df_aux.aid.unique())

# pd.set_option('display.max_rows', len(df_aux))
display(df_aux[df_aux.aid == 208]) #208 1760
# pd.reset_option('display.max_rows')


# In[106]:


# print("aids: ", df_training.aid.unique())

aid = 208 # 208 1760
method = "gauss" # gauss kde

## Training

df_aux = df_training[df_training.aid == aid]
vals = np.sort(df_aux.time_diffs.values)
pdf = training_dict[aid][method].pdf(vals) 

mu = df_aux.time_diffs.mean()
std = df_aux.time_diffs.std() 
max_val = max(vals)

# print(type(vals), type(vals[0]), len(vals), vals)
# print("mu: ", training_dict[aid]["mu"], mu)
# print("std: ", training_dict[aid]["std"], std)
# print("max: ", max_val)
# print(pdf)

new_model = scipy.stats.norm(loc=mu, scale=std)

fig, ax = plt.subplots(1, 1)

#ax.hist(vals, bins=np.arange(0, 0.2, 0.001), density=True)
ax.hist(vals, bins=25, edgecolor="black", density=True)
#ax.plot(vals, pdf, "k", lw=2, alpha=0.6)
# ax.plot(vals, new_model.pdf(vals), "r", lw=2, alpha=0.6)
ax.set_yscale("log", nonpositive="clip")
# ax.set_xlim(xmin=0, xmax=0.025)

plt.xlabel('Inter-arrival time (s)')
plt.ylabel('PDF')
plt.title("Training:" + " " + r"$\mu=$" + f"{mu:.5f}" + "  " + r"$\sigma=$" + f"{std:.5f}" + "  " + r"max$=$" + f"{max_val:.5f}")

plt.show()

# ## Testing

df_aux = df_aggregation[0][df_aggregation[0].aid == aid]
vals = np.sort(df_aux.time_diffs.values)
pdf = training_dict[aid][method].pdf(vals)

mu = df_aux.time_diffs.mean()
std = df_aux.time_diffs.std() 
max_val = max(vals)

# print(type(vals), type(vals[0]), len(vals), vals)
# print("mu: ", training_dict[aid]["mu"], mu)
# print("std: ", training_dict[aid]["std"], std)
# print("max: ", max_val)
# print(pdf)

new_model = scipy.stats.norm(loc=mu, scale=std)

fig, ax = plt.subplots(1, 1)

ax.hist(vals, bins=25, edgecolor="black", density=True)
#ax.hist(vals, bins=np.arange(0, 0.2, 0.001), density=True)
ax.set_yscale("log", nonpositive="clip")

# ax.plot(vals, pdf, "k", lw=2, alpha=0.6)
# ax.plot(vals, new_model.pdf(vals), "r", lw=2, alpha=0.6)
ax.set_xlim(xmin=0, xmax=0.025)

plt.xlabel('Inter-arrival time (s)')
plt.ylabel('PDF')
plt.title("Testing:" + " " + r"$\mu=$" + f"{mu:.5f}" + "  " + r"$\sigma=$" + f"{std:.5f}" + "  " + r"max$=$" + f"{max_val:.5f}")

plt.show()


# ## Training distributions

# ## 208

# In[45]:


aid = 208

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8)) # figsize=(3.4, 1.7) (default: [6.4, 4.8])
fig.subplots_adjust(wspace=0.3)

df_aux = df_training[df_training.aid == aid]
vals = np.sort(df_aux.time_diffs.values) 

mu = df_aux.time_diffs.mean()
std = df_aux.time_diffs.std() 
min_val = min(vals)
max_val = max(vals)

axes[0].hist(vals, bins=15, edgecolor="black")
axes[0].set_yscale("log", nonpositive="clip")
axes[0].set_xticks(np.arange(0, 250, 50))
axes[0].set_ylim([0.2, 10**6])
# ax.set_yticks(np.arange(0, 1.25, 0.25))

# axes[0].set_xlabel('Inter-arrival time (s)')
axes[0].set_ylabel('Frequency')
axes[0].text(80, 8*10**4, r"$\mu=$" + f"{mu:.3f}", color="black")
axes[0].text(80, 2.7*10**4, r"$\sigma=$" + f"{std:.3f}", color="black")
axes[0].text(80, 1*10**4, r"min$=$" + f"{min_val:.3f}", color="black")
axes[0].text(80, 3.5*10**3, r"max$=$" + f"{max_val:.3f}", color="black")
axes[0].annotate("(a)", xy=(25, 10**5))
#axes[0].set_title(r"$\mu=$" + f"{mu:.2f}" + "  " + r"$\sigma=$" + f"{std:.2f}" + "  " + r"max$=$" + f"{max_val:.2f}")

###

# identify outliers in the dataset
ee = EllipticEnvelope(contamination=0.0001, support_fraction=0.999) # support_fraction=0.99
inliers = ee.fit_predict(vals.reshape(-1, 1))

# select all rows that are not outliers
mask = inliers != -1
outliers = sum(mask == False)
print("outliers: ", outliers, 100*outliers/len(vals))

vals = vals[mask]

mu = vals.mean()
std = vals.std() 
min_val = min(vals)
max_val = max(vals)


axes[1].hist(vals, bins=15, edgecolor="black")
axes[1].set_yscale("log", nonpositive="clip")
axes[1].set_xticks(np.arange(0, 0.04, 0.01))
axes[1].set_ylim([0.2, 10**6])
# axes[0].set_xticks(np.arange(0, 250, 50))
# ax.set_yticks(np.arange(0, 1.25, 0.25))

# axes[0].set_xlabel('Inter-arrival time (s)')
# axes[0].set_ylabel('Frequency')
axes[1].text(0.016, 8*10**4, r"$\mu=$" + f"{mu:.3f}", color="black")
axes[1].text(0.016, 2.7*10**4, r"$\sigma=$" + f"{std:.3f}", color="black")
axes[1].text(0.016, 1*10**4, r"min$=$" + f"{min_val:.3f}", color="black")
axes[1].text(0.016, 3.5*10**3, r"max$=$" + f"{max_val:.3f}", color="black")
axes[1].annotate("(b)", xy=(0.003, 10**5))
# plt.title(r"$\mu=$" + f"{mu:.5f}" + "  " + r"$\sigma=$" + f"{std:.5f}" + "  " + r"max$=$" + f"{max_val:.5f}")

fig.text(0.5, 0.00, "Inter-arrival time (s)", ha="center")

#plt.savefig("../../figs/training_distributions_208.pdf", dpi=200, bbox_inches="tight")

plt.show()


# ## Fitting 208

# In[31]:


aid = 208
method = "gauss" # gauss, kde

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8)) # figsize=(3.4, 1.7) (default: [6.4, 4.8])
fig.subplots_adjust(wspace=0.3)

df_aux = df_training[df_training.aid == aid]
vals = np.sort(df_aux.time_diffs.values)
pdf = training_dict[aid][method].pdf(vals) 

mu = df_aux.time_diffs.mean()
std = df_aux.time_diffs.std() 
min_val = min(vals)
max_val = max(vals)

axes[0].hist(vals, bins=15, density=True, edgecolor="black")
axes[0].plot(vals, pdf, "k-", lw=2)
axes[0].set_yscale("log", nonpositive="clip")
axes[0].set_xticks(np.arange(0, 250, 50))
axes[0].set_ylim([0.2, 10**4])
# ax.set_yticks(np.arange(0, 1.25, 0.25))

# axes[0].set_xlabel('Inter-arrival time (s)')
axes[0].set_ylabel('PDF')
axes[0].text(80, 4*10**3, r"$\mu=$" + f"{mu:.3f}", color="black")
axes[0].text(80, 2*10**3, r"$\sigma=$" + f"{std:.3f}", color="black")
axes[0].text(80, 1*10**3, r"min$=$" + f"{min_val:.3f}", color="black")
axes[0].text(80, 5*10**2, r"max$=$" + f"{max_val:.3f}", color="black")
axes[0].annotate("(a)", xy=(25, 10**3))
#axes[0].set_title(r"$\mu=$" + f"{mu:.2f}" + "  " + r"$\sigma=$" + f"{std:.2f}" + "  " + r"max$=$" + f"{max_val:.2f}")


###

# identify outliers in the dataset
ee = EllipticEnvelope(contamination=0.0001, support_fraction=0.999) # support_fraction=0.99
inliers = ee.fit_predict(vals.reshape(-1, 1))

# select all rows that are not outliers
mask = inliers != -1
outliers = sum(mask == False)
print("outliers: ", outliers, 100*outliers/len(vals))

vals = vals[mask]
pdf = training_dict[aid][method].pdf(vals) 

mu = vals.mean()
std = vals.std() 
min_val = min(vals)
max_val = max(vals)

axes[1].hist(vals, bins=15, density=True, edgecolor="black")
axes[1].plot(vals, pdf, "k-", lw=2)
axes[1].set_yscale("log", nonpositive="clip")
axes[1].set_xticks(np.arange(0, 0.04, 0.01))
axes[1].set_ylim([0.2, 10**4])
# axes[0].set_xticks(np.arange(0, 250, 50))
# ax.set_yticks(np.arange(0, 1.25, 0.25))

# axes[0].set_xlabel('Inter-arrival time (s)')
# axes[0].set_ylabel('Frequency')
axes[1].text(0.016, 4*10**3, r"$\mu=$" + f"{mu:.3f}", color="black")
axes[1].text(0.016, 2*10**3, r"$\sigma=$" + f"{std:.3f}", color="black")
axes[1].text(0.016, 1*10**3, r"min$=$" + f"{min_val:.3f}", color="black")
axes[1].text(0.016, 5*10**2, r"max$=$" + f"{max_val:.3f}", color="black")
axes[1].annotate("(b)", xy=(0.003, 10**3))
# plt.title(r"$\mu=$" + f"{mu:.5f}" + "  " + r"$\sigma=$" + f"{std:.5f}" + "  " + r"max$=$" + f"{max_val:.5f}")

fig.text(0.5, 0.00, "Inter-arrival time (s)", ha="center")

#plt.savefig("../../figs/training_distributions_208_fit.pdf", dpi=200, bbox_inches="tight")

plt.show()


# ## 1760

# In[43]:


aid = 1760

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8)) # figsize=(3.4, 1.7) (default: [6.4, 4.8])
fig.subplots_adjust(wspace=0.3)

df_aux = df_training[df_training.aid == aid]
vals = np.sort(df_aux.time_diffs.values) 

mu = df_aux.time_diffs.mean()
std = df_aux.time_diffs.std() 
min_val = min(vals)
max_val = max(vals)


axes[0].hist(vals, bins=15, edgecolor="black")
axes[0].set_yscale("log", nonpositive="clip")
axes[0].set_xticks(np.arange(0, 35, 5))
axes[0].set_ylim([0.2, 10**6])
# # ax.set_yticks(np.arange(0, 1.25, 0.25))

# # axes[0].set_xlabel('Inter-arrival time (s)')
axes[0].set_ylabel('Frequency')
axes[0].text(15, 8*10**4, r"$\mu=$" + f"{mu:.3f}", color="black")
axes[0].text(15, 2.7*10**4, r"$\sigma=$" + f"{std:.3f}", color="black")
axes[0].text(15, 1*10**4, r"min$=$" + f"{min_val:.3f}", color="black")
axes[0].text(15, 3.5*10**3, r"max$=$" + f"{max_val:.3f}", color="black")
axes[0].annotate("(a)", xy=(5, 10**5))
# #axes[0].set_title(r"$\mu=$" + f"{mu:.2f}" + "  " + r"$\sigma=$" + f"{std:.2f}" + "  " + r"max$=$" + f"{max_val:.2f}")

###

# identify outliers in the dataset
ee = EllipticEnvelope(contamination=0.0001, support_fraction=0.999) # support_fraction=0.99
inliers = ee.fit_predict(vals.reshape(-1, 1))

# select all rows that are not outliers
mask = inliers != -1
outliers = sum(mask == False)
print("outliers: ", outliers, 100*outliers/len(vals))

vals = vals[mask]

mu = vals.mean()
std = vals.std() 
min_val = min(vals)
max_val = max(vals)

axes[1].hist(vals, bins=15, edgecolor="black")
axes[1].set_yscale("log", nonpositive="clip")
axes[1].set_xticks(np.arange(0, 0.04, 0.01))
axes[1].set_ylim([0.2, 10**6])
# # axes[0].set_xticks(np.arange(0, 250, 50))
# # ax.set_yticks(np.arange(0, 1.25, 0.25))

# axes[0].set_xlabel('Inter-arrival time (s)')
# axes[0].set_ylabel('Frequency')
axes[1].text(0.017, 8*10**4, r"$\mu=$" + f"{mu:.3f}", color="black")
axes[1].text(0.017, 2.7*10**4, r"$\sigma=$" + f"{std:.3f}", color="black")
axes[1].text(0.017, 1*10**4, r"min$=$" + f"{min_val:.3f}", color="black")
axes[1].text(0.017, 3.5*10**3, r"max$=$" + f"{max_val:.3f}", color="black")
axes[1].annotate("(b)", xy=(0.003, 10**5))
# plt.title(r"$\mu=$" + f"{mu:.5f}" + "  " + r"$\sigma=$" + f"{std:.5f}" + "  " + r"max$=$" + f"{max_val:.5f}")

fig.text(0.5, 0.00, "Inter-arrival time (s)", ha="center")

# plt.savefig("../../figs/training_distributions_1760.pdf", dpi=200, bbox_inches="tight")
plt.show()


# ## Fitting 1760

# In[42]:


aid = 1760
method = "gauss"

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8)) # figsize=(3.4, 1.7) (default: [6.4, 4.8])
fig.subplots_adjust(wspace=0.3)

df_aux = df_training[df_training.aid == aid]
vals = np.sort(df_aux.time_diffs.values)
pdf = training_dict[aid][method].pdf(vals) 

mu = df_aux.time_diffs.mean()
std = df_aux.time_diffs.std() 
min_val = min(vals)
max_val = max(vals)


axes[0].hist(vals, bins=15, density=True, edgecolor="black")
axes[0].plot(vals, pdf, "k-", lw=2)
axes[0].set_yscale("log", nonpositive="clip")
axes[0].set_xticks(np.arange(0, 35, 5))
#axes[0].set_xlim([0, 2])
axes[0].set_ylim([0.2, 10**4])
# # ax.set_yticks(np.arange(0, 1.25, 0.25))

axes[0].set_ylabel('PDF')
axes[0].text(15, 4*10**3, r"$\mu=$" + f"{mu:.3f}", color="black")
axes[0].text(15, 2*10**3, r"$\sigma=$" + f"{std:.3f}", color="black")
axes[0].text(15, 1*10**3, r"min$=$" + f"{min_val:.3f}", color="black")
axes[0].text(15, 5*10**2, r"max$=$" + f"{max_val:.3f}", color="black")
axes[0].annotate("(a)", xy=(5, 10**3))
# #axes[0].set_title(r"$\mu=$" + f"{mu:.2f}" + "  " + r"$\sigma=$" + f"{std:.2f}" + "  " + r"max$=$" + f"{max_val:.2f}")

##

# identify outliers in the dataset
ee = EllipticEnvelope(contamination=0.0001, support_fraction=0.999) # support_fraction=0.99
inliers = ee.fit_predict(vals.reshape(-1, 1))

# select all rows that are not outliers
mask = inliers != -1
outliers = sum(mask == False)
print("outliers: ", outliers, 100*outliers/len(vals))

vals = vals[mask]
pdf = training_dict[aid][method].pdf(vals) 

mu = vals.mean()
std = vals.std() 
min_val = min(vals)
max_val = max(vals)

axes[1].hist(vals, bins=15, density=True, edgecolor="black")
axes[1].plot(vals, pdf, "k-", lw=2)
axes[1].set_yscale("log", nonpositive="clip")
axes[1].set_xticks(np.arange(0, 0.04, 0.01))
axes[1].set_ylim([0.2, 10**4])
# # axes[0].set_xticks(np.arange(0, 250, 50))
# # ax.set_yticks(np.arange(0, 1.25, 0.25))

# axes[0].set_xlabel('Inter-arrival time (s)')
# axes[0].set_ylabel('Frequency')
axes[1].text(0.017, 4*10**3, r"$\mu=$" + f"{mu:.3f}", color="black")
axes[1].text(0.017, 2*10**3, r"$\sigma=$" + f"{std:.3f}", color="black")
axes[1].text(0.017, 1*10**3, r"min$=$" + f"{min_val:.3f}", color="black")
axes[1].text(0.017, 5*10**2, r"max$=$" + f"{max_val:.3f}", color="black")
axes[1].annotate("(b)", xy=(0.003, 10**3))
# plt.title(r"$\mu=$" + f"{mu:.5f}" + "  " + r"$\sigma=$" + f"{std:.5f}" + "  " + r"max$=$" + f"{max_val:.5f}")

fig.text(0.5, 0.00, "Inter-arrival time (s)", ha="center")

# plt.savefig("../../figs/training_distributions_1760_fit.pdf", dpi=200, bbox_inches="tight")
plt.show()


# ## Fitting 208 and 1760

# In[55]:


aid = 208
method = "gauss" # gauss, kde

fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.4)) # figsize=(3.4, 1.7) (default: [6.4, 4.8])
fig.subplots_adjust(wspace=0.3, hspace=0.3)

df_aux = df_training[df_training.aid == aid]
vals = np.sort(df_aux.time_diffs.values)
pdf = training_dict[aid][method].pdf(vals) 

mu = df_aux.time_diffs.mean()
std = df_aux.time_diffs.std() 
min_val = min(vals)
max_val = max(vals)

axes[0][0].hist(vals, bins=15, density=True, edgecolor="black")
axes[0][0].plot(vals, pdf, "k-", lw=2)
axes[0][0].set_yscale("log", nonpositive="clip")
axes[0][0].set_xticks(np.arange(0, 250, 50))
axes[0][0].set_ylim([0.2, 10**4])
# ax.set_yticks(np.arange(0, 1.25, 0.25))

# axes[0].set_xlabel('Inter-arrival time (s)')
axes[0][0].set_ylabel('PDF')
axes[0][0].text(80, 4*10**3, r"$\mu=$" + f"{mu:.3f}", color="black")
axes[0][0].text(80, 2*10**3, r"$\sigma=$" + f"{std:.3f}", color="black")
axes[0][0].text(80, 1*10**3, r"min$=$" + f"{min_val:.3f}", color="black")
axes[0][0].text(80, 5*10**2, r"max$=$" + f"{max_val:.3f}", color="black")
axes[0][0].annotate("(a)", xy=(25, 10**3))
#axes[0].set_title(r"$\mu=$" + f"{mu:.2f}" + "  " + r"$\sigma=$" + f"{std:.2f}" + "  " + r"max$=$" + f"{max_val:.2f}")

# identify outliers in the dataset
ee = EllipticEnvelope(contamination=0.0001, support_fraction=0.999) # support_fraction=0.99
inliers = ee.fit_predict(vals.reshape(-1, 1))

# select all rows that are not outliers
mask = inliers != -1
outliers = sum(mask == False)
print("outliers: ", outliers, 100*outliers/len(vals))

vals = vals[mask]
pdf = training_dict[aid][method].pdf(vals) 

mu = vals.mean()
std = vals.std() 
min_val = min(vals)
max_val = max(vals)

axes[0][1].hist(vals, bins=15, density=True, edgecolor="black")
axes[0][1].plot(vals, pdf, "k-", lw=2)
axes[0][1].set_yscale("log", nonpositive="clip")
axes[0][1].set_xticks(np.arange(0, 0.04, 0.01))
axes[0][1].set_ylim([0.2, 10**4])
# axes[0].set_xticks(np.arange(0, 250, 50))
# ax.set_yticks(np.arange(0, 1.25, 0.25))

# axes[0].set_xlabel('Inter-arrival time (s)')
# axes[0].set_ylabel('Frequency')
axes[0][1].text(0.016, 4*10**3, r"$\mu=$" + f"{mu:.3f}", color="black")
axes[0][1].text(0.016, 2*10**3, r"$\sigma=$" + f"{std:.3f}", color="black")
axes[0][1].text(0.016, 1*10**3, r"min$=$" + f"{min_val:.3f}", color="black")
axes[0][1].text(0.016, 5*10**2, r"max$=$" + f"{max_val:.3f}", color="black")
axes[0][1].annotate("(b)", xy=(0.003, 10**3))
# plt.title(r"$\mu=$" + f"{mu:.5f}" + "  " + r"$\sigma=$" + f"{std:.5f}" + "  " + r"max$=$" + f"{max_val:.5f}")

###########

aid = 1760

df_aux = df_training[df_training.aid == aid]
vals = np.sort(df_aux.time_diffs.values)
pdf = training_dict[aid][method].pdf(vals) 

mu = df_aux.time_diffs.mean()
std = df_aux.time_diffs.std() 
min_val = min(vals)
max_val = max(vals)

axes[1][0].hist(vals, bins=15, density=True, edgecolor="black")
axes[1][0].plot(vals, pdf, "k-", lw=2)
axes[1][0].set_yscale("log", nonpositive="clip")
axes[1][0].set_xticks(np.arange(0, 35, 5))
#axes[0].set_xlim([0, 2])
axes[1][0].set_ylim([0.2, 10**4])
# # ax.set_yticks(np.arange(0, 1.25, 0.25))

axes[1][0].set_ylabel('PDF')
axes[1][0].text(15, 4*10**3, r"$\mu=$" + f"{mu:.3f}", color="black")
axes[1][0].text(15, 2*10**3, r"$\sigma=$" + f"{std:.3f}", color="black")
axes[1][0].text(15, 1*10**3, r"min$=$" + f"{min_val:.3f}", color="black")
axes[1][0].text(15, 5*10**2, r"max$=$" + f"{max_val:.3f}", color="black")
axes[1][0].annotate("(c)", xy=(5, 10**3))
# #axes[0].set_title(r"$\mu=$" + f"{mu:.2f}" + "  " + r"$\sigma=$" + f"{std:.2f}" + "  " + r"max$=$" + f"{max_val:.2f}")

# identify outliers in the dataset
ee = EllipticEnvelope(contamination=0.0001, support_fraction=0.999) # support_fraction=0.99
inliers = ee.fit_predict(vals.reshape(-1, 1))

# select all rows that are not outliers
mask = inliers != -1
outliers = sum(mask == False)
print("outliers: ", outliers, 100*outliers/len(vals))

vals = vals[mask]
pdf = training_dict[aid][method].pdf(vals) 

mu = vals.mean()
std = vals.std() 
min_val = min(vals)
max_val = max(vals)

axes[1][1].hist(vals, bins=15, density=True, edgecolor="black")
axes[1][1].plot(vals, pdf, "k-", lw=2)
axes[1][1].set_yscale("log", nonpositive="clip")
axes[1][1].set_xticks(np.arange(0, 0.04, 0.01))
axes[1][1].set_ylim([0.2, 10**4])
# # axes[0].set_xticks(np.arange(0, 250, 50))
# # ax.set_yticks(np.arange(0, 1.25, 0.25))

# axes[0].set_xlabel('Inter-arrival time (s)')
# axes[0].set_ylabel('Frequency')
axes[1][1].text(0.017, 4*10**3, r"$\mu=$" + f"{mu:.3f}", color="black")
axes[1][1].text(0.017, 2*10**3, r"$\sigma=$" + f"{std:.3f}", color="black")
axes[1][1].text(0.017, 1*10**3, r"min$=$" + f"{min_val:.3f}", color="black")
axes[1][1].text(0.017, 5*10**2, r"max$=$" + f"{max_val:.3f}", color="black")
axes[1][1].annotate("(d)", xy=(0.003, 10**3))
# plt.title(r"$\mu=$" + f"{mu:.5f}" + "  " + r"$\sigma=$" + f"{std:.5f}" + "  " + r"max$=$" + f"{max_val:.5f}")

fig.text(0.5, 0.07, "Inter-arrival time (s)", ha="center")

plt.savefig("../../figs/training_distributions_fit_combined.pdf", dpi=200, bbox_inches="tight")

plt.show()


# ## Testing distributions

# In[44]:


##

aid = 208 # 1760 1255

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8)) # figsize=(3.4, 1.7) (default: [6.4, 4.8])
fig.subplots_adjust(wspace=0.3)

df_aux = pd.concat(df_aggregation)
df_aux = df_aux[df_aux.aid == aid]
vals = np.sort(df_aux.time_diffs.values)

mu = vals.mean()
std = vals.std() 
max_val = max(vals)
min_val = min(vals)

# print(type(vals), type(vals[0]), len(vals), vals)
# print("mu: ", training_dict[aid]["mu"], mu)
# print("std: ", training_dict[aid]["std"], std)
# print("max: ", max_val)
# print(pdf)

axes[0].hist(vals, bins=15, edgecolor="black")
axes[0].set_yscale("log", nonpositive="clip")
axes[0].set_xticks(np.arange(0, 0.026, 0.01))
axes[0].set_ylim([0.2, 10**6])
axes[0].annotate("(a)", xy=(0.002, 10**5))

axes[0].set_ylabel('Frequency')
axes[0].text(0.0115, 3*10**5, r"$\mu=$" + f"{mu:.3f}", color="black")
axes[0].text(0.0115, 1*10**5, r"$\sigma=$" + f"{std:.3f}", color="black")
axes[0].text(0.0115, 3.5*10**4, r"min$=$" + f"{min_val:.3f}", color="black")
axes[0].text(0.0115, 1.3*10**4, r"max$=$" + f"{max_val:.3f}", color="black")

###

aid = 1760

df_aux = pd.concat(df_aggregation)
df_aux = df_aux[df_aux.aid == aid]
vals = np.sort(df_aux.time_diffs.values)

mu = vals.mean()
std = vals.std() 
min_val = min(vals)
max_val = max(vals)

# print(type(vals), type(vals[0]), len(vals), vals)
# print("mu: ", training_dict[aid]["mu"], mu)
# print("std: ", training_dict[aid]["std"], std)
# print("max: ", max_val)
# print(pdf)

axes[1].hist(vals, bins=15, edgecolor="black")
axes[1].set_yscale("log", nonpositive="clip")
axes[1].set_xticks(np.arange(0, 0.026, 0.01))
axes[1].set_ylim([0.2, 10**6])
axes[1].annotate("(b)", xy=(0.002, 10**5))

axes[1].text(0.0115, 3*10**5, r"$\mu=$" + f"{mu:.3f}", color="black")
axes[1].text(0.0115, 1*10**5, r"$\sigma=$" + f"{std:.3f}", color="black")
axes[1].text(0.0115, 3.5*10**4, r"min$=$" + f"{min_val:.3f}", color="black")
axes[1].text(0.0115, 1.3*10**4, r"max$=$" + f"{max_val:.3f}", color="black")

fig.text(0.5, 0.00, "Inter-arrival time (s)", ha="center")

plt.savefig("../../figs/testing_distributions.pdf", dpi=200, bbox_inches="tight")

plt.show()


# ## ROC/PRC Curves

# In[20]:


pvals_mean = np.linspace(0, 1, 19)
# print(len(pvals_mean), pvals_mean)

for pval in pvals_mean:
    print(pval, results_mean_final[pval]["f1"])
    
optimum = 0.4444444444444444

## Generate ROC curve
x_roc_mean = [results_mean_final[p]['false_pos'] for p in pvals_mean]
y_roc_mean = [results_mean_final[p]['recall'] for p in pvals_mean]

plt.plot(x_roc_mean, y_roc_mean, '.-')
plt.plot(results_mean_final[optimum]['false_pos'], results_mean_final[optimum]['recall'], '.-r')
plt.xlabel('False Positive Rate')
plt.ylabel('Recall')
plt.title('ROC Curve')

plt.show()
    
# Can use this dictionary to generate PRC curve
x_prc_mean = [results_mean_final[p]['recall'] for p in pvals_mean]
y_prc_mean = [results_mean_final[p]['prec'] for p in pvals_mean]

# print(x_roc_kde, y_roc_kde)
plt.plot(x_prc_mean, y_prc_mean, '.-')
plt.plot(results_mean_final[optimum]['recall'], results_mean_final[optimum]['prec'], '.-r')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRC Curve')

plt.show()


print("confusion matrix: \n", results_mean_final[optimum]["cm"])
print("precision: ", results_mean_final[optimum]["prec"])
print("recall: ", results_mean_final[optimum]["recall"])
print("f1 score: ", results_mean_final[optimum]["f1"])
print("false positive rate", results_mean_final[optimum]["false_pos"])

# print("confusion matrix: \n", results_mean_final["total"]["cm"])
# print("precision: ", results_mean_final["total"]["prec"])
# print("recall: ", results_mean_final["total"]["recall"])
# print("f1 score: ", results_mean_final["total"]["f1"])
# print("false positive rate", results_mean_final["total"]["false_pos"])


# In[23]:


# training_dict
# df_aggregation[0]


# ## Method 4: Binning

# In[21]:


reload(helper_functions)

# # The above dictionary informed our decision for how to write our alert_by_bin function
# # Now we get results using that function
# helper_functions.get_results_binning(df_aggregation, training_dict, 6) # 10 seconds

# helper_functions.get_results_binning_various_p(df_aggregation, training_dict, 6) # 10 seconds

results_binning_final = helper_functions.unpickle(os.path.dirname(os.getcwd()) + "/results_binning_final.pkl") 


# In[22]:


pvals_binning = np.linspace(1, 10, 19) # 0, 4
# print(len(pvals_binning), pvals_binning)

for pval in pvals_binning:
    print(pval, results_binning_final[pval]["f1"])
    
optimum = 3.5

## Generate ROC curve
x_roc_binning = [results_binning_final[p]['false_pos'] for p in pvals_binning]
y_roc_binning = [results_binning_final[p]['recall'] for p in pvals_binning]

plt.plot(x_roc_binning, y_roc_binning, '.-')
plt.plot(results_binning_final[optimum]['false_pos'], results_binning_final[optimum]['recall'], '.-r')
plt.xlabel('False Positive Rate')
plt.ylabel('Recall')
plt.title('ROC Curve')

plt.show()


# Can use this dictionary to generate PRC curve
x_prc_binning = [results_binning_final[p]['recall'] for p in pvals_binning]
y_prc_binning = [results_binning_final[p]['prec'] for p in pvals_binning]

# print(x_roc_kde, y_roc_kde)
plt.plot(x_prc_binning, y_prc_binning, '.-')
plt.plot(results_binning_final[optimum]['recall'], results_binning_final[optimum]['prec'], '.-r')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRC Curve')

plt.show()


print("confusion matrix: \n", results_binning_final[optimum]["cm"])
print("precision: ", results_binning_final[optimum]["prec"])
print("recall: ", results_binning_final[optimum]["recall"])
print("f1 score: ", results_binning_final[optimum]["f1"])
print("false positive rate", results_binning_final[optimum]["false_pos"])

# print("confusion matrix: \n", results_binning_final["total"]["cm"])
# print("precision: ", results_binning_final["total"]["prec"])
# print("recall: ", results_binning_final["total"]["recall"])
# print("f1 score: ", results_binning_final["total"]["f1"])
# print("false positive rate", results_binning_final["total"]["false_pos"])


# In[28]:


df_aggregation[0]


# ## Method 3: Gaussian

# In[29]:


reload(helper_functions)

with open("capture_metadata.json", "r") as read_file:
    attack_dict = json.load(read_file)

count = 0
for file_name in tqdm(os.listdir(".")):
    
    if "masquerade" not in file_name and "accelerator" not in file_name and "metadata" not in file_name:
    # if "masquerade" not in file_name and "accelerator" not in file_name and "fuzzing" not in file_name and "metadata" not in file_name:
        print(file_name)
        
        # print(attack_dict[file_name[:-4]]["injection_id"], int(attack_dict[file_name[:-4]]["injection_id"], 16), attack_dict[file_name[:-4]]["injection_data_str"])
        
        # Adds column to df with the value of the Guassian approximation at each time_diff in the df
        df_aggregation[count] = helper_functions.add_gauss_val_col(df_aggregation[count], training_dict)                                                   
        
        count += 1


# In[30]:


df_aggregation[0]


# In[31]:


reload(helper_functions)

# Decide which pvalues to test with the Gaussain approximation
pvals_gauss = sorted(list(np.arange(0.001, 0.01, 0.001)) + list(np.arange(0.01, 0.1, 0.01)))
print(pvals_gauss)


# pvals_gauss = list(np.arange(0.0001, 0.0011, 0.0001))
# print(pvals_gauss)

# # Fill each 'y_thresholds_gauss' dictionary with p values and their corresponding y_thresholds
# # After this has been done once, we don't need to do it again since it has been pickled.
# # Commenting out the code to avoid rewriting the data we have.

for aid in tqdm(df_training.aid.unique()):
    
    training_dict[aid]['y_thresholds_gauss'] = {}
    for p in pvals_gauss:
        helper_functions.y_threshold_gauss(training_dict, aid, p)
        
# helper_functions.picklify(training_dict, os.path.dirname(os.getcwd()) + "/training_dict.pkl")


# In[32]:


helper_functions.picklify(training_dict, os.path.dirname(os.getcwd()) + "/training_dict.pkl")
training_dict[37]


# In[33]:


df_aggregation[0]


# In[23]:


reload(helper_functions)

# training_dict = helper_functions.unpickle(os.path.dirname(os.getcwd()) + "/training_dict.pkl") 
# helper_functions.get_results_gauss(df_aggregation, training_dict)

results_gauss_final = helper_functions.unpickle(os.path.dirname(os.getcwd()) + "/results_gauss_final.pkl") 


# In[24]:


pvals_gauss = sorted(list(np.arange(0.001, 0.01, 0.001)) + list(np.arange(0.01, 0.1, 0.01)))
# print(pvals_gauss)

for pval in pvals_gauss:
    print(pval, results_gauss_final[pval]["f1"])
    
optimum = 0.09

## Generate ROC curve
x_roc_gauss = [results_gauss_final[p]['false_pos'] for p in pvals_gauss]
y_roc_gauss = [results_gauss_final[p]['recall'] for p in pvals_gauss]

plt.plot(x_roc_gauss, y_roc_gauss, '.-')
plt.plot(results_gauss_final[optimum]['false_pos'], results_gauss_final[optimum]['recall'], '.-r')
plt.xlabel('False Positive Rate')
plt.ylabel('Recall')
plt.title('ROC Curve')

plt.show()

# Can use this dictionary to generate PRC curve
x_prc_gauss = [results_gauss_final[p]['recall'] for p in pvals_gauss]
y_prc_gauss = [results_gauss_final[p]['prec'] for p in pvals_gauss]

# print(x_roc_kde, y_roc_kde)
plt.plot(x_prc_gauss, y_prc_gauss, '.-')
plt.plot(results_gauss_final[optimum]['recall'], results_gauss_final[optimum]['prec'], '.-r')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRC Curve')

plt.show()

print("confusion matrix: \n", results_gauss_final[optimum]["cm"])
print("precision: ", results_gauss_final[optimum]["prec"])
print("recall: ", results_gauss_final[optimum]["recall"])
print("f1 score: ", results_gauss_final[optimum]["f1"])
print("false positive rate", results_gauss_final[optimum]["false_pos"])


# ## Total Statistics

# In[38]:


print(100*np.array(pvals_gauss))

count = 0
total = 0
for df in df_aggregation:
    count += np.sum(df.actual_attack.values)
    total += df.shape[0]
    
print("attacks: ", count)
print("benign: ", total-count)
print("total: ", total)


# ## Method 2: KDE

# In[39]:


reload(helper_functions)

with open("capture_metadata.json", "r") as read_file:
    attack_dict = json.load(read_file)

count = 0
for file_name in tqdm(os.listdir(".")):
    
    if "masquerade" not in file_name and "accelerator" not in file_name and "metadata" not in file_name:
    # if "masquerade" not in file_name and "accelerator" not in file_name and "fuzzing" not in file_name and "metadata" not in file_name:
        print(file_name)
        
        # print(attack_dict[file_name[:-4]]["injection_id"], int(attack_dict[file_name[:-4]]["injection_id"], 16), attack_dict[file_name[:-4]]["injection_data_str"])
    
        # Add column to each attack dataframe to indicate attack the gaussian corresponding value
        df_aggregation[count] = helper_functions.add_kde_val_col(df_aggregation[count], training_dict)                                                   
        
        count += 1


# In[40]:


df_aggregation[0]


# In[42]:


# helper_functions.picklify(df_aggregation, os.path.dirname(os.getcwd()) + "/df_aggregation_final.pkl")

df_aggregation = helper_functions.unpickle(os.path.dirname(os.getcwd()) + "/df_aggregation_final.pkl") 

df_aggregation[0]


# In[43]:


# Fill this list with the pvals we want to find y_thresholds for
pvals_kde = sorted(list(np.arange(0.001, 0.01, 0.001)) + list(np.arange(0, 0.1, 0.01)))
print(pvals_kde)


# In[94]:


#training_dict


# In[44]:


# reload(helper_functions)

# # Fill each 'y_thresholds_kde' dictionary with p values and their corresponding y_thresholds
# # Once this has been done, results are pickled and don't need to be re-run.
# # Will comment this out to avoid overwriting
# #cc = 0
# for aid in tqdm(df_training.aid.unique()):
#     #cc += 1
#     #print (f'aid {cc} of 128')
#     for p in pvals_kde:
#         helper_functions.y_threshold_kde(training_dict, aid, p)
        
# helper_functions.picklify(training_dict, os.path.dirname(os.getcwd()) + "/training_dict.pkl")


# In[45]:


training_dict = helper_functions.unpickle(os.path.dirname(os.getcwd()) + "/training_dict.pkl") 


# In[47]:


#training_dict
#df_aggregation[0]


# In[48]:


# reload(helper_functions)
# helper_functions.get_results_kde(pvals_kde, df_aggregation, training_dict)


# In[25]:


results_kde_final = helper_functions.unpickle(os.path.dirname(os.getcwd()) + "/results_kde_final.pkl") 


# In[26]:


pvals_kde = sorted(list(np.arange(0.001, 0.01, 0.001)) + list(np.arange(0, 0.1, 0.01)))
# print(pvals_kde)

for pval in pvals_kde:
    print(pval, results_kde_final[pval]["f1"])
    
optimum = 0.09

#Can use this dictionary to generate ROC curve
x_roc_kde = [results_kde_final[p]['false_pos'] for p in pvals_kde]
y_roc_kde = [results_kde_final[p]['recall'] for p in pvals_kde]

# print(x_roc_kde, y_roc_kde)
plt.plot(x_roc_kde, y_roc_kde, '.-')
plt.plot(results_kde_final[optimum]['false_pos'], results_kde_final[optimum]['recall'], '.-r')
plt.xlabel('False Positive Rate')
plt.ylabel('Recall')
plt.title('ROC Curve')

plt.show()

# Can use this dictionary to generate PRC curve
x_prc_kde = [results_kde_final[p]['recall'] for p in pvals_kde]
y_prc_kde = [results_kde_final[p]['prec'] for p in pvals_kde]

# print(x_roc_kde, y_roc_kde)
plt.plot(x_prc_kde, y_prc_kde, '.-')
plt.plot(results_kde_final[optimum]['recall'], results_kde_final[optimum]['prec'], '.-r')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRC Curve')

plt.show()

print("confusion matrix: \n", results_kde_final[optimum]["cm"])
print("precision: ", results_kde_final[optimum]["prec"])
print("recall: ", results_kde_final[optimum]["recall"])
print("f1 score: ", results_kde_final[optimum]["f1"])
print("false positive rate", results_kde_final[optimum]["false_pos"])


# ## Aggregated Figure

# In[19]:


fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.4))

ax.plot(x_prc_mean, y_prc_mean, color="#377eb8", marker="o", markersize=5, label="Mean")
# print(list(x_prc_mean))
# print(list(y_prc_mean))
print("pr auc mean: ", auc(np.nan_to_num(x_prc_mean), np.nan_to_num(y_prc_mean)))

ax.plot(x_prc_binning, y_prc_binning, color="#ff7f00", marker="s", markersize=5, label="Binning")
print("pr auc binning: ", auc(x_prc_binning, y_prc_binning))

ax.plot(x_prc_gauss, y_prc_gauss, color="#4daf4a", marker="^", markersize=5, label="Gaussian")
print("pr auc gauss: ", auc(x_prc_gauss, y_prc_gauss))

ax.plot(x_prc_kde, y_prc_kde, color="#f781bf", marker="v", markersize=5, label="KDE")
print("pr auc kde: ", auc(x_prc_kde, y_prc_kde))

ax.plot(np.linspace(0, 1, 20), 20*[0.039], "--", color="gray", markersize=5)
ax.text(0.25, -0.05, "Baseline: 0.039", color="gray")

ax.set_xticks(np.arange(0, 1.25, 0.25))
ax.set_yticks(np.arange(0, 1.25, 0.25))

ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.1, 1.1])

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend(loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.2),  ncol=2, fancybox=False, shadow=False, labelspacing=0.5, columnspacing=1)


#plt.savefig("../../figs/figure_prc.pdf", dpi=200, bbox_inches="tight")
plt.show()


# In[45]:


pvals_gauss = sorted(list(np.arange(0.001, 0.01, 0.001)) + list(np.arange(0.01, 0.1, 0.01)))
print(len(pvals_gauss), pvals_gauss)


# ## Unified Figure

# In[18]:


# Without outliers

results_mean_final_no_pretraining = helper_functions.unpickle(os.path.dirname(os.getcwd()) + "/results_mean_final_no_pretraining.pkl") 
pvals_mean_no_pretraining = np.linspace(0, 1, 19)
x_prc_mean_no_pretraining = [results_mean_final_no_pretraining[p]['recall'] for p in pvals_mean_no_pretraining]
y_prc_mean_no_pretraining = [results_mean_final_no_pretraining[p]['prec'] for p in pvals_mean_no_pretraining]

results_binning_final_no_pretraining = helper_functions.unpickle(os.path.dirname(os.getcwd()) + "/results_binning_final_no_pretraining.pkl") 
pvals_binning_no_pretraining = np.linspace(1, 10, 19)
x_prc_binning_no_pretraining = [results_binning_final_no_pretraining[p]['recall'] for p in pvals_binning_no_pretraining]
y_prc_binning_no_pretraining = [results_binning_final_no_pretraining[p]['prec'] for p in pvals_binning_no_pretraining]

results_gauss_final_no_pretraining = helper_functions.unpickle(os.path.dirname(os.getcwd()) + "/results_gauss_final_no_pretraining.pkl")
pvals_gauss_no_pretraining = sorted(list(np.arange(0.001, 0.01, 0.001)) + list(np.arange(0.01, 0.1, 0.01)))
x_prc_gauss_no_pretraining  = [results_gauss_final_no_pretraining [p]['recall'] for p in pvals_gauss_no_pretraining]
y_prc_gauss_no_pretraining  = [results_gauss_final_no_pretraining [p]['prec'] for p in pvals_gauss_no_pretraining]

results_kde_final_no_pretraining = helper_functions.unpickle(os.path.dirname(os.getcwd()) + "/results_kde_final_no_pretraining.pkl")
pvals_kde_no_pretraining = sorted(list(np.arange(0.001, 0.01, 0.001)) + list(np.arange(0, 0.1, 0.01)))
x_prc_kde_no_pretraining  = [results_kde_final_no_pretraining [p]['recall'] for p in pvals_kde_no_pretraining]
y_prc_kde_no_pretraining  = [results_kde_final_no_pretraining [p]['prec'] for p in pvals_kde_no_pretraining]

# With outliers

results_mean_final = helper_functions.unpickle(os.path.dirname(os.getcwd()) + "/results_mean_final.pkl") 
pvals_mean = np.linspace(0, 1, 19)
x_prc_mean = [results_mean_final[p]['recall'] for p in pvals_mean]
y_prc_mean = [results_mean_final[p]['prec'] for p in pvals_mean]

results_binning_final = helper_functions.unpickle(os.path.dirname(os.getcwd()) + "/results_binning_final.pkl") 
pvals_binning = np.linspace(1, 10, 19)
x_prc_binning = [results_binning_final[p]['recall'] for p in pvals_binning]
y_prc_binning = [results_binning_final[p]['prec'] for p in pvals_binning]

results_gauss_final = helper_functions.unpickle(os.path.dirname(os.getcwd()) + "/results_gauss_final.pkl") 
pvals_gauss = sorted(list(np.arange(0.001, 0.01, 0.001)) + list(np.arange(0.01, 0.1, 0.01)))
x_prc_gauss = [results_gauss_final[p]['recall'] for p in pvals_gauss]
y_prc_gauss = [results_gauss_final[p]['prec'] for p in pvals_gauss]

results_kde_final = helper_functions.unpickle(os.path.dirname(os.getcwd()) + "/results_kde_final.pkl")
pvals_kde = sorted(list(np.arange(0.001, 0.01, 0.001)) + list(np.arange(0, 0.1, 0.01)))
x_prc_kde = [results_kde_final[p]['recall'] for p in pvals_kde]
y_prc_kde = [results_kde_final[p]['prec'] for p in pvals_kde]


# In[39]:


fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8)) # figsize=(3.4, 1.7) (default: [6.4, 4.8])
fig.subplots_adjust(wspace=0.3)

axes[0].plot(x_prc_mean_no_pretraining, y_prc_mean_no_pretraining, color="#377eb8", marker="o", markersize=5, label="Mean")
# print("pr auc mean: ", auc(np.nan_to_num(x_prc_mean), np.nan_to_num(y_prc_mean)))

axes[0].plot(x_prc_binning_no_pretraining, y_prc_binning_no_pretraining, color="#ff7f00", marker="s", markersize=5, label="Binning")
# print("pr auc binning: ", auc(x_prc_binning, y_prc_binning))

axes[0].plot(x_prc_gauss_no_pretraining, y_prc_gauss_no_pretraining, color="#4daf4a", marker="^", markersize=5, label="Gaussian")
# print("pr auc gauss: ", auc(x_prc_gauss, y_prc_gauss))

axes[0].plot(x_prc_kde_no_pretraining, y_prc_kde_no_pretraining, color="#f781bf", marker="v", markersize=5, label="KDE")
# print("pr auc kde: ", auc(x_prc_kde, y_prc_kde))

axes[0].plot(np.linspace(0, 1, 20), 20*[0.039], "--", color="gray", markersize=5)
axes[0].text(0.25, -0.05, "Baseline: 0.039", color="gray")

axes[0].set_xticks(np.arange(0, 1.25, 0.25))
axes[0].set_yticks(np.arange(0, 1.25, 0.25))

axes[0].set_xlim([-0.1, 1.1])
axes[0].set_ylim([-0.1, 1.1])

#axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].annotate("(a)", xy=(0.15, 0.95))

axes[1].plot(x_prc_mean, y_prc_mean, color="#377eb8", marker="o", markersize=5, label="Mean")
# print("pr auc mean: ", auc(np.nan_to_num(x_prc_mean), np.nan_to_num(y_prc_mean)))

axes[1].plot(x_prc_binning, y_prc_binning, color="#ff7f00", marker="s", markersize=5, label="Binning")
# print("pr auc binning: ", auc(x_prc_binning, y_prc_binning))

axes[1].plot(x_prc_gauss, y_prc_gauss, color="#4daf4a", marker="^", markersize=5, label="Gaussian")
# print("pr auc gauss: ", auc(x_prc_gauss, y_prc_gauss))

axes[1].plot(x_prc_kde, y_prc_kde, color="#f781bf", marker="v", markersize=5, label="KDE")
# print("pr auc kde: ", auc(x_prc_kde, y_prc_kde))

axes[1].plot(np.linspace(0, 1, 20), 20*[0.039], "--", color="gray", markersize=5)
axes[1].text(0.25, -0.05, "Baseline: 0.039", color="gray")

axes[1].set_xticks(np.arange(0, 1.25, 0.25))
axes[1].set_yticks(np.arange(0, 1.25, 0.25))

axes[1].set_xlim([-0.1, 1.1])
axes[1].set_ylim([-0.1, 1.1])

#axes[0].set_xlabel("Recall")
#axes[1].set_ylabel("Precision")
axes[1].annotate("(b)", xy=(0.15, 0.95))

fig.text(0.5, 0.00, "Recall", ha="center")

plt.legend(loc="upper center", frameon=False, bbox_to_anchor=(-0.15, 1.2),  ncol=4, fancybox=False, shadow=False, labelspacing=0.5, columnspacing=1)

#plt.savefig("../../figs/figure_prc_condensed.pdf", dpi=200, bbox_inches="tight")

plt.show()


# In[ ]:




