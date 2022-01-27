#!/usr/bin/env python3
# Anna Ollerenshaw 2021


from pwcca import compute_pwcca
import pickle
import torch
import numpy as np
import os
import re
from scipy import interpolate

def atoi(text):
  return int(text) if text.isdigit() else text

def natural_keys(text):
  return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def import_data(model_files):
  epoch=[]
  for i in range(len(model_files)):
    epoch.append(torch.load(model_files[i]))
  return epoch


def pwcca(embedding_files):
  # cka convolutions change for new format
  # BATCH X NUMBER OF FRAMES X (DIMENSIONALITY * FEATURE SIZE) X CHANNEL
  results = []
  for f_index in range(len(embedding_files)-1):
    print(embedding_files[f_index], "-----", embedding_files[f_index+1])
    net1 = torch.load(embedding_files[f_index])
    net2 = torch.load(embedding_files[f_index+1])
    key1,values=list(net1.items())[0]
    key2,values=list(net2.items())[0]
    if "conv" in key1 and "conv" in key2: 
      layerA = net1[key1]
      layerB = net2[key2]
      print("interpolation of conv layers")
      layerAn=layerA.cpu().numpy()[:,:,:,:] # convert to np
      layerBn=layerB.cpu().numpy()[:,:,:,:]
      pwcca_m=interpolation(layerAn,layerBn)
      print("pwcca convolutions: ",pwcca_m)
      results.append(pwcca_m)
    elif "lstm" in key1 and "lstm" in key2:
      f, b, h = net1[key1].shape
      print("net1 lstm f={} b={} h={}".format(f,b,h))
      layerA = net1[key1].reshape((f*b*h,1))
      layerA = layerA.cpu().numpy()[:,:]
      f1, b1, h1 = net2[key2].shape
      print("net2 lstm f={} b={} h={}".format(f1,b1,h1))
      layerB = net2[key2].reshape((f1*b1*h1,1))
      layerB = layerB.cpu().numpy()[:,:]
      pwcca_l, w, _ = compute_pwcca(layerA.T,layerB.T, epsilon=1e-10)
      print("pwcca lstm layers: ",pwcca_l)
      results.append(pwcca_l)
    else:
      print("n/a for pwcca")
  np.save('/share/mini1/sw/spl/espresso/new_svcca/svcca/exp/svcca/results/pwcca/{0}.npy'.format(file_name),results)

def interpolation(net_1,net_2):
  net_1_interp = []
  net_2_interp = []
  print("net1: ",net_1.shape,"net2: ",net_2.shape)
  if net_1.shape > net_2.shape:
    num_d , _, w, h = net_1.shape
    num_c = net_2.shape[1]
    net_2_interp = np.zeros((num_d, num_c, w, h))
    print("interpolating from: ", net_2.shape,"to: ",net_2_interp.shape)
    for d in range(num_d):
      for c in range(num_c):
        # form interpolation function
        idxs1 = np.linspace(0, net_2.shape[2],
                            net_2.shape[2],
                            endpoint=False)
        idxs2 = np.linspace(0, net_2.shape[3],
                            net_2.shape[3],
                            endpoint=False)
        arr = net_2[d,c,:,:]
        f_interp = interpolate.interp2d(idxs2, idxs1, arr)
        large_idxs1 = np.linspace(0,net_2.shape[2],net_1.shape[2],endpoint=False)
        large_idxs2 = np.linspace(0, net_2.shape[3],net_1.shape[3],endpoint=False)
        net_2_interp[d, c, :, :] = f_interp(large_idxs2, large_idxs1)
    b, c, f, d = net_1.shape
    b1, c1, f1, d1 = net_2_interp.shape
    net_1 = net_1.reshape(b*f*d,c)
    net_2 = net_2_interp.reshape(b1*f1*d1,c)
    print("net_1: ",net_1.shape,"net_2: ",net_2.shape)
    pwcca_mean, w, _ = compute_pwcca(net_1.T,net_2.T, epsilon=1e-10)
  elif net_2.shape > net_1.shape:
    num_d , i, w, h = net_2.shape
    num_c = net_1.shape[2]
    net_1_interp = np.zeros((num_d, i, num_c, h))
    print("interpolating from: ", net_1.shape,"to: ",net_1_interp.shape)
    for d in range(num_d):
      for c in range(num_c):
        # form interpolation function
        idxs1 = np.linspace(0, net_1.shape[1],
                            net_1.shape[1],
                            endpoint=False)
        idxs2 = np.linspace(0, net_1.shape[3],
                            net_1.shape[3],
                            endpoint=False)
        arr = net_1[d,:,c,:]
        f_interp = interpolate.interp2d(idxs2, idxs1, arr)
        large_idxs1 = np.linspace(0,net_1.shape[1],net_2.shape[1],endpoint=False)
        large_idxs2 = np.linspace(0, net_1.shape[3],net_2.shape[3],endpoint=False)
        net_1_interp[d, :, c, :] = f_interp(large_idxs2, large_idxs1)
    b, c, f, d = net_1_interp.shape
    b1, c1, f1, d1 = net_2.shape
    net_1 = net_1_interp.reshape(b*f*d,c)
    net_2 = net_2.reshape(b1*f1*d1,c)
    print("net_1: ",net_1.shape,"net_2: ",net_2.shape)
    pwcca_mean, w, _ = compute_pwcca(net_1.T,net_2.T, epsilon=1e-10)
  else:
    b, c, f, d = net_1.shape
    b1, c1, f1, d1 = net_2.shape
    net_1 = net_1.reshape(b*f*d,c)
    net_2 = net_2.reshape(b1*f1*d1,c1)
    print("net_1: ",net_1.shape,"net_2: ",net_2.shape)
    pwcca_mean, w, _ = compute_pwcca(net_1.T,net_2.T, epsilon=1e-10)
  return pwcca_mean
#####################################################################################
### task - layer comparison net1 = conv_0 for epoch 35, compared to conv_1 epoch 35

def main_fct(root_dir,keys):
  embedding_files = []
  model_dirs = os.listdir(root_dir)
  for f in model_dirs:
    embedding_epoch_dirs = os.path.join(root_dir,f)
    embedding_files_all = os.listdir(embedding_epoch_dirs)
    for g in embedding_files_all:
      if keys in embedding_epoch_dirs and ".pkl" in g: #might need to add if lstm/conv
        embedding_files.append(os.path.join(embedding_epoch_dirs,g))
  embedding_files.sort(key=natural_keys)
  pwcca_results = pwcca(embedding_files)


file_name = 'transformer'
file_path = '/share/mini1/sw/spl/espresso/new_svcca/svcca/exp/all_epoch_embeddings/{0}/'.format(file_name)
key = '35.'

main_fct(file_path,key)

