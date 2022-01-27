#!/usr/bin/env python3
# Anna Ollerenshaw 2021

import pickle
import torch
import cca_core
import numpy as np
import os
import re
#from scipy import interpolate
from CKA import gram_linear, gram_rbf, cka, feature_space_linear_cka

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def import_data(model_files):
  epoch=[]
  for i in range(len(model_files)):
    epoch.append(torch.load(model_files[i]))
  return epoch
  
def cca_conv(embedding_files,keys):
  # cca convolutions change for new format
  # BATCH X NUMBER OF FRAMES X (DIMENSIONALITY * FEATURE SIZE) X CHANNEL
  results = []
  for f_index in range(len(embedding_files)-1):
    print(embedding_files[f_index], "-----", embedding_files[f_index+1])
    net1 = torch.load(embedding_files[f_index])
    net2 = torch.load(embedding_files[f_index+1])
    ## CCA
    b, c, f, d = net1[keys].shape
    b1, c1, f1, d1 = net2[keys].shape
    if (b!=b1 or f!=f1 or d!=d1):
       print("net1 b={} c={} f={} d={} net2 b={} c={} f={} d ={}".format(b,c,f,d,b1,c,f1,d1))
       print("Dimension mismatch")
       break
    print("net1 b={} c={} f={} d={}".format(b,c,f,d))
    print("net2 b={} c={} f={} d ={}".format(b1,c1,f1,d1))
    layerA = net1[keys].permute(0,2,3,1)
    layerB = net2[keys].permute(0,2,3,1)
    layerA = layerA.reshape((b*f*d,c))
    layerB = layerB.reshape((b1*f1*d1,c1))
    print("net1 ={} net2={}".format(layerA.shape,layerB.shape))
    similarity = cca_core.get_cca_similarity(layerA.cpu().T.detach().numpy(), layerB.cpu().T.detach().numpy(), epsilon=1e-10, verbose=False)
    results.append(np.mean(similarity["cca_coef1"]))
    print("cca value: ", results)
  np.save('/share/mini1/sw/spl/espresso/new_svcca/svcca/exp/svcca/results/lstm_conv_5_layers_x0.5/%s.npy' % keys,results)
  
  #results_0 = []
  #for f_index in range(len(embedding_files)-1):
  #  print(embedding_files[0], "-----", embedding_files[f_index+1])
  #  net1 = torch.load(embedding_files[0])
  #  net2 = torch.load(embedding_files[f_index+1])
  #  ## CCA
  #  #results = []
  #  b, c, f, d = net1[keys].shape
  #  b1, c1, f1, d1 = net2[keys].shape
  #  layerA = net1[keys].permute(0,2,3,1)
  #  layerB = net2[keys].permute(0,2,3,1)
  #  layerA = layerA.reshape((b*f*d,c))
  #  layerB = layerB.reshape((b1*f1*d1,c1))
  #  similarity = cca_core.get_cca_similarity(layerA.cpu().T.detach().numpy(), layerB.cpu().T.detach().numpy(), epsilon=1e-10, verbose=False)
  #  results_0.append(np.mean(similarity["cca_coef1"]))
  #  print("cca value for 0th to {}th is {} ".format(f_index+1,results_0))
  #np.save('/share/mini1/sw/spl/espresso/new_svcca/svcca/exp/svcca/results/lstm_conv_x0.5/roc_%s.npy' % keys,results_0)

def cca_lstm(embedding_files,keys):
  # cca convolutions change for new format
  # BATCH X NUMBER OF FRAMES X (DIMENSIONALITY * FEATURE SIZE)
  results = []
  for f_index in range(len(embedding_files)-1):
    print(embedding_files[f_index], "-----", embedding_files[f_index+1])
    net1 = torch.load(embedding_files[f_index])
    net2 = torch.load(embedding_files[f_index+1])
    ## CCA
    f, b, h = net1[keys].shape
    f1, b1, h1 = net2[keys].shape
    if (f!=f1 or b!=b1 or h!=h1):
       print("net1 f={} b={} h={} net2 f={} b={} h={}".format(f,b,h,f1,b1,h1))
       print("Dimension mismatch")
       break
    print("net1 f={} b={} h={}".format(f,b,h))
    print("net2 f={} b={} h={}".format(f1,b1,h1))
    layerA = net1[keys].reshape((f*b*h,1))
    layerB = net2[keys].reshape((f1*b1*h1,1))
    print("net1 ={} net2={}".format(layerA.shape,layerB.shape))
    similarity = cca_core.get_cca_similarity(layerA.cpu().T.detach().numpy(), layerB.cpu().T.detach().numpy(), epsilon=1e-10, verbose=False)
    results.append(np.mean(similarity["cca_coef1"]))
    print("cca value: ", results)
  np.save('/share/mini1/sw/spl/espresso/new_svcca/svcca/exp/svcca/results/lstm_conv_5_layers_x0.5/%s.npy' % keys,results)
  
  #results_0 = []
  #for f_index in range(len(embedding_files)-1):
  #  print(embedding_files[0], "-----", embedding_files[f_index+1])
  #  net1 = torch.load(embedding_files[0])
  #  net2 = torch.load(embedding_files[f_index+1])
    ## CCA
    #results = []
  #  f, b, h = net1[keys].shape
  #  f1, b1, h1 = net2[keys].shape
  #  if (f!=f1 or b!=b1 or h!=h1):
  #     print("net1 f={} b={} h={} net2 f={} b={} h={}".format(f,b,h,f1,b1,h1))
  #     print("Dimension mismatch")
  #     break
  #  print("net1 f={} b={} h={}".format(f,b,h))
  #  print("net2 f={} b={} h={}".format(f1,b1,h1))
  #  layerA = net1[keys].reshape((f*b*h,1))
  #  layerB = net2[keys].reshape((f1*b1*h1,1))
  #  similarity = cca_core.get_cca_similarity(layerA.cpu().T.detach().numpy(), layerB.cpu().T.detach().numpy(), epsilon=1e-10, verbose=False)
  #  results_0.append(np.mean(similarity["cca_coef1"]))
  #  print("cca value for 0th to {}th is {} ".format(f_index+1,results_0))
  #np.save('/share/mini1/sw/spl/espresso/new_svcca/svcca/exp/svcca/results/lstm_conv_x0.5/roc_%s.npy' % keys,results_0)

def cca_trans(embedding_files,subkeys):
  # cca convolutions change for new format
  # BATCH X NUMBER OF FRAMES X (DIMENSIONALITY * FEATURE SIZE)
  results = []
  for f_index in range(len(embedding_files)-1):
    print(embedding_files[f_index], "-----", embedding_files[f_index+1])
    net1 = torch.load(embedding_files[f_index])
    net2 = torch.load(embedding_files[f_index+1])
    ## CCA
    f, b, h = net1[subkeys].shape
    f1, b1, h1 = net2[subkeys].shape
    if (f!=f1 or b!=b1 or h!=h1):
       print("net1 f={} b={} h={} net2 f={} b={} h={}".format(f,b,h,f1,b1,h1))
       print("Dimension mismatch")
       break
    print("net1 f={} b={} h={}".format(f,b,h))
    print("net2 f={} b={} h={}".format(f1,b1,h1))
    layerA = net1[subkeys].reshape((f*b*h,1))
    layerB = net2[subkeys].reshape((f1*b1*h1,1))
    print("net1 ={} net2={}".format(layerA.shape,layerB.shape))
    similarity = cca_core.get_cca_similarity(layerA.cpu().T.detach().numpy(), layerB.cpu().T.detach().numpy(), epsilon=1e-10, verbose=False)
    results.append(np.mean(similarity["cca_coef1"]))
    print("cca coeff: ", results)
  np.save('/share/mini1/sw/spl/espresso/new_svcca/svcca/exp/svcca/results/transformer/%s.npy' % subkeys,results)

def cka_conv(embedding_files,keys):
  # cka convolutions change for new format
  # BATCH X NUMBER OF FRAMES X (DIMENSIONALITY * FEATURE SIZE) X CHANNEL
  results_ex = []
  results_feat = []
  for f_index in range(len(embedding_files)-1):
    print(embedding_files[f_index], "-----", embedding_files[f_index+1])
    net1 = torch.load(embedding_files[f_index])
    net2 = torch.load(embedding_files[f_index+1])
    ## CCA
    b, c, f, d = net1[keys].shape
    b1, c1, f1, d1 = net2[keys].shape
    if (b!=b1 or f!=f1 or d!=d1):
       print("net1 b={} c={} f={} d={} net2 b={} c={} f={} d ={}".format(b,c,f,d,b1,c,f1,d1))
       print("Dimension mismatch")
       break
    print("net1 b={} c={} f={} d={}".format(b,c,f,d))
    print("net2 b={} c={} f={} d ={}".format(b1,c1,f1,d1))
    layerA = net1[keys].permute(0,2,3,1)
    layerB = net2[keys].permute(0,2,3,1)
    layerA = layerA.reshape((b*f*d,c))
    layerB = layerB.reshape((b1*f1*d1,c1))
    print("net1 ={} net2={}".format(layerA.shape,layerB.shape))
    cka_from_examples = cka(gram_linear(layerA.cpu().detach().numpy()), gram_linear(layerB.cpu().detach().numpy()))
    cka_from_features = feature_space_linear_cka(layerA.cpu().detach().numpy(),layerB.cpu().detach().numpy())
    results_ex.append(cka_from_examples)
    results_feat.append(cka_from_features)
    rbf = cka(gram_rbf(layerA.cpu().detach().numpy()),gram_rbf(layerB.cpu().detach().numpy()))
    print("cka examples: ", results_ex)
    print("cka features: ", results.feat)
    print("rbf kernel: ", rbf)
  #np.save('/share/mini1/sw/spl/espresso/new_svcca/svcca/exp/svcca/results_cka/{0}/{1}.npy'.format(filename,subkeys),results)


####################################################################################
def main_fct(root_dir,keys,subkeys):
  embedding_files = []
  model_dirs = os.listdir(root_dir)
  for f in model_dirs:
    embedding_epoch_dirs = os.path.join(root_dir,f)
    embedding_files_all = os.listdir(embedding_epoch_dirs)
    for g in embedding_files_all:
      if keys in g and ".pkl" in g:
        embedding_files.append(os.path.join(embedding_epoch_dirs,g))
  embedding_files.sort(key=natural_keys)
  #embedding_epoch_layers = import_data(model_files)
  #svd_conv(net_1)
  if "conv" in keys:
      print("Convolutional analysis...")
      #cca = cca_conv(embedding_files,keys)
      cka = cka_conv(embedding_files,keys)
  elif "lstm" in keys:
      print("LSTM analysis...")
      cca = cca_lstm(embedding_files,keys)
  elif "trans" in keys:
      print("Transformer analysis...")
      cca = cca_trans(embedding_files,subkeys) 

#####################################################################################
file_name = 'lstm_conv_5_layers_x0.5'
file_path = '/share/mini1/sw/spl/espresso/new_svcca/svcca/exp/all_epoch_embeddings/{0}/'.format(file_name)
key ="conv_0"
subkey = "11.self_attn.out_proj"

main_fct(file_path,key,subkey)
