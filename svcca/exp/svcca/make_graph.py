#!/usr/bin/env python3
# Anna Ollerenshaw 2021


from matplotlib import pyplot as plt

def cca_plot_helper(arr1,xlabel,ylabel):
  plt.plot(arr1,lw=2.0,color='k')
  #plt.plot(arr2,lw=2.0,color='r')
  #plt.plot(arr3,lw=2.0,color='m')
  #plt.plot(arr4,lw=2.0,color='y')
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.ylim(0,1)
  plt.grid()

