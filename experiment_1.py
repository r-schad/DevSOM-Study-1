# -*- coding: utf-8 -*-
"""Experiment_1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PSji7vlGlglXYlneTmjTxcmaZ4IqA3ms

# **Experiment 1:** 
### *Train a 24x24 SOFM on MNIST-b1 for 3m epochs.*

### Setup and Import Data
"""

# from google.colab import drive
# drive.mount("/content/drive")

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/Colab_Notebooks/Research/DevSOM-Study-1/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import seaborn as sns

from load_MNIST import load_mnist_data
from SOFM import SOFM



# # importing takes 2 attempts for some reason
# try:
#   from load_MNIST_data import load_mnist_data
# except:
#   from load_MNIST_data import load_mnist_data

train_img_arr, train_label_arr, test_img_arr, test_label_arr = load_mnist_data("..\\..\\MNIST")

# using black numbers on white background
train_arr_full = 1 - (train_img_arr.reshape(train_img_arr.shape[0], train_img_arr.shape[1] * train_img_arr.shape[2]) / 255)
test_arr_full = 1 - (test_img_arr.reshape(test_img_arr.shape[0], test_img_arr.shape[1] * test_img_arr.shape[2]) / 255)

SIZE_OF_TRAINING_SET = 5000
num_examples = SIZE_OF_TRAINING_SET // 10

# get random balanced subset of full MNIST training dataset 
train_arr = np.empty((SIZE_OF_TRAINING_SET,784))
for i in range(10):
  idxs = np.where(train_label_arr==i)[0]
  rand_idxs = np.random.choice(idxs, num_examples)
  train_arr[i*num_examples:(i*num_examples)+num_examples] = train_arr_full[rand_idxs]

np.random.shuffle(train_arr)

### HYPERPARAMETERS ###
NETWORK_D1 = 24
NETWORK_D2 = 24
LEARNING_RATE = 0.025 # learning rate
STARTING_NEIGHBORHOOD_SIZE = 3.0
NEIGHBORHOOD_DECAY_RATE = 25
NUM_EPOCHS = 1 # CHANGEME

sofm = SOFM(d1=NETWORK_D1, d2=NETWORK_D2, num_features=784, sigma_o=STARTING_NEIGHBORHOOD_SIZE, tau_N=NEIGHBORHOOD_DECAY_RATE)

train_start_time = datetime.now()
sofm.train(train_arr, NUM_EPOCHS, LEARNING_RATE)
train_end_time = datetime.now()

# create directory for results
print('Creating results directory...')
if not os.path.isdir('results'):
  os.mkdir('results')

date_str = datetime.today().strftime('%Y-%m-%d')
new_dir = 'results\\' + date_str
i = 0
while os.path.isdir(new_dir):
  i += 1
  new_dir = 'results\\' + date_str + ' (' + str(i) + ')'
os.mkdir(new_dir)


# get win percentage of each neuron by class (using training data) - needed for classification metric
train_win_counts, train_win_percentages = sofm.calc_win_percentages(train_arr_full, train_label_arr)

# get total number of wins among all classes for each neuron
total_wins_train = np.sum(train_win_counts, axis=0)

# assign label to each neuron based on for which class it won most often 
# (i.e. which win_percentage for each class is the highest for each neuron)
# if a neuron did not win for any examples, assign a random class label - need this so we can compute classification metric
neuron_labels = np.where(total_wins_train != 0, np.argmax(train_win_percentages, axis=0), np.rint(9 * np.random.rand(24, 24))).astype(np.int32) 

# plot neuron class tuning labels
sofm.visualize_neuron_classes(neuron_labels, new_dir + '\\class_tuning')

# plot weights of all neurons
sofm.visualize_weights(new_dir + '\\weights')

# get win percentage of each neuron by class (using testing data) - needed for entropy and NCL metric
test_win_counts, test_win_percentages = sofm.calc_win_percentages(test_arr_full, test_label_arr)

# calculate entropy of each neuron
entropy = sofm.calc_entropy(test_win_counts)

# create entropy plot
sofm.create_entropy_plot(entropy, new_dir + '\\entropy')

# classification metric
hit_rate = sofm.calc_classification_metric(test_arr_full, test_label_arr, neuron_labels, 
                                          plot_conf_matrix=True, filename=new_dir + '\\confusion_matrix')

# normalized category localization (NCL) metric
ncl_score = sofm.calc_ncl_metric(test_win_counts, test_win_percentages)

# plot win percentages on a heatmap for each class
sofm.plot_win_percentages(test_win_percentages, new_dir + '\\win_percentages')

# write out stats.txt
sofm.write_stats(train_start_time, train_end_time, NUM_EPOCHS, len(train_arr), LEARNING_RATE, new_dir + '\\stats.txt', ncl_score=ncl_score, classification_score=hit_rate)
