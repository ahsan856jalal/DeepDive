#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:49:30 2025

@author: ahsanjalal
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy, pearsonr
from pyemd import emd

# File path
file_path = "combined_depth.csv"

# Load the CSV file
df = pd.read_csv(file_path)

# Ensure the columns 'Length' and 'est_length' exist in the DataFrame
if 'Length' in df.columns and 'est_length' in df.columns:
    
    # Divide each column by 10
    df['Length'] = df['Length'] 
    df['est_length'] = df['est_length'] 

    # Pearson Correlation Coefficient
    pearson_corr, _ = pearsonr(df['Length'], df['est_length'])

    # Mean Squared Error (MSE)
    mse = np.mean((df['Length'] - df['est_length']) ** 2)

    # Root Mean Squared Error (RMSE)
    # rmse = np.sqrt(mse)

    # Root Mean Squared Deviation (RMSD)
    rmsd = np.sqrt(np.mean((df['Length'] - df['est_length'])**2))
    mean_length = np.mean(df['Length'])
    rmsd_percentage = (rmsd / mean_length) * 100

    # Kullback-Leibler (KL) Divergence
    # We need probability distributions, so normalize the values first
    length_hist, bins = np.histogram(df['Length'], bins=30, density=True)
    est_length_hist, _ = np.histogram(df['est_length'], bins=bins, density=True)
    
    # Avoiding log(0) by adding a small epsilon
    epsilon = 1e-10
    kl_divergence = entropy(length_hist + epsilon, est_length_hist + epsilon)

    # Earth Mover's Distance (EMD)
    # Normalize the histograms for EMD calculation
    length_hist_norm = length_hist / np.sum(length_hist)
    est_length_hist_norm = est_length_hist / np.sum(est_length_hist)
    
    # Calculate the distance matrix between bin edges
    bin_centers = (bins[:-1] + bins[1:]) / 2
    distance_matrix = np.abs(bin_centers[:, None] - bin_centers)  # Absolute difference between bin centers

    # Calculate EMD using pyemd package
    emd_value = emd(length_hist_norm, est_length_hist_norm, distance_matrix)

    # Print results
    print(f"Pearson Correlation Coefficient: {pearson_corr:.4f}")
    print(f"RMSD: {rmsd_percentage:.4f}%")
    print(f"KL Divergence: {kl_divergence:.4f}")
    print(f"Earth Mover's Distance: {emd_value:.4f}")

else:
    print("Columns 'Length' and/or 'est_length' not found in the CSV file.")
