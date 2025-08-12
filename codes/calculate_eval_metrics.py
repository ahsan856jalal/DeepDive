#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate metrics between Length and est_length columns of a CSV file.

@author: ahsan jalal
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy, pearsonr
from pyemd import emd
import argparse
import sys

# ------------------------------
# Argument parsing
# ------------------------------
parser = argparse.ArgumentParser(description="Evaluate length estimation metrics from CSV.")
parser.add_argument("--filepath", help="Path to the CSV file")
args = parser.parse_args()

file_path = args.filepath

# ------------------------------
# Read CSV
# ------------------------------
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"❌ File not found: {file_path}")
    sys.exit(1)

# ------------------------------
# Check required columns
# ------------------------------
if 'Length' in df.columns and 'est_length' in df.columns:
    # Pearson Correlation Coefficient
    pearson_corr, _ = pearsonr(df['Length'], df['est_length'])

    # RMSD (percentage)
    rmsd = np.sqrt(np.mean((df['Length'] - df['est_length'])**2))
    mean_length = np.mean(df['Length'])
    rmsd_percentage = (rmsd / mean_length) * 100

    # KL Divergence
    length_hist, bins = np.histogram(df['Length'], bins=30, density=True)
    est_length_hist, _ = np.histogram(df['est_length'], bins=bins, density=True)
    epsilon = 1e-10
    kl_divergence = entropy(length_hist + epsilon, est_length_hist + epsilon)

    # Earth Mover's Distance
    length_hist_norm = length_hist / np.sum(length_hist)
    est_length_hist_norm = est_length_hist / np.sum(est_length_hist)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    distance_matrix = np.abs(bin_centers[:, None] - bin_centers)
    emd_value = emd(length_hist_norm, est_length_hist_norm, distance_matrix)

    # Output results
    print(f"📊 Pearson Correlation Coefficient: {pearson_corr:.4f}")
    print(f"📊 RMSD: {rmsd_percentage:.4f}%")
    print(f"📊 KL Divergence: {kl_divergence:.4f}")
    print(f"📊 Earth Mover's Distance: {emd_value:.4f}")

else:
    print("❌ Columns 'Length' and/or 'est_length' not found in the CSV file.")
