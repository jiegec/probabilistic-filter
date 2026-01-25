#!/usr/bin/env python3
"""Plot benchmark results from CSV file."""

import csv
import matplotlib.pyplot as plt
import numpy as np

# Read data from CSV
data = []
with open("benchmark_results.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

# Separate by filter type
bloom_data = [
    (float(r["bits_per_key"]), float(r["fp_rate"]))
    for r in data
    if r["filter_type"] == "Bloom"
]
cuckoo_data = [
    (float(r["bits_per_key"]), float(r["fp_rate"]))
    for r in data
    if r["filter_type"] == "Cuckoo"
]
xor_data = [
    (float(r["bits_per_key"]), float(r["fp_rate"]))
    for r in data
    if r["filter_type"] == "XOR"
]
binaryfuse3_data = [
    (float(r["bits_per_key"]), float(r["fp_rate"]))
    for r in data
    if r["filter_type"] == "BinaryFuse3"
]
binaryfuse4_data = [
    (float(r["bits_per_key"]), float(r["fp_rate"]))
    for r in data
    if r["filter_type"] == "BinaryFuse4"
]

# Sort by bits_per_key for each filter type
bloom_data.sort(key=lambda x: x[0])
cuckoo_data.sort(key=lambda x: x[0])
xor_data.sort(key=lambda x: x[0])
binaryfuse3_data.sort(key=lambda x: x[0])
binaryfuse4_data.sort(key=lambda x: x[0])

# Unzip for plotting
if bloom_data:
    bloom_x, bloom_y = zip(*bloom_data)
else:
    bloom_x, bloom_y = [], []

if cuckoo_data:
    cuckoo_x, cuckoo_y = zip(*cuckoo_data)
else:
    cuckoo_x, cuckoo_y = [], []

if xor_data:
    xor_x, xor_y = zip(*xor_data)
else:
    xor_x, xor_y = [], []

if binaryfuse3_data:
    bf3_x, bf3_y = zip(*binaryfuse3_data)
else:
    bf3_x, bf3_y = [], []

if binaryfuse4_data:
    bf4_x, bf4_y = zip(*binaryfuse4_data)
else:
    bf4_x, bf4_y = [], []

# Create plot
fig, ax = plt.subplots(figsize=(10, 7))

# Plot data points
if bloom_x:
    ax.scatter(bloom_x, bloom_y, c="blue", label="Bloom", s=100, alpha=0.7)
if cuckoo_x:
    ax.scatter(cuckoo_x, cuckoo_y, c="red", label="Cuckoo", s=100, alpha=0.7)
if xor_x:
    ax.scatter(xor_x, xor_y, c="green", label="XOR", s=100, alpha=0.7)
if bf3_x:
    ax.scatter(bf3_x, bf3_y, c="purple", label="BinaryFuse-3", s=100, alpha=0.7)
if bf4_x:
    ax.scatter(bf4_x, bf4_y, c="orange", label="BinaryFuse-4", s=100, alpha=0.7)

# Connect points with lines for each filter type
if len(bloom_x) > 1:
    ax.plot(bloom_x, bloom_y, c="blue", alpha=0.5, linestyle="--")
if len(cuckoo_x) > 1:
    ax.plot(cuckoo_x, cuckoo_y, c="red", alpha=0.5, linestyle="--")
if len(xor_x) > 1:
    ax.plot(xor_x, xor_y, c="green", alpha=0.5, linestyle="--")
if len(bf3_x) > 1:
    ax.plot(bf3_x, bf3_y, c="purple", alpha=0.5, linestyle="--")
if len(bf4_x) > 1:
    ax.plot(bf4_x, bf4_y, c="orange", alpha=0.5, linestyle="--")

# Plot theoretical curves
# Generate false positive rates from 1e-6 to 1e-1
eps_theoretical = np.logspace(-6, -1, 100)

# Theoretical formulas:
# Bloom: bits_per_key = 1.44 * log2(1/ε)
# Cuckoo: bits_per_key = (log2(1/ε) + 3) / 0.955
# XOR: bits_per_key = 1.23 * log2(1/ε)
# BinaryFuse-3: bits_per_key = 1.125 * log2(1/ε)
# BinaryFuse-4: bits_per_key = 1.075 * log2(1/ε)

log2_eps = np.log2(eps_theoretical)
theoretical_bloom = 1.44 * (-log2_eps)
theoretical_cuckoo = (-log2_eps + 3) / 0.955
theoretical_xor = 1.23 * (-log2_eps)
theoretical_bf3 = 1.125 * (-log2_eps)
theoretical_bf4 = 1.075 * (-log2_eps)

ax.plot(
    theoretical_bloom,
    eps_theoretical,
    c="blue",
    alpha=0.8,
    linestyle="-.",
    linewidth=2,
    label="Bloom (theoretical)",
)
ax.plot(
    theoretical_cuckoo,
    eps_theoretical,
    c="red",
    alpha=0.8,
    linestyle="-.",
    linewidth=2,
    label="Cuckoo (theoretical)",
)
ax.plot(
    theoretical_xor,
    eps_theoretical,
    c="green",
    alpha=0.8,
    linestyle="-.",
    linewidth=2,
    label="XOR (theoretical)",
)
ax.plot(
    theoretical_bf3,
    eps_theoretical,
    c="purple",
    alpha=0.8,
    linestyle="-.",
    linewidth=2,
    label="BinaryFuse-3 (theoretical)",
)
ax.plot(
    theoretical_bf4,
    eps_theoretical,
    c="orange",
    alpha=0.8,
    linestyle="-.",
    linewidth=2,
    label="BinaryFuse-4 (theoretical)",
)

# Set log scale for y axis
ax.set_yscale("log")
ax.set_ylim(1e-6, 1.0)

# Labels and title
ax.set_xlabel("Space Usage (bits per key)", fontsize=12)
ax.set_ylabel("False Positive Rate", fontsize=12)
ax.set_title("False Positive Rate vs Space Usage", fontsize=16)

# Grid and legend
ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig("fp_rate.png", dpi=150)
print("Plot saved to fp_rate.png")
