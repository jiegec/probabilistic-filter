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

# Sort by bits_per_key for each filter type
bloom_data.sort(key=lambda x: x[0])
cuckoo_data.sort(key=lambda x: x[0])
xor_data.sort(key=lambda x: x[0])

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

# Create plot
fig, ax = plt.subplots(figsize=(10, 7))

# Plot data points
if bloom_x:
    ax.scatter(bloom_x, bloom_y, c="blue", label="Bloom", s=100, alpha=0.7)
if cuckoo_x:
    ax.scatter(cuckoo_x, cuckoo_y, c="red", label="Cuckoo", s=100, alpha=0.7)
if xor_x:
    ax.scatter(xor_x, xor_y, c="green", label="XOR", s=100, alpha=0.7)

# Connect points with lines for each filter type
if len(bloom_x) > 1:
    ax.plot(bloom_x, bloom_y, c="blue", alpha=0.5, linestyle="--")
if len(cuckoo_x) > 1:
    ax.plot(cuckoo_x, cuckoo_y, c="red", alpha=0.5, linestyle="--")
if len(xor_x) > 1:
    ax.plot(xor_x, xor_y, c="green", alpha=0.5, linestyle="--")

# Set log scale for y axis
ax.set_yscale("log")
ax.set_ylim(1e-5, 1.0)

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
