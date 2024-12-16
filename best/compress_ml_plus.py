#!/usr/bin/env python
import struct
import wave
import numpy as np
import scipy as sp
import sys
from scipy.fftpack import dct, idct
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# Parameter settings
BLOCK_SIZE = 2048  # Size of each block
# N_COEFF = int(sys.argv[1]) if len(sys.argv) > 1 else 200
# N_COEFF = 1700

# Open input audio file
fin = wave.open('step.wav', 'r')
(nchannels, sampwidth, framerate, nframes, comptype, compname) = fin.getparams()
inbytes = fin.readframes(nframes)
fin.close()

# Convert byte stream to float array (normalized to [-1,1] range)
samplesint = np.array([struct.unpack('<h', inbytes[2*i:2*i+2])[0] for i in range(nframes)])
samples = samplesint.astype(np.float32) / (2**15)

# Process in blocks
num_blocks = int(np.ceil(len(samples) / BLOCK_SIZE))
# For the part that is less than one block, pad with zeros
pad_length = num_blocks * BLOCK_SIZE - len(samples)
if pad_length > 0:
    samples = np.concatenate((samples, np.zeros(pad_length, dtype=np.float32)))

# Perform DCT on each block
blocks = samples.reshape(num_blocks, BLOCK_SIZE)
dct_blocks = dct(blocks, type=2, norm='ortho', axis=1)  # Perform DCT on each block row

# Assume we already have the original data after block processing
original_blocks = samples.reshape(num_blocks, BLOCK_SIZE)

# Initialize list to store the best number of coefficients
best_coeffs_per_block = []

# Define weights
# mse_weight = 0.7
# compression_weight = 0.3

# 2048
mse_weight = 0.99928
compression_weight = 0.00072

# 512
# mse_weight = 0.99925
# compression_weight = 0.00075

# Traverse each block
for block_idx, block in enumerate(dct_blocks):
    min_weighted_score = float('inf')
    best_num_coeffs = 0
    
    # Try different numbers of coefficients
    for num_coeffs in range(1, BLOCK_SIZE + 1):
        # print(f"Block {block_idx}: Trying {num_coeffs} coefficients")
        # Extract coefficients and reconstruct block
        truncated_block = np.zeros_like(block)
        truncated_block[:num_coeffs] = block[:num_coeffs]
        reconstructed_block = idct(truncated_block, type=2, norm='ortho', axis=0)
        
        # Calculate MSE
        mse = np.mean((original_blocks[block_idx] - reconstructed_block) ** 2)
        
        # Calculate compression ratio
        compression_ratio = num_coeffs / BLOCK_SIZE
        
        # Calculate weighted score
        # weighted_score = mse_weight * mse + compression_weight * compression_ratio
        # Example: Nonlinear weighting
        weighted_score = mse_weight * np.log1p(mse) + compression_weight * np.log1p(compression_ratio)
        
        # Update the best number of coefficients
        if weighted_score < min_weighted_score:
            min_weighted_score = weighted_score
            best_num_coeffs = num_coeffs
    
    best_coeffs_per_block.append(best_num_coeffs)

# Generate features and target variables
features = []
targets = best_coeffs_per_block

for block in dct_blocks:
    # Calculate features, such as statistical features of DCT coefficients
    mean = np.mean(block)
    std = np.std(block)
    max_val = np.max(block)
    min_val = np.min(block)
    energy = np.sum(np.square(block))
    
    # Add to feature list
    features.append([mean, std, max_val, min_val, energy])

# Convert to numpy array
features = np.array(features)
targets = np.array(targets)

# Train model
# model = RandomForestRegressor(n_estimators=100, max_depth=10)
# model.fit(features, targets)

# Use multi-layer perceptron regressor
# model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, learning_rate_init=0.01)
# model.fit(features, targets)

from xgboost import XGBRegressor

# Use XGBoost regressor
model = XGBRegressor(n_estimators=200, max_depth=10, learning_rate=0.1, objective='reg:squarederror')
model.fit(features, targets)

# # Calculate the number of frequency coefficients required for each block
# coeffs_per_block = [calculate_energy_threshold(block) for block in dct_blocks]
# print("coeffs_per_block[75]:",coeffs_per_block[75])
# print("coeffs_per_block[98]:",coeffs_per_block[98])
# print("coeffs_per_block[99]:",coeffs_per_block[99])
# print("coeffs_per_block[100]:",coeffs_per_block[100])
# print("coeffs_per_block[71]:",coeffs_per_block[71])
# print("coeffs_per_block[149]:",coeffs_per_block[149])
# # coeffs_per_block = [429] * num_blocks
# Use the model to predict in the compression process
predicted_coeffs_per_block = model.predict(features).astype(int)

# Use the predicted number of coefficients to truncate
truncated_dct_blocks = [block[:n] for block, n in zip(dct_blocks, predicted_coeffs_per_block)]

def print_bits(value, bits):
    """
    Print each bit of a number
    
    Parameters:
    value: The number to print
    bits: The number of bits
    """
    print(value,end=' : ')
    for i in range(bits-1, -1, -1):
        bit = (value >> i) & 1
        print(bit, end='')
    print()  # New line


def quantize_blocks(blocks, bits_per_block):
    assert len(bits_per_block) == len(blocks), "The length of the bit array must be equal to the number of blocks"
    
    quant_scales = np.zeros(len(bits_per_block), dtype=np.float32)
    block_byte_streams = []  # Store byte streams for each block
    block_padding_bits = []  # Store padding bits for each block
    
    for i, (block, bits) in enumerate(zip(blocks, bits_per_block)):
        max_val = np.max(np.abs(block))
        if max_val == 0:
            max_val = 1e-9
        
        # Calculate quantization scale
        quant_scale = (2**(bits-1) - 1) / max_val
        quant_scales[i] = quant_scale
        
        # Calculate the total number of bits and bytes required for the current block
        block_total_bits = bits * len(block)
        block_total_bytes = (block_total_bits + 7) // 8
        block_padding = (8 - (block_total_bits % 8)) % 8
        
        # Create byte stream for the current block
        block_stream = bytearray(block_total_bytes)
        current_bit_pos = 0
        
        # Quantize and store values in the current block
        quantized_values = np.round(block * quant_scale).astype(np.int32)
        
        for value in quantized_values:
            # Ensure the value is within the valid range
            value = max(-(1 << (bits-1)), min(value, (1 << (bits-1)) - 1))
            if value < 0:
                value = value + (1 << bits)
            
            byte_index = current_bit_pos // 8
            bit_offset = current_bit_pos % 8
            
            # Initialize remaining bits and current value
            remaining_value = value
            bits_left = bits
            
            while bits_left > 0:
                # Calculate the number of remaining bits that can be written to the current byte
                remaining_bits = 8 - (current_bit_pos % 8)
                # Determine the number of bits to write
                bits_to_write = min(remaining_bits, bits_left)
                
                # Prepare the value to write
                if bits_left > bits_to_write:
                    # If there are more bits to write, take the highest bits_to_write bits
                    write_value = (remaining_value >> (bits_left - bits_to_write))
                else:
                    # If this is the last time to write, take all remaining bits
                    write_value = remaining_value
                
                # Write the value
                byte_index = current_bit_pos // 8
                block_stream[byte_index] |= (write_value & ((1 << bits_to_write) - 1)) << (remaining_bits - bits_to_write)
                
                # Update remaining value and bits
                remaining_value &= ((1 << (bits_left - bits_to_write)) - 1)
                bits_left -= bits_to_write
                current_bit_pos += bits_to_write

        
        block_byte_streams.append(block_stream)
        block_padding_bits.append(block_padding)
    
    return block_byte_streams, quant_scales, block_padding_bits

# Set all blocks to 11 bits
bits_per_block = np.full(num_blocks, 10)
block_byte_streams, quant_scales, block_padding_bits = quantize_blocks(truncated_dct_blocks, bits_per_block)

print(best_coeffs_per_block)
print("coeffs_per_block[75]:",best_coeffs_per_block[75])
print("coeffs_per_block[98]:",best_coeffs_per_block[98])
print("coeffs_per_block[99]:",best_coeffs_per_block[99])
print("coeffs_per_block[100]:",best_coeffs_per_block[100])
print("coeffs_per_block[71]:",best_coeffs_per_block[71])
print("coeffs_per_block[149]:",best_coeffs_per_block[149])


print("best_coeffs_per_block:",best_coeffs_per_block)
# Find the indices of the five largest values
largest_indices = np.argsort(best_coeffs_per_block)[-5:]

# Print the largest values and their indices
for index in largest_indices:
    print(f"Index: {index}, Value: {best_coeffs_per_block[index]}")

# Modify the file writing part
fout = open('compressed', 'wb')
fout.write(struct.pack('<i', BLOCK_SIZE))
fout.write(struct.pack('<i', num_blocks))
fout.write(struct.pack('<i', pad_length))
fout.write(struct.pack('<i', framerate))

# Write the number of coefficients for each block
for coeff_count in best_coeffs_per_block:
    fout.write(struct.pack('<h', coeff_count))

# Write the quantization bits for each block (now all are 11)
for bits in bits_per_block:
    fout.write(struct.pack('<B', bits))

# Write the quantization scales for each block
for scale in quant_scales:
    fout.write(struct.pack('<f', scale))

# Write the padding bits for each block
for padding in block_padding_bits:
    fout.write(struct.pack('<B', padding))

# Write the compressed data (write each block separately)
for block_stream in block_byte_streams:
    fout.write(block_stream)

fout.close()