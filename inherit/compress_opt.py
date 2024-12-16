#!/usr/bin/env python
import struct
import wave
import numpy as np
import scipy as sp
import sys
from scipy.fftpack import dct

# Parameter settings
BLOCK_SIZE = 2048  # Size of each block
# N_COEFF = int(sys.argv[1]) if len(sys.argv) > 1 else 200
# N_COEFF = 1700

# Open input audio file
fin = wave.open('step.wav', 'r')
(nchannels, sampwidth, framerate, nframes, comptype, compname) = fin.getparams()
inbytes = fin.readframes(nframes)
fin.close()

# Convert byte stream to float array (normalized to [-1,1])
samplesint = np.array([struct.unpack('<h', inbytes[2*i:2*i+2])[0] for i in range(nframes)])
samples = samplesint.astype(np.float32) / (2**15)

# Process in blocks
# For the part that is less than one block, pad with zeros
num_blocks = int(np.ceil(len(samples) / BLOCK_SIZE))
# 对于不足一个块的部分进行补零
pad_length = num_blocks * BLOCK_SIZE - len(samples)
if pad_length > 0:
    samples = np.concatenate((samples, np.zeros(pad_length, dtype=np.float32)))

# Perform DCT on each block
blocks = samples.reshape(num_blocks, BLOCK_SIZE)
dct_blocks = dct(blocks, type=2, norm='ortho', axis=1)  # 对每个块行进行DCT

def calculate_energy_threshold(dct_block, energy_threshold=0.995):
    """
    Calculate the spectral energy of each block and select the number of frequency coefficients that reach the energy threshold.
    
    Parameters:
    dct_block: DCT transformed block
    energy_threshold: Energy threshold percentage (0-1)
    
    Returns:
    Number of selected frequency coefficients
    """
    # Calculate the energy of each frequency coefficient
    energy = np.square(dct_block)
    total_energy = np.sum(energy)
    
    # Calculate cumulative energy
    cumulative_energy = np.cumsum(energy)
    
    # Find the number of frequency coefficients that reach the cumulative energy threshold
    num_coeffs = np.searchsorted(cumulative_energy, energy_threshold * total_energy) + 1
    
    return num_coeffs

# Calculate the required number of frequency coefficients for each block
coeffs_per_block = [calculate_energy_threshold(block) for block in dct_blocks]
coeffs_per_block = [429] * num_blocks

# Process command line arguments
# First parameter: index of the block to modify
# Second parameter: new number of coefficients
if len(sys.argv) > 2:
    block_index = int(sys.argv[1])  # 第一个参数：要修改的块的索引
    new_coeff_count = int(sys.argv[2])  # 第二个参数：新的系数数量
    if 0 <= block_index < num_blocks:  # 确保索引在有效范围内
        coeffs_per_block[block_index] = new_coeff_count

# Truncate the first coeffs_per_block[i] coefficients of each block
truncated_dct_blocks = [block[:n] for block, n in zip(dct_blocks, coeffs_per_block)]

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
    print()  # 换行


def quantize_blocks(blocks, bits_per_block):
    assert len(bits_per_block) == len(blocks), "The length of the bits array must equal the number of blocks"
    
    quant_scales = np.zeros(len(bits_per_block), dtype=np.float32)
    block_byte_streams = []  # 存储每个块的字节流
    block_padding_bits = []  # 存储每个块的填充位数
    
    for i, (block, bits) in enumerate(zip(blocks, bits_per_block)):
        max_val = np.max(np.abs(block))
        if max_val == 0:
            max_val = 1e-9
        
        # Calculate quantization scale
        quant_scale = (2**(bits-1) - 1) / max_val
        quant_scales[i] = quant_scale
        
        # Calculate the total number of bits and bytes for the current block
        block_total_bits = bits * len(block)
        block_total_bytes = (block_total_bits + 7) // 8
        block_padding = (8 - (block_total_bits % 8)) % 8
        
        # Create the byte stream for the current block
        block_stream = bytearray(block_total_bytes)
        current_bit_pos = 0
        
        # Quantize and store the current block's values
        quantized_values = np.round(block * quant_scale).astype(np.int32)
        
        for value in quantized_values:
            # Ensure the value is within the valid range
            value = max(-(1 << (bits-1)), min(value, (1 << (bits-1)) - 1))
            if value < 0:
                value = value + (1 << bits)
            
            byte_index = current_bit_pos // 8
            bit_offset = current_bit_pos % 8
            
            # Initialize the remaining bits and current value
            remaining_value = value
            bits_left = bits
            
            while bits_left > 0:
                # Calculate the remaining bits that can be written to the current byte
                remaining_bits = 8 - (current_bit_pos % 8)
                # Determine the number of bits to write
                bits_to_write = min(remaining_bits, bits_left)
                
                # Prepare the value to write
                if bits_left > bits_to_write:
                    # If there are more bits to write, take the highest bits_to_write bits
                    write_value = (remaining_value >> (bits_left - bits_to_write))
                else:
                    # If this is the last time to write, take all remaining
                    write_value = remaining_value
                
                # Write the value
                byte_index = current_bit_pos // 8
                block_stream[byte_index] |= (write_value & ((1 << bits_to_write) - 1)) << (remaining_bits - bits_to_write)
                
                # Update the remaining value and bits
                remaining_value &= ((1 << (bits_left - bits_to_write)) - 1)
                bits_left -= bits_to_write
                current_bit_pos += bits_to_write

        
        block_byte_streams.append(block_stream)
        block_padding_bits.append(block_padding)
    
    return block_byte_streams, quant_scales, block_padding_bits

# Set all blocks to 11 bits
bits_per_block = np.full(num_blocks, 10)
block_byte_streams, quant_scales, block_padding_bits = quantize_blocks(truncated_dct_blocks, bits_per_block)

# print(coeffs_per_block)

# Modify the part that writes to the file
fout = open('compressed', 'wb')
fout.write(struct.pack('<i', BLOCK_SIZE))
fout.write(struct.pack('<i', num_blocks))
fout.write(struct.pack('<i', pad_length))
fout.write(struct.pack('<i', framerate))

# Write the number of coefficients for each block
for coeff_count in coeffs_per_block:
    fout.write(struct.pack('<i', coeff_count))

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