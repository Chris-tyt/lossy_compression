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
# You can modify the number of coefficients for specific blocks as needed
# For example: N_COEFFS[-5:] = [100] * 5  # Use only 100 coefficients for the last 5 blocks

# Open input audio file
fin = wave.open('step.wav', 'r')
(nchannels, sampwidth, framerate, nframes, comptype, compname) = fin.getparams()
inbytes = fin.readframes(nframes)
fin.close()

# Convert byte stream to float array (normalized to [-1,1])
samplesint = np.array([struct.unpack('<h', inbytes[2*i:2*i+2])[0] for i in range(nframes)])
samples = samplesint.astype(np.float32) / (2**15)

# Block processing
num_blocks = int(np.ceil(len(samples) / BLOCK_SIZE))
N_COEFFS = [429] * num_blocks  # Initialize each block to use 429 coefficients
# Pad with zeros for the part that is less than one block
pad_length = num_blocks * BLOCK_SIZE - len(samples)
if pad_length > 0:
    samples = np.concatenate((samples, np.zeros(pad_length, dtype=np.float32)))

# Perform DCT on each block
blocks = samples.reshape(num_blocks, BLOCK_SIZE)
dct_blocks = dct(blocks, type=2, norm='ortho', axis=1)  # Perform DCT on each block row

# Extract the first N_COEFF coefficients (low-frequency components)
print(dct_blocks.shape)
truncated_dct_blocks = np.zeros((num_blocks, max(N_COEFFS)))
for i in range(num_blocks):
    truncated_dct_blocks[i, :N_COEFFS[i]] = dct_blocks[i, :N_COEFFS[i]]
print(truncated_dct_blocks.shape)

# Example usage:
# print_bits(10, 4)  # Output: 1010
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
    assert len(bits_per_block) == blocks.shape[0], "The length of the bits array must equal the number of blocks"
    
    quant_scales = np.zeros(len(bits_per_block), dtype=np.float32)
    block_byte_streams = []  # Store the byte streams of each block
    block_padding_bits = []  # Store the padding bits of each block
    
    for i, (block, bits) in enumerate(zip(blocks, bits_per_block)):
        max_val = np.max(np.abs(block))
        if max_val == 0:
            max_val = 1e-9
        
        # Calculate quantization scale
        quant_scale = (2**(bits-1) - 1) / max_val
        quant_scales[i] = quant_scale
        
        # Calculate the total bits and bytes needed for the current block
        block_total_bits = bits * block.shape[0]
        block_total_bytes = (block_total_bits + 7) // 8
        block_padding = (8 - (block_total_bits % 8)) % 8
        
        # Create the byte stream for the current block
        block_stream = bytearray(block_total_bytes)
        current_bit_pos = 0
        
        # Quantize and store the values of the current block
        quantized_values = np.round(block * quant_scale).astype(np.int32)
        
        for value in quantized_values:
            # Ensure the value is within the valid range
            value = max(-(1 << (bits-1)), min(value, (1 << (bits-1)) - 1))
            if value < 0:
                value = value + (1 << bits)
            
            byte_index = current_bit_pos // 8
            bit_offset = current_bit_pos % 8
            
            # Initialize remaining bits to write and current value
            remaining_value = value
            bits_left = bits
            
            while bits_left > 0:
                # Calculate the remaining writable bits in the current byte
                remaining_bits = 8 - (current_bit_pos % 8)
                # Determine the number of bits to write this time
                bits_to_write = min(remaining_bits, bits_left)
                
                # Prepare the value to write
                if bits_left > bits_to_write:
                    # If there are more bits to write, take the highest bits_to_write bits
                    write_value = (remaining_value >> (bits_left - bits_to_write))
                else:
                    # If this is the last write, take all remaining bits
                    write_value = remaining_value
                
                # Write the value
                byte_index = current_bit_pos // 8
                block_stream[byte_index] |= (write_value & ((1 << bits_to_write) - 1)) << (remaining_bits - bits_to_write)
                
                # Update remaining value and bits
                remaining_value &= ((1 << (bits_left - bits_to_write)) - 1)
                bits_left -= bits_to_write
                current_bit_pos += bits_to_write
        
        # Print the binary representation of the first 8 bytes for the first block
        if i == 0:
            print("First block's first 8 quantized values:")
            for j in range(8):
                value = block_stream[j]
                print_bits(value, 8)
        
        block_byte_streams.append(block_stream)
        block_padding_bits.append(block_padding)
    
    return block_byte_streams, quant_scales, block_padding_bits

# Set most blocks to 10 bits, and the last 5 blocks to 1 bit
bits_per_block = np.full(num_blocks, 10)
# bits_per_block[1] = 7 
bits_per_block[-5:] = 1  # Set the last 5 elements to 1
block_byte_streams, quant_scales, block_padding_bits = quantize_blocks(truncated_dct_blocks, bits_per_block)

# Modify the file writing part
fout = open('compressed', 'wb')
fout.write(struct.pack('<i', BLOCK_SIZE))
fout.write(struct.pack('<i', max(N_COEFFS)))
fout.write(struct.pack('<i', num_blocks))
fout.write(struct.pack('<i', pad_length))
fout.write(struct.pack('<i', framerate))

# Write the number of coefficients for each block
for n_coeff in N_COEFFS:
    fout.write(struct.pack('<i', n_coeff))

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
    if block_stream == block_byte_streams[0]:
        print(block_stream)
        for i in range(8):
            print_bits(block_stream[i], 8)
    fout.write(block_stream)

fout.close()
