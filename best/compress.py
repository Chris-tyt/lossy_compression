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

# Convert byte stream to floating-point array (normalized to [-1,1])
samplesint = np.array([struct.unpack('<h', inbytes[2*i:2*i+2])[0] for i in range(nframes)])
samples = samplesint.astype(np.float32) / (2**15)

# Process in blocks
num_blocks = int(np.ceil(len(samples) / BLOCK_SIZE))
# Pad the part that is less than one block with zeros
pad_length = num_blocks * BLOCK_SIZE - len(samples)
if pad_length > 0:
    samples = np.concatenate((samples, np.zeros(pad_length, dtype=np.float32)))

# Calculate the number of frequency coefficients that reach the energy threshold for each block
blocks = samples.reshape(num_blocks, BLOCK_SIZE)
dct_blocks = dct(blocks, type=2, norm='ortho', axis=1)  # Calculate the DCT for each block row

def calculate_energy_threshold(dct_block, energy_threshold=0.948):
    """
    Calculate the number of frequency coefficients that reach the energy threshold for each block.
    
    Parameters:
    dct_block: DCT transformed block
    energy_threshold: Energy threshold percentage (0-1)
    
    Returns:
    Number of selected frequency coefficients
    """
    # Calculate the energy of each frequency coefficient
    # energy = np.square(dct_block)
    energy = np.abs(dct_block)
    total_energy = np.sum(energy)
    
    # Calculate cumulative energy
    cumulative_energy = np.cumsum(energy)
    
    # Find the number of frequency coefficients that reach the threshold
    num_coeffs = np.searchsorted(cumulative_energy, energy_threshold * total_energy) + 1
    
    return num_coeffs


def calculate_weighted_energy_threshold(dct_block, energy_threshold=0.95):
    """
    Calculate the number of frequency coefficients that reach the energy threshold for each block.
    
    Parameters:
    dct_block: DCT transformed block
    energy_threshold: Energy threshold percentage (0-1)
    
    Returns:
    Number of selected frequency coefficients
    """
    # Calculate the energy of each frequency coefficient
    energy = np.abs(dct_block)
    
    # Use exponential weighting, with higher weights for low-frequency coefficients
    weights = np.exp(-np.linspace(0, 1, len(dct_block)))  # Exponential decay weights
    weighted_energy = energy * weights
    
    total_energy = np.sum(weighted_energy)
    
    # Calculate cumulative energy
    cumulative_energy = np.cumsum(weighted_energy)
    
    # Find the number of frequency coefficients that reach the threshold
    num_coeffs = np.searchsorted(cumulative_energy, energy_threshold * total_energy) + 1
    
    return num_coeffs


# Calculate the number of frequency coefficients required for each block
# coeffs_per_block = [calculate_energy_threshold(block) for block in dct_blocks]
# coeffs_per_block = [calculate_weighted_energy_threshold(block) for block in dct_blocks]
# coeffs_per_block = [68, 335, 409, 406, 275, 326, 279, 306, 330, 314, 325, 345, 275, 272, 265, 269, 263, 246, 277, 329, 323, 309, 328, 322, 312, 278, 258, 273, 257, 442, 368, 344, 299, 248, 167, 85, 267, 360, 310, 264, 358, 362, 498, 574, 486, 377, 402, 247, 252, 174, 85, 88, 213, 253, 284, 276, 326, 299, 306, 323, 369, 329, 326, 363, 361, 373, 369, 368, 309, 337, 338, 383, 324, 365, 330, 715, 333, 427, 347, 337, 332, 289, 372, 378, 305, 346, 301, 250, 248, 246, 266, 264, 276, 307, 370, 421, 422, 399, 694, 694, 365, 412, 351, 367, 383, 329, 327, 474, 490, 380, 312, 270, 303, 342, 390, 436, 407, 262, 243, 267, 302, 393, 300, 268, 472, 514, 388, 361, 304, 346, 351, 320, 338, 308, 310, 265, 313, 255, 251, 265, 410, 251, 232, 255, 244, 268, 267, 271, 235, 239, 213, 182, 251, 251, 178, 1, 1, 1, 1, 1]
# coeffs_per_block = [68, 338, 449, 406, 313, 346, 308, 315, 330, 314, 360, 389, 275, 272, 269, 333, 311, 266, 313, 361, 337, 330, 328, 324, 314, 296, 270, 297, 270, 463, 430, 391, 318, 267, 184, 190, 276, 360, 310, 293, 407, 474, 498, 610, 556, 476, 422, 312, 301, 246, 246, 244, 242, 311, 348, 311, 363, 356, 363, 362, 397, 489, 386, 419, 621, 431, 423, 622, 481, 381, 345, 706, 358, 469, 436, 773, 572, 505, 462, 397, 422, 451, 497, 378, 341, 408, 342, 306, 255, 256, 267, 311, 332, 350, 446, 697, 580, 499, 707, 735, 731, 417, 441, 620, 383, 429, 329, 474, 490, 411, 312, 271, 309, 342, 396, 455, 429, 327, 271, 267, 329, 393, 300, 298, 494, 528, 517, 386, 361, 365, 416, 335, 391, 331, 338, 307, 322, 377, 364, 316, 431, 265, 265, 265, 274, 348, 275, 280, 252, 242, 267, 274, 251, 273, 241, 1, 1, 1, 1, 1]
# coeffs_per_block = [68, 338, 464, 406, 324, 346, 337, 315, 337, 314, 392, 389, 311, 305, 333, 335, 311, 319, 313, 365, 337, 330, 328, 327, 314, 303, 270, 297, 270, 488, 430, 418, 318, 268, 184, 244, 299, 360, 310, 293, 407, 474, 498, 610, 556, 476, 422, 312, 301, 246, 244, 244, 242, 307, 312, 311, 363, 301, 312, 362, 389, 416, 362, 419, 621, 431, 423, 622, 481, 381, 342, 706, 333, 413, 413, 773, 423, 505, 462, 397, 422, 451, 484, 378, 341, 365, 320, 250, 255, 256, 267, 274, 304, 350, 446, 697, 580, 399, 707, 735, 731, 417, 441, 620, 383, 415, 329, 474, 490, 411, 312, 271, 305, 342, 396, 455, 407, 262, 271, 267, 329, 393, 300, 290, 486, 528, 517, 386, 361, 354, 416, 335, 391, 331, 321, 267, 322, 278, 364, 316, 431, 260, 234, 255, 250, 348, 275, 280, 239, 242, 267, 274, 251, 273, 182, 1, 1, 1, 1, 1]

# 2048 ml_plus
# coeffs_per_block = [68, 338, 458, 406, 318, 346, 308, 315, 337, 314, 360, 389, 311, 272, 269, 335, 311, 266, 313, 365, 337, 330, 328, 324, 314, 296, 270, 297, 270, 488, 430, 391, 318, 267, 184, 190, 276, 360, 310, 293, 407, 474, 498, 610, 556, 476, 422, 312, 301, 246, 244, 244, 242, 307, 312, 311, 363, 301, 312, 362, 389, 416, 362, 419, 621, 431, 423, 622, 481, 381, 342, 706, 333, 413, 413, 773, 423, 505, 462, 397, 422, 451, 484, 378, 341, 365, 320, 250, 255, 256, 267, 274, 304, 350, 446, 697, 580, 399, 707, 735, 731, 417, 441, 620, 383, 415, 329, 474, 490, 411, 312, 271, 305, 342, 396, 455, 407, 262, 271, 267, 329, 393, 300, 290, 486, 528, 517, 386, 361, 354, 416, 335, 391, 331, 321, 267, 322, 278, 364, 316, 431, 260, 234, 255, 250, 348, 275, 280, 239, 242, 267, 274, 251, 273, 182, 1, 1, 1, 1, 1]

# 1024
# coeffs_per_block = [1, 34, 34, 179, 204, 229, 207, 202, 205, 155, 174, 174, 192, 151, 146, 159, 172, 165, 158, 157, 173, 196, 198, 168, 134, 156, 136, 153, 133, 142, 166, 168, 153, 135, 133, 160, 136, 158, 151, 182, 169, 166, 195, 168, 165, 165, 163, 158, 163, 160, 152, 134, 135, 135, 136, 149, 110, 135, 263, 247, 215, 213, 181, 196, 159, 144, 134, 127, 92, 114, 92, 96, 122, 150, 185, 202, 176, 158, 147, 144, 155, 211, 222, 240, 245, 281, 295, 312, 255, 280, 250, 184, 212, 203, 155, 135, 155, 134, 124, 122, 123, 46, 120, 88, 108, 118, 127, 152, 154, 172, 155, 156, 182, 185, 124, 151, 156, 155, 181, 183, 185, 207, 206, 170, 194, 156, 212, 203, 262, 311, 184, 217, 212, 200, 287, 323, 316, 213, 168, 237, 167, 284, 371, 281, 173, 167, 183, 235, 215, 175, 218, 392, 213, 185, 241, 254, 327, 196, 199, 169, 219, 184, 222, 217, 317, 236, 202, 169, 166, 171, 197, 185, 166, 132, 126, 136, 129, 124, 128, 121, 133, 139, 153, 156, 155, 152, 174, 158, 163, 221, 349, 333, 290, 281, 201, 251, 354, 370, 368, 347, 381, 366, 197, 209, 221, 320, 256, 355, 215, 178, 226, 170, 178, 164, 210, 246, 252, 244, 237, 190, 158, 136, 129, 145, 155, 152, 170, 170, 191, 198, 227, 261, 297, 160, 136, 124, 136, 135, 135, 134, 157, 178, 209, 183, 150, 135, 135, 151, 214, 250, 269, 265, 247, 255, 195, 182, 199, 170, 155, 253, 210, 215, 158, 185, 180, 195, 168, 166, 150, 167, 151, 133, 161, 157, 138, 160, 188, 182, 155, 133, 217, 247, 133, 126, 117, 128, 121, 135, 121, 133, 173, 147, 137, 138, 158, 139, 128, 120, 117, 121, 122, 134, 126, 137, 129, 124, 134, 132, 126, 54, 1, 1, 1, 1, 1, 1, 1, 1, 1]

coeffs_per_block= [68, 338, 458, 406, 318, 346, 308, 315, 337, 314, 360, 389, 311, 272, 269, 335, 311, 266, 313, 365, 337, 330, 328, 324, 314, 296, 270, 297, 270, 488, 430, 391, 318, 267, 184, 190, 276, 360, 310, 293, 407, 474, 498, 610, 556, 476, 422, 312, 301, 246, 244, 244, 242, 307, 312, 311, 363, 301, 312, 362, 389, 416, 362, 419, 621, 431, 423, 622, 481, 381, 342, 706, 333, 413, 413, 773, 423, 505, 462, 397, 422, 451, 484, 378, 341, 365, 320, 250, 255, 256, 267, 274, 304, 350, 446, 697, 580, 399, 707, 735, 731, 417, 441, 620, 383, 415, 329, 474, 490, 411, 312, 271, 305, 342, 396, 455, 407, 262, 271, 267, 329, 393, 300, 290, 486, 528, 517, 386, 361, 354, 416, 335, 391, 331, 321, 267, 322, 278, 364, 316, 431, 260, 234, 255, 250, 348, 275, 280, 239, 242, 267, 274, 251, 273, 182, 1, 1, 1, 1, 1]



print("coeffs_per_block[75]:",coeffs_per_block[75])
print("coeffs_per_block[98]:",coeffs_per_block[98])
print("coeffs_per_block[99]:",coeffs_per_block[99])
print("coeffs_per_block[100]:",coeffs_per_block[100])
print("coeffs_per_block[71]:",coeffs_per_block[71])
print("coeffs_per_block[149]:",coeffs_per_block[149])
# coeffs_per_block = [429] * num_blocks

# Truncate the first coeffs_per_block[i] coefficients of each block
truncated_dct_blocks = [block[:n] for block, n in zip(dct_blocks, coeffs_per_block)]

def print_bits(value, bits):
    """
    Print each bit of a number
    
    Parameters:
    value: Number to print
    bits: Number of bits
    """
    print(value,end=' : ')
    for i in range(bits-1, -1, -1):
        bit = (value >> i) & 1
        print(bit, end='')
    print()  # New line


def quantize_blocks(blocks, bits_per_block):
    assert len(bits_per_block) == len(blocks), "The length of the bits array must equal the number of blocks"
    
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
        
        # Calculate total bits and bytes needed for the current block
        block_total_bits = bits * len(block)
        block_total_bytes = (block_total_bits + 7) // 8
        block_padding = (8 - (block_total_bits % 8)) % 8
        
        # Create byte stream for the current block
        block_stream = bytearray(block_total_bytes)
        current_bit_pos = 0
        
        # Quantize and store values for the current block
        quantized_values = np.round(block * quant_scale).astype(np.int32)
        
        for value in quantized_values:
            # Ensure value is within valid range
            value = max(-(1 << (bits-1)), min(value, (1 << (bits-1)) - 1))
            if value < 0:
                value = value + (1 << bits)
            
            byte_index = current_bit_pos // 8
            bit_offset = current_bit_pos % 8
            
            # Initialize remaining bits and current value
            remaining_value = value
            bits_left = bits
            
            while bits_left > 0:
                # Calculate remaining bits in the current byte
                remaining_bits = 8 - (current_bit_pos % 8)
                # Determine how many bits to write this time
                bits_to_write = min(remaining_bits, bits_left)
                
                # Prepare value to write
                if bits_left > bits_to_write:
                    # If there are more bits to write, take the highest bits_to_write bits
                    write_value = (remaining_value >> (bits_left - bits_to_write))
                else:
                    # If this is the last time to write, take all remaining bits
                    write_value = remaining_value
                
                # Write value
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

print(coeffs_per_block)

# Modify the part to write to the file
fout = open('compressed', 'wb')
fout.write(struct.pack('<i', BLOCK_SIZE))
fout.write(struct.pack('<i', num_blocks))
fout.write(struct.pack('<i', pad_length))
fout.write(struct.pack('<i', framerate))

# Write the number of coefficients for each block
for coeff_count in coeffs_per_block:
    fout.write(struct.pack('<h', coeff_count))

# Write the quantization bit count for each block (now all are 11)
for bits in bits_per_block:
    fout.write(struct.pack('<B', bits))

# Write the quantization scale for each block
for scale in quant_scales:
    fout.write(struct.pack('<f', scale))

# Write the padding bits for each block
for padding in block_padding_bits:
    fout.write(struct.pack('<B', padding))

# Write the compressed data (write each block separately)
for block_stream in block_byte_streams:
    fout.write(block_stream)

fout.close()