#!/usr/bin/env python
import struct
import wave
import numpy as np
from scipy.fftpack import idct

# Read header information from the file
fin = open('compressed', 'rb')
header = fin.read(4*5)  # Read 5 integers from the file
BLOCK_SIZE = struct.unpack('<i', header[0:4])[0]
N_COEFF = struct.unpack('<i', header[4:8])[0]
num_blocks = struct.unpack('<i', header[8:12])[0]
pad_length = struct.unpack('<i', header[12:16])[0]
framerate = struct.unpack('<i', header[16:20])[0]

# Read the quantization bits for each block
bits_per_block = np.array([struct.unpack('<B', fin.read(1))[0] for _ in range(num_blocks)])

# Read the quantization scales for each block
quant_scales = np.array([struct.unpack('<f', fin.read(4))[0] for _ in range(num_blocks)])

# Read the padding bits for each block
padding_bits = np.array([struct.unpack('<B', fin.read(1))[0] for _ in range(num_blocks)])

# Read compressed data
compressed_data = fin.read()
fin.close()

# Example usage:
# Print the bits of a number
def print_bits(value, bits):
    """
    Print each bit of a number
    
    Parameters:
    value: The number to be printed
    bits: The number of bits
    """
    print(value,end=' : ')
    for i in range(bits-1, -1, -1):
        bit = (value >> i) & 1
        print(bit, end='')
    print()  # New line

# Decompress the data for each block
dct_blocks = np.zeros((num_blocks, N_COEFF), dtype=np.float32)
current_byte_pos = 0

for block_idx in range(num_blocks):
    bits = bits_per_block[block_idx]
    # Calculate the number of bytes for the current block
    block_total_bits = bits * N_COEFF
    block_total_bytes = (block_total_bits + 7) // 8
    
    # Read the bytes of the current block
    block_bytes = compressed_data[current_byte_pos:current_byte_pos + block_total_bytes]
    if block_idx == 75:
        for j in range(8):
            value = block_bytes[j]
            print_bits(value, 8)
    current_byte_pos += block_total_bytes
    
    # Parse the bitstream
    current_bit_pos = 0
    for coef_idx in range(N_COEFF):
        value = 0
        bits_remaining = bits
        while bits_remaining > 0:
            byte_idx = current_bit_pos // 8
            bit_offset = current_bit_pos % 8
            
            # Determine how many bits can be read from the current byte
            bits_from_byte = min(8 - bit_offset, bits_remaining)
            
            # Extract bits from the byte
            mask = ((1 << bits_from_byte) - 1) << (8 - bit_offset - bits_from_byte)
            byte_value = (block_bytes[byte_idx] & mask) >> (8 - bit_offset - bits_from_byte)
            
            # Place the extracted bits in the correct position
            value = (value << bits_from_byte) | byte_value
            
            bits_remaining -= bits_from_byte
            current_bit_pos += bits_from_byte
            
        # Convert the value back to a signed number
        if value >= (1 << (bits-1)):
            value -= (1 << bits)
        if block_idx == 0 and 0<byte_idx<16:
            print(value)
            
        # De-quantization
        dct_blocks[block_idx, coef_idx] = value / quant_scales[block_idx]
    # if block_idx == 0:
    #     for i in range(8):
    #         print(dct_blocks[block_idx, i])

# Fill DCT blocks to full size
full_dct_blocks = np.zeros((num_blocks, BLOCK_SIZE), dtype=np.float32)
full_dct_blocks[:, :N_COEFF] = dct_blocks

# Perform IDCT to recover time-domain signal for each block
reconstructed_blocks = idct(full_dct_blocks, type=2, norm='ortho', axis=1)

# Merge all blocks
reconstructed_samples = reconstructed_blocks.flatten()

# Remove padding
if pad_length > 0:
    reconstructed_samples = reconstructed_samples[:-pad_length]

# Convert float [-1,1] back to 16-bit integer
samplesint = np.clip(np.round(reconstructed_samples * (2**15)), -32768, 32767).astype(np.int16)

# Write to wav file
fp = wave.open('out.wav', 'wb')
fp.setparams((1, 2, framerate, len(samplesint), 'NONE', 'not compressed'))
fp.writeframes(struct.pack('<' + 'h'*len(samplesint), *samplesint))
fp.close()
