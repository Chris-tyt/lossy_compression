#!/usr/bin/env python
import struct
import wave
import numpy as np
from scipy.fftpack import idct

# Read header information from the file
fin = open('compressed', 'rb')
BLOCK_SIZE = struct.unpack('<i', fin.read(4))[0]
num_blocks = struct.unpack('<i', fin.read(4))[0]
pad_length = struct.unpack('<i', fin.read(4))[0]
framerate = struct.unpack('<i', fin.read(4))[0]

# Read the number of coefficients for each block
coeffs_per_block = [struct.unpack('<h', fin.read(2))[0] for _ in range(num_blocks)]

# Read the quantization bits for each block
bits_per_block = np.array([struct.unpack('<B', fin.read(1))[0] for _ in range(num_blocks)])

# Read the quantization scales for each block
quant_scales = np.array([struct.unpack('<f', fin.read(4))[0] for _ in range(num_blocks)])

# Read the padding bits for each block
padding_bits = np.array([struct.unpack('<B', fin.read(1))[0] for _ in range(num_blocks)])

# Read compressed data
compressed_data = fin.read()
fin.close()

# Decompress data for each block
dct_blocks = []
current_byte_pos = 0

for block_idx in range(num_blocks):
    bits = bits_per_block[block_idx]
    n_coeff = coeffs_per_block[block_idx]
    # Calculate the number of bytes for the current block
    block_total_bits = bits * n_coeff
    block_total_bytes = (block_total_bits + 7) // 8
    
    # Read the bytes for the current block
    block_bytes = compressed_data[current_byte_pos:current_byte_pos + block_total_bytes]
    current_byte_pos += block_total_bytes
    
    # Parse the bitstream
    current_bit_pos = 0
    block_dct = np.zeros(n_coeff, dtype=np.float32)
    for coef_idx in range(n_coeff):
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
            
        # Convert value back to signed number
        if value >= (1 << (bits-1)):
            value -= (1 << bits)

        # Dequantize
        block_dct[coef_idx] = value / quant_scales[block_idx]
    
    # Fill DCT block to full size
    full_block_dct = np.zeros(BLOCK_SIZE, dtype=np.float32)
    full_block_dct[:n_coeff] = block_dct
    dct_blocks.append(full_block_dct)

# Perform IDCT to recover time-domain signal for each block
reconstructed_blocks = idct(dct_blocks, type=2, norm='ortho', axis=1)

# Concatenate all blocks
reconstructed_samples = np.concatenate(reconstructed_blocks)

# Remove padding
if pad_length > 0:
    reconstructed_samples = reconstructed_samples[:-pad_length]

# Convert floating point [-1,1] back to 16-bit integer
samplesint = np.clip(np.round(reconstructed_samples * (2**15)), -32768, 32767).astype(np.int16)

# Write to wav file
fp = wave.open('out.wav', 'wb')
fp.setparams((1, 2, framerate, len(samplesint), 'NONE', 'not compressed'))
fp.writeframes(struct.pack('<' + 'h'*len(samplesint), *samplesint))
fp.close()