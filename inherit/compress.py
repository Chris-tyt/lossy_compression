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

# Process by blocks
num_blocks = int(np.ceil(len(samples) / BLOCK_SIZE))
# Zero-padding for incomplete blocks
pad_length = num_blocks * BLOCK_SIZE - len(samples)
if pad_length > 0:
    samples = np.concatenate((samples, np.zeros(pad_length, dtype=np.float32)))

# Perform DCT on each block
blocks = samples.reshape(num_blocks, BLOCK_SIZE)
dct_blocks = dct(blocks, type=2, norm='ortho', axis=1)  # Perform DCT on each block row

def calculate_energy_threshold(dct_block, energy_threshold=0.995):
    """
    Calculate spectral energy for each block and select the number of frequency coefficients 
    that reach the energy threshold.
    
    Parameters:
    dct_block: Block after DCT transformation
    energy_threshold: Energy threshold percentage (0-1)
    
    Returns:
    Number of selected frequency coefficients
    """
    # Calculate energy for each frequency coefficient
    energy = np.square(dct_block)
    total_energy = np.sum(energy)
    
    # Calculate cumulative energy
    cumulative_energy = np.cumsum(energy)
    
    # Find the number of frequency coefficients that reach the threshold
    num_coeffs = np.searchsorted(cumulative_energy, energy_threshold * total_energy) + 1
    
    return num_coeffs

# 对每个块计算所需的频率系数数量
coeffs_per_block = [calculate_energy_threshold(block) for block in dct_blocks]
coeffs_per_block = [250, 429, 439, 399, 319, 299, 339, 319, 319, 299, 359, 379, 299, 279, 339, 319, 319, 319, 319, 339, 339, 339, 319, 339, 299, 299, 259, 299, 259, 489, 399, 399, 319, 279, 250, 259, 299, 359, 319, 299, 399, 449, 509, 569, 489, 489, 429, 319, 299, 259, 250, 259, 259, 299, 319, 319, 429, 429, 429, 429, 429, 589, 429, 429, 509, 429, 429, 589, 509, 469, 429, 689, 429, 429, 429, 689, 469, 489, 469, 469, 429, 429, 489, 359, 339, 379, 339, 259, 259, 259, 279, 279, 339, 449, 449, 689, 509, 449, 689, 689, 669, 449, 449, 549, 429, 419, 319, 469, 489, 419, 299, 279, 299, 359, 399, 449, 429, 299, 259, 279, 319, 399, 299, 299, 469, 509, 509, 379, 339, 379, 399, 339, 359, 319, 319, 299, 339, 319, 339, 319, 439, 259, 259, 259, 299, 339, 279, 279, 279, 259, 259, 279, 279, 279, 259,1,1,1,1,1]# coeffs_per_block[2] = 250
# coeffs_per_block[149] = 250
# coeffs_per_block[75] = 700
# coeffs_per_block[98] = 600
# coeffs_per_block[99] = 600
# coeffs_per_block[100] = 600
# coeffs_per_block[71] = 600

# 截取每个块的前 coeffs_per_block[i] 个系数
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
    assert len(bits_per_block) == len(blocks), "位数数组长度必须等于块数"
    
    quant_scales = np.zeros(len(bits_per_block), dtype=np.float32)
    block_byte_streams = []  # 存储每个块的字节流
    block_padding_bits = []  # 存储每个块的填充位数
    
    for i, (block, bits) in enumerate(zip(blocks, bits_per_block)):
        max_val = np.max(np.abs(block))
        if max_val == 0:
            max_val = 1e-9
        
        # 计算量化比例
        quant_scale = (2**(bits-1) - 1) / max_val
        quant_scales[i] = quant_scale
        
        # 计算当前块需要的总位数和字节数
        block_total_bits = bits * len(block)
        block_total_bytes = (block_total_bits + 7) // 8
        block_padding = (8 - (block_total_bits % 8)) % 8
        
        # 创建当前块的字节流
        block_stream = bytearray(block_total_bytes)
        current_bit_pos = 0
        
        # 量化并存储当前块的值
        quantized_values = np.round(block * quant_scale).astype(np.int32)
        
        for value in quantized_values:
            # 确保值在有效范围内
            value = max(-(1 << (bits-1)), min(value, (1 << (bits-1)) - 1))
            if value < 0:
                value = value + (1 << bits)
            
            byte_index = current_bit_pos // 8
            bit_offset = current_bit_pos % 8
            
            # 初始化剩余需要写入的位数和当前值
            remaining_value = value
            bits_left = bits
            
            while bits_left > 0:
                # 计算当前字节剩余可写入的位数
                remaining_bits = 8 - (current_bit_pos % 8)
                # 确定这次写入的位数
                bits_to_write = min(remaining_bits, bits_left)
                
                # 准备写入的值
                if bits_left > bits_to_write:
                    # 如果还有更多位要写，取最高的bits_to_write位
                    write_value = (remaining_value >> (bits_left - bits_to_write))
                else:
                    # 如果这是最后一次写入，取所有剩余位
                    write_value = remaining_value
                
                # 写入值
                byte_index = current_bit_pos // 8
                block_stream[byte_index] |= (write_value & ((1 << bits_to_write) - 1)) << (remaining_bits - bits_to_write)
                
                # 更新剩余值和位数
                remaining_value &= ((1 << (bits_left - bits_to_write)) - 1)
                bits_left -= bits_to_write
                current_bit_pos += bits_to_write

        
        block_byte_streams.append(block_stream)
        block_padding_bits.append(block_padding)
    
    return block_byte_streams, quant_scales, block_padding_bits

# 将所有块设置为11位
bits_per_block = np.full(num_blocks, 10)
block_byte_streams, quant_scales, block_padding_bits = quantize_blocks(truncated_dct_blocks, bits_per_block)

print(coeffs_per_block)

# 修改写入文件的部分
fout = open('compressed', 'wb')
fout.write(struct.pack('<i', BLOCK_SIZE))
fout.write(struct.pack('<i', num_blocks))
fout.write(struct.pack('<i', pad_length))
fout.write(struct.pack('<i', framerate))

# 写入每个块的截取系数数量
for coeff_count in coeffs_per_block:
    fout.write(struct.pack('<h', coeff_count))

# 写入每个块的量化位数（现在都是11）
for bits in bits_per_block:
    fout.write(struct.pack('<B', bits))

# 写入每个块的量化比例
for scale in quant_scales:
    fout.write(struct.pack('<f', scale))

# 写入每个块的padding位数
for padding in block_padding_bits:
    fout.write(struct.pack('<B', padding))

# 写入压缩后的数据（每个块分别写入）
for block_stream in block_byte_streams:
    fout.write(block_stream)

fout.close()