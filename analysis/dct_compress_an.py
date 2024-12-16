#!/usr/bin/env python
import struct
import wave
import numpy as np
import scipy as sp
import sys
from scipy.fftpack import dct
import matplotlib.pyplot as plt

# Parameter settings
# Each block size
BLOCK_SIZE = 2048
# N_COEFF = int(sys.argv[1]) if len(sys.argv) > 1 else 200
N_COEFF = 429

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
# Pad with zeros for the last block if necessary
pad_length = num_blocks * BLOCK_SIZE - len(samples)
if pad_length > 0:
    samples = np.concatenate((samples, np.zeros(pad_length, dtype=np.float32)))

# Perform DCT on each block
blocks = samples.reshape(num_blocks, BLOCK_SIZE)
dct_blocks = dct(blocks, type=2, norm='ortho', axis=1)  # Perform DCT on each block row

# Extract the first N_COEFF coefficients (frequency components)
print(dct_blocks.shape)
truncated_dct_blocks = dct_blocks[:, :N_COEFF]
print(truncated_dct_blocks.shape)

# Example usage:
# Print the bits of a number
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
    assert len(bits_per_block) == blocks.shape[0], "位数数组长度必须等于块数"
    
    quant_scales = np.zeros(len(bits_per_block), dtype=np.float32)
    block_byte_streams = []  # 存储每个块的字节流
    block_padding_bits = []  # 存储每个块的填充位数
    mse_values = []  # 存储所有块的MSE值
    
    for i, (block, bits) in enumerate(zip(blocks, bits_per_block)):
        max_val = np.max(np.abs(block))
        if max_val == 0:
            max_val = 1e-9
        
        # 计算量化比例
        quant_scale = (2**(bits-1) - 1) / max_val
        quant_scales[i] = quant_scale
        
        # 计算当前块需要的总位数和字节数
        block_total_bits = bits * block.shape[0]
        block_total_bytes = (block_total_bits + 7) // 8
        block_padding = (8 - (block_total_bits % 8)) % 8
        
        # 创建当前块的字节流
        block_stream = bytearray(block_total_bytes)
        current_bit_pos = 0
        # if i == 0:
        #     for j in range(8):
        #         print(block[j])
        
        # 量化并存储当前块的值
        quantized_values = np.round(block * quant_scale).astype(np.int32)
        # 计算原始值和量化后值的MSE
        dequantized_values = quantized_values.astype(np.float32) / quant_scale
        mse = np.mean((block - dequantized_values) ** 2)
        mse_values.append((i, mse))  # 存储块索引和MSE值
        print(f"Block {i} MSE: {mse:.10f}")
        
        # if i == 0:
        #     for j in range(8):
        #         print(quantized_values[j])
        
        for value in quantized_values:
            # if(value == 975):
            #     print("975---")
            #     print(max(-(1 << (bits-1)), min(value, (1 << (bits-1)) - 1)))
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
        # print("-------------------------end-------------------------------")
        
        # 处理第一个块时打印前8个字节的二进制表示
        if i == 75:
            print("First block's first 8 quantized values:")
            for j in range(8):
                value = block_stream[j]
                print_bits(value, 8)
        
        block_byte_streams.append(block_stream)
        block_padding_bits.append(block_padding)
    
    # 在函数返回前打印前5个最大的MSE值
    sorted_mse = sorted(mse_values, key=lambda x: x[1], reverse=True)
    print("\nTop 5 largest MSE values:")
    for idx, (block_idx, mse_val) in enumerate(sorted_mse[:5]):
        print(f"#{idx+1}: Block {block_idx}, MSE = {mse_val:.10f}")

    return block_byte_streams, quant_scales, block_padding_bits

# Set most blocks to 10 bits, the last 5 blocks to 1 bit
bits_per_block = np.full(num_blocks, 10)
# bits_per_block[1] = 7 
bits_per_block[-5:] = 1  # 将最后5个元素设置为1
block_byte_streams, quant_scales, block_padding_bits = quantize_blocks(truncated_dct_blocks, bits_per_block)

# Modify the part that writes to the file
fout = open('compressed', 'wb')
fout.write(struct.pack('<i', BLOCK_SIZE))
fout.write(struct.pack('<i', N_COEFF))
fout.write(struct.pack('<i', num_blocks))
fout.write(struct.pack('<i', pad_length))
fout.write(struct.pack('<i', framerate))

# Write the quantization bit count for each block (now all are 11)
for bits in bits_per_block:
    fout.write(struct.pack('<B', bits))

# Write the quantization scale for each block
for scale in quant_scales:
    fout.write(struct.pack('<f', scale))

# Write the padding bit count for each block
for padding in block_padding_bits:
    fout.write(struct.pack('<B', padding))

# Write the compressed data (write each block separately)
for block_stream in block_byte_streams:
    # if block_stream == block_byte_streams[0]:
    #     print(block_stream)
    #     for i in range(8):
    #         print_bits(block_stream[i], 8)
    fout.write(block_stream)

fout.close()

if True:
    # Add the following code after DCT conversion
    # Choose the first block to visualize
    for i, per_block in enumerate(dct_blocks):
        if i in range(len(dct_blocks)):
            freq = np.arange(len(per_block))

            # Plot the frequency spectrum
            plt.figure(figsize=(12, 6))
            plt.plot(freq, np.abs(per_block))
            plt.title(f'DCT Spectrum of {i} Block')
            plt.xlabel('Frequency Index')
            plt.ylabel('Magnitude')
            plt.grid(True)

            # # Use a logarithmic scale to see details more easily
            # plt.yscale('log')
            plt.show()

if True:
    # If you want to see the average spectrum of all blocks
    average_spectrum = np.mean(np.abs(dct_blocks), axis=0)
    plt.figure(figsize=(12, 6))
    freq = np.arange(len(dct_blocks[0]))
    plt.plot(freq, average_spectrum)
    plt.title('Average DCT Spectrum Across All Blocks')
    plt.xlabel('Frequency Index')
    plt.ylabel('Magnitude')
    plt.grid(True)
    # plt.yscale('log')
    plt.show()
