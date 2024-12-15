#!/usr/bin/env python
import struct
import wave
import numpy as np
import scipy as sp
import sys
from scipy.fftpack import dct

# 参数设定
BLOCK_SIZE = 2048  # 每个块的大小
# N_COEFF = int(sys.argv[1]) if len(sys.argv) > 1 else 200
# N_COEFF = 1700

# 打开输入音频文件
fin = wave.open('step.wav', 'r')
(nchannels, sampwidth, framerate, nframes, comptype, compname) = fin.getparams()
inbytes = fin.readframes(nframes)
fin.close()

# 将字节流转换为浮点型数组（归一化到[-1,1]之间）
samplesint = np.array([struct.unpack('<h', inbytes[2*i:2*i+2])[0] for i in range(nframes)])
samples = samplesint.astype(np.float32) / (2**15)

# 分块处理
num_blocks = int(np.ceil(len(samples) / BLOCK_SIZE))
# 对于不足一个块的部分进行补零
pad_length = num_blocks * BLOCK_SIZE - len(samples)
if pad_length > 0:
    samples = np.concatenate((samples, np.zeros(pad_length, dtype=np.float32)))

# 对每个块进行DCT
blocks = samples.reshape(num_blocks, BLOCK_SIZE)
dct_blocks = dct(blocks, type=2, norm='ortho', axis=1)  # 对每个块行进行DCT

def calculate_energy_threshold(dct_block, energy_threshold=0.995):
    """
    计算每个块的频谱能量，并选择达到能量阈值的频率系数数量。
    
    参数:
    dct_block: DCT 变换后的块
    energy_threshold: 能量阈值百分比 (0-1)
    
    返回:
    选择的频率系数数量
    """
    # 计算每个频率系数的能量
    energy = np.square(dct_block)
    total_energy = np.sum(energy)
    
    # 计算累积能量
    cumulative_energy = np.cumsum(energy)
    
    # 找到累积能量达到阈值的频率系数数量
    num_coeffs = np.searchsorted(cumulative_energy, energy_threshold * total_energy) + 1
    
    return num_coeffs

# 对每个块计算所需的频率系数数量
coeffs_per_block = [calculate_energy_threshold(block) for block in dct_blocks]
coeffs_per_block = [429] * num_blocks

# 处理命令行参数
if len(sys.argv) > 2:
    block_index = int(sys.argv[1])  # 第一个参数：要修改的块的索引
    new_coeff_count = int(sys.argv[2])  # 第二个参数：新的系数数量
    if 0 <= block_index < num_blocks:  # 确保索引在有效范围内
        coeffs_per_block[block_index] = new_coeff_count

# 截取每个块的前 coeffs_per_block[i] 个系数
truncated_dct_blocks = [block[:n] for block, n in zip(dct_blocks, coeffs_per_block)]

def print_bits(value, bits):
    """
    打印一个数字的每一位
    
    参数:
    value: 要打印的数字
    bits: 数字的位数
    """
    print(value,end=' : ')
    for i in range(bits-1, -1, -1):
        bit = (value >> i) & 1
        print(bit, end='')
    print()  # 换行


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
                    # 如果这是最后一次写入，取所有剩余���
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

# print(coeffs_per_block)

# 修改写入文件的部分
fout = open('compressed', 'wb')
fout.write(struct.pack('<i', BLOCK_SIZE))
fout.write(struct.pack('<i', num_blocks))
fout.write(struct.pack('<i', pad_length))
fout.write(struct.pack('<i', framerate))

# 写入每个块的截取系数数量
for coeff_count in coeffs_per_block:
    fout.write(struct.pack('<i', coeff_count))

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