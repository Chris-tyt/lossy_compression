#!/usr/bin/env python
import struct
import wave
import numpy as np
from scipy.fftpack import idct

# 从文件中读取头部信息
fin = open('compressed', 'rb')
BLOCK_SIZE = struct.unpack('<i', fin.read(4))[0]
num_blocks = struct.unpack('<i', fin.read(4))[0]
pad_length = struct.unpack('<i', fin.read(4))[0]
framerate = struct.unpack('<i', fin.read(4))[0]

# 读取每个块的截取系数数量
coeffs_per_block = [struct.unpack('<h', fin.read(2))[0] for _ in range(num_blocks)]

# 读取每个块的量化位数
bits_per_block = np.array([struct.unpack('<B', fin.read(1))[0] for _ in range(num_blocks)])

# 读取每个块的量化比例
quant_scales = np.array([struct.unpack('<f', fin.read(4))[0] for _ in range(num_blocks)])

# 读取每个块的padding位数
padding_bits = np.array([struct.unpack('<B', fin.read(1))[0] for _ in range(num_blocks)])

# 读取压缩数据
compressed_data = fin.read()
fin.close()

# 解压每个块的数据
dct_blocks = []
current_byte_pos = 0

for block_idx in range(num_blocks):
    bits = bits_per_block[block_idx]
    n_coeff = coeffs_per_block[block_idx]
    # 计算当前块的字节数
    block_total_bits = bits * n_coeff
    block_total_bytes = (block_total_bits + 7) // 8
    
    # 读取当前块的字节
    block_bytes = compressed_data[current_byte_pos:current_byte_pos + block_total_bytes]
    current_byte_pos += block_total_bytes
    
    # 解析比特流
    current_bit_pos = 0
    block_dct = np.zeros(n_coeff, dtype=np.float32)
    for coef_idx in range(n_coeff):
        value = 0
        bits_remaining = bits
        while bits_remaining > 0:
            byte_idx = current_bit_pos // 8
            bit_offset = current_bit_pos % 8
            
            # 确定从当前字节可以读取多少位
            bits_from_byte = min(8 - bit_offset, bits_remaining)
            
            # 从字节中提取位
            mask = ((1 << bits_from_byte) - 1) << (8 - bit_offset - bits_from_byte)
            byte_value = (block_bytes[byte_idx] & mask) >> (8 - bit_offset - bits_from_byte)
            
            # 将提取的位放到正确的位置
            value = (value << bits_from_byte) | byte_value
            
            bits_remaining -= bits_from_byte
            current_bit_pos += bits_from_byte
            
        # 将值转换回有符号数
        if value >= (1 << (bits-1)):
            value -= (1 << bits)

        # 反量化
        block_dct[coef_idx] = value / quant_scales[block_idx]
    
    # 填充DCT块到完整大小
    full_block_dct = np.zeros(BLOCK_SIZE, dtype=np.float32)
    full_block_dct[:n_coeff] = block_dct
    dct_blocks.append(full_block_dct)

# 对每个块进行IDCT恢复时域信号
reconstructed_blocks = idct(dct_blocks, type=2, norm='ortho', axis=1)

# 合并所有块
reconstructed_samples = np.concatenate(reconstructed_blocks)

# 去除填充部分
if pad_length > 0:
    reconstructed_samples = reconstructed_samples[:-pad_length]

# 将浮点[-1,1]转换回16-bit整数
samplesint = np.clip(np.round(reconstructed_samples * (2**15)), -32768, 32767).astype(np.int16)

# 写入wav文件
fp = wave.open('out.wav', 'wb')
fp.setparams((1, 2, framerate, len(samplesint), 'NONE', 'not compressed'))
fp.writeframes(struct.pack('<' + 'h'*len(samplesint), *samplesint))
fp.close()