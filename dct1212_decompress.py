#!/usr/bin/env python
import struct
import wave
import numpy as np
from scipy.fftpack import idct

# 从文件中读取头部信息
fin = open('compressed', 'rb')
header = fin.read(4*5)  # 读取5个整数参数
BLOCK_SIZE = struct.unpack('<i', header[0:4])[0]
N_COEFF = struct.unpack('<i', header[4:8])[0]
num_blocks = struct.unpack('<i', header[8:12])[0]
pad_length = struct.unpack('<i', header[12:16])[0]
framerate = struct.unpack('<i', header[16:20])[0]

# 读取每个块的量化位数
bits_per_block = np.array([struct.unpack('<B', fin.read(1))[0] for _ in range(num_blocks)])

# 读取每个块的量化比例
quant_scales = np.array([struct.unpack('<f', fin.read(4))[0] for _ in range(num_blocks)])

# 读取每个块的padding位数
padding_bits = np.array([struct.unpack('<B', fin.read(1))[0] for _ in range(num_blocks)])

# 读取压缩数据
compressed_data = fin.read()
fin.close()

# 使用示例:
# print_bits(10, 4)  # 输出: 1010
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

# 解压每个块的数据
dct_blocks = np.zeros((num_blocks, N_COEFF), dtype=np.float32)
current_byte_pos = 0

for block_idx in range(num_blocks):
    bits = bits_per_block[block_idx]
    # 计算当前块的字节数
    block_total_bits = bits * N_COEFF
    block_total_bytes = (block_total_bits + 7) // 8
    
    # 读取当前块的字节
    block_bytes = compressed_data[current_byte_pos:current_byte_pos + block_total_bytes]
    if block_idx == 0:
        for j in range(8):
            value = block_bytes[j]
            print_bits(value, 8)
    current_byte_pos += block_total_bytes
    
    # 解析比特流
    current_bit_pos = 0
    for coef_idx in range(N_COEFF):
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
        if block_idx == 0 and 0<byte_idx<16:
            print(value)
            
        # 反量化
        dct_blocks[block_idx, coef_idx] = value / quant_scales[block_idx]
    # if block_idx == 0:
    #     for i in range(8):
    #         print(dct_blocks[block_idx, i])

# 填充DCT块到完整大小
full_dct_blocks = np.zeros((num_blocks, BLOCK_SIZE), dtype=np.float32)
full_dct_blocks[:, :N_COEFF] = dct_blocks

# 对每个块进行IDCT恢复时域信号
reconstructed_blocks = idct(full_dct_blocks, type=2, norm='ortho', axis=1)

# 合并所有块
reconstructed_samples = reconstructed_blocks.flatten()

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
