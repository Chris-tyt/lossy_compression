#!/usr/bin/env python
import struct
import wave
import numpy as np
import scipy as sp
import sys
from scipy.fftpack import dct

# 参数设定
BLOCK_SIZE = 8192  # 每个块的大小
# N_COEFF = int(sys.argv[1]) if len(sys.argv) > 1 else 200
N_COEFF = 1700

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

# 截取前N_COEFF个系数(低频分量)
print(dct_blocks.shape)
truncated_dct_blocks = dct_blocks[:, :N_COEFF]
print(truncated_dct_blocks.shape)

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


def quantize_blocks(blocks, bits_per_block):
    assert len(bits_per_block) == blocks.shape[0], "位数数组长度必须等于块数"
    
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
        if i == 0:
            for j in range(8):
                print(quantized_values[j])
        
        for value in quantized_values:
            if(value == 975):
                print("975---")
                print(max(-(1 << (bits-1)), min(value, (1 << (bits-1)) - 1)))
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
        if i == 0:
            print("First block's first 8 quantized values:")
            for j in range(8):
                value = block_stream[j]
                print_bits(value, 8)
        
        block_byte_streams.append(block_stream)
        block_padding_bits.append(block_padding)
    
    return block_byte_streams, quant_scales, block_padding_bits

# 将所有块设置为11位
bits_per_block = np.full(num_blocks, 11)
block_byte_streams, quant_scales, block_padding_bits = quantize_blocks(truncated_dct_blocks, bits_per_block)

# 修改写入文件的部分
fout = open('compressed', 'wb')
fout.write(struct.pack('<i', BLOCK_SIZE))
fout.write(struct.pack('<i', N_COEFF))
fout.write(struct.pack('<i', num_blocks))
fout.write(struct.pack('<i', pad_length))
fout.write(struct.pack('<i', framerate))

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
    if block_stream == block_byte_streams[0]:
        print(block_stream)
        for i in range(8):
            print_bits(block_stream[i], 8)
    fout.write(block_stream)

fout.close()
