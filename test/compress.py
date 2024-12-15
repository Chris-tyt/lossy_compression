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

def calculate_energy_threshold(dct_block, energy_threshold=0.948):
    """
    计算每个块的频谱能量，并选择达到能量阈值的频率系数数量。
    
    参数:
    dct_block: DCT 变换后的块
    energy_threshold: 能量阈值百分比 (0-1)
    
    返回:
    选择的频率系数数量
    """
    # 计算每个频率系数的能量
    # energy = np.square(dct_block)
    energy = np.abs(dct_block)
    total_energy = np.sum(energy)
    
    # 计算累积能量
    cumulative_energy = np.cumsum(energy)
    
    # 找到累积能量达到阈值的频率系数数量
    num_coeffs = np.searchsorted(cumulative_energy, energy_threshold * total_energy) + 1
    
    return num_coeffs


def calculate_weighted_energy_threshold(dct_block, energy_threshold=0.95):
    """
    计算每个块的加权频谱能量，并选择达到能量阈值的频率系数数量。
    
    参数:
    dct_block: DCT 变换后的块
    energy_threshold: 能量阈值百分比 (0-1)
    
    返回:
    选择的频率系数数量
    """
    # 计算每个频率系数的能量
    energy = np.abs(dct_block)
    
    # 使用指数权重，低频系数权重更高
    weights = np.exp(-np.linspace(0, 1, len(dct_block)))  # 指数衰减权重
    weighted_energy = energy * weights
    
    total_energy = np.sum(weighted_energy)
    
    # 计算累积能量
    cumulative_energy = np.cumsum(weighted_energy)
    
    # 找到累积能量达到阈值的频率系数数量
    num_coeffs = np.searchsorted(cumulative_energy, energy_threshold * total_energy) + 1
    
    return num_coeffs


# 对每个块计算所需的频率系数数量
# coeffs_per_block = [calculate_energy_threshold(block) for block in dct_blocks]
# coeffs_per_block = [calculate_weighted_energy_threshold(block) for block in dct_blocks]
# coeffs_per_block = [68, 335, 409, 406, 275, 326, 279, 306, 330, 314, 325, 345, 275, 272, 265, 269, 263, 246, 277, 329, 323, 309, 328, 322, 312, 278, 258, 273, 257, 442, 368, 344, 299, 248, 167, 85, 267, 360, 310, 264, 358, 362, 498, 574, 486, 377, 402, 247, 252, 174, 85, 88, 213, 253, 284, 276, 326, 299, 306, 323, 369, 329, 326, 363, 361, 373, 369, 368, 309, 337, 338, 383, 324, 365, 330, 715, 333, 427, 347, 337, 332, 289, 372, 378, 305, 346, 301, 250, 248, 246, 266, 264, 276, 307, 370, 421, 422, 399, 694, 694, 365, 412, 351, 367, 383, 329, 327, 474, 490, 380, 312, 270, 303, 342, 390, 436, 407, 262, 243, 267, 302, 393, 300, 268, 472, 514, 388, 361, 304, 346, 351, 320, 338, 308, 310, 265, 313, 255, 251, 265, 410, 251, 232, 255, 244, 268, 267, 271, 235, 239, 213, 182, 251, 251, 178, 1, 1, 1, 1, 1]
# coeffs_per_block = [68, 338, 449, 406, 313, 346, 308, 315, 330, 314, 360, 389, 275, 272, 269, 333, 311, 266, 313, 361, 337, 330, 328, 324, 314, 296, 270, 274, 267, 463, 430, 391, 318, 267, 184, 190, 276, 360, 310, 293, 407, 474, 498, 574, 543, 377, 402, 312, 301, 246, 85, 176, 242, 307, 310, 310, 363, 301, 312, 362, 389, 345, 362, 374, 509, 404, 399, 510, 426, 375, 338, 706, 333, 365, 339, 744, 396, 427, 403, 368, 355, 370, 426, 378, 341, 365, 303, 250, 248, 256, 267, 274, 304, 316, 370, 421, 445, 399, 707, 694, 731, 412, 426, 441, 383, 415, 329, 474, 490, 411, 312, 270, 304, 342, 390, 455, 407, 262, 271, 267, 329, 393, 300, 269, 472, 528, 507, 361, 339, 354, 416, 335, 391, 331, 312, 267, 322, 257, 308, 303, 431, 260, 234, 255, 250, 312, 275, 280, 239, 242, 241, 271, 251, 273, 182, 1, 1, 1, 1, 1]
# coeffs_per_block = [68, 338, 464, 406, 324, 346, 337, 315, 337, 314, 392, 389, 311, 305, 333, 335, 311, 319, 313, 365, 337, 330, 328, 327, 314, 303, 270, 297, 270, 488, 430, 418, 318, 268, 184, 244, 299, 360, 310, 293, 407, 474, 498, 610, 556, 476, 422, 312, 308, 246, 246, 244, 242, 311, 348, 311, 363, 356, 363, 362, 397, 489, 386, 419, 621, 431, 423, 622, 481, 381, 345, 706, 358, 469, 436, 773, 572, 505, 462, 397, 422, 451, 497, 378, 341, 408, 342, 306, 255, 256, 267, 311, 332, 350, 446, 697, 580, 499, 707, 735, 731, 417, 441, 620, 383, 429, 329, 474, 490, 473, 312, 271, 309, 342, 396, 455, 429, 327, 271, 267, 329, 393, 300, 298, 494, 528, 517, 386, 361, 365, 416, 335, 391, 331, 338, 307, 322, 377, 364, 316, 431, 265, 265, 265, 274, 348, 275, 280, 252, 242, 267, 274, 251, 273, 241, 1, 1, 1, 1, 1]

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