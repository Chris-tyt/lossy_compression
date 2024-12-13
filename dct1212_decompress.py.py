#!/usr/bin/env python
import struct
import wave
import numpy as np
import scipy as sp
from scipy.fftpack import dct, idct

# 从文件中读取头部信息和量化后的系数
fin = open('compressed', 'rb')
header = fin.read(4*6) # 读取6个整数/浮点参数
BLOCK_SIZE = struct.unpack('<i', header[0:4])[0]
N_COEFF = struct.unpack('<i', header[4:8])[0]
num_blocks = struct.unpack('<i', header[8:12])[0]
quant_scale = struct.unpack('<f', header[12:16])[0]
pad_length = struct.unpack('<i', header[16:20])[0]
framerate = struct.unpack('<i', header[20:24])[0]

# 剩下的数据为量化的DCT系数
compressed_data = fin.read()
fin.close()

total_coefs = num_blocks * N_COEFF
coeffs_int16 = [struct.unpack('<h', compressed_data[2*i:2*i+2])[0] for i in range(total_coefs)]

# 将量化整数系数转换回浮点形式
coeffs = np.array(coeffs_int16, dtype=np.float32).reshape(num_blocks, N_COEFF)
dct_blocks = coeffs / quant_scale

# 填充DCT后的块，使其恢复到BLOCK_SIZE大小的DCT系数长度
# 剩余系数填0，因为只保留了低频分量
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
