#!/usr/bin/env python
import struct
import wave
import numpy as np
import scipy as sp
from scipy.fftpack import dct, idct

# 参数设定
BLOCK_SIZE = 1024  # 每个块的大小
N_COEFF = 200      # 每个DCT块保留的系数数目，可根据需要调整

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

# 简单量化：这里使用16-bit uniform quantizer进行量化
# 实际中可根据频率分量能量分布使用更高级的水填充(waterfilling)策略动态分配比特率
max_val = np.max(np.abs(truncated_dct_blocks))
if max_val == 0:
    max_val = 1e-9
quant_scale = (2**15 - 1) / max_val
quantized = np.round(truncated_dct_blocks * quant_scale).astype(np.int16)

# 存储用于解压的信息（如量化比例等）
# 格式：头部存储必要的参数，然后存储系数数据
# 存储格式： 
#   1) 文件头(包含BLOCK_SIZE, N_COEFF, num_blocks, quant_scale, pad_length, framerate)
#   2) 量化后的DCT块系数
fout = open('compressed', 'wb')
fout.write(struct.pack('<i', BLOCK_SIZE))
fout.write(struct.pack('<i', N_COEFF))
fout.write(struct.pack('<i', num_blocks))
fout.write(struct.pack('<f', quant_scale))
fout.write(struct.pack('<i', pad_length))
fout.write(struct.pack('<i', framerate))

# 写入量化数据
for block in quantized:
    for coef in block:
        fout.write(struct.pack('<h', coef))
fout.close()
