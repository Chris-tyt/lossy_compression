#!/usr/bin/env python

# Needed for struct.pack

import struct
import wave
import numpy as np
import scipy as sp

# Open the input and output files. 'wb' is needed on some platforms
# to indicate that 'compressed' is a binary file.

fin = wave.open('step.wav','r')
(nchannels, sampwidth, framerate, nframes, comptype, compname) = fin.getparams()
inbytes = fin.readframes(nframes)
fin.close()

samplesint = [struct.unpack('<h',inbytes[2*i:2*i+2]) for i in range(nframes)]

samplesfloat = [float(x[0])/(2**15) for x in samplesint]
samples1 = np.array(samplesfloat)

fin = wave.open('out.wav','r')
(nchannels, sampwidth, framerate, nframes, comptype, compname) = fin.getparams()
inbytes = fin.readframes(nframes)
fin.close()

samplesint = [struct.unpack('<h',inbytes[2*i:2*i+2]) for i in range(nframes)]

samplesfloat = [float(x[0])/(2**15) for x in samplesint]
samples2 = np.array(samplesfloat)


# 设定块大小
block_size = 2048  # 可以根据需要修改这个值

# 计算需要多少个完整的块
num_blocks = len(samples1) // block_size

# 存储所有块的MSE
mse_values = []

# 对每个块计算MSE
for i in range(num_blocks):
    start_idx = i * block_size
    end_idx = (i + 1) * block_size
    block_mse = np.mean((samples1[start_idx:end_idx] - samples2[start_idx:end_idx])**2)
    mse_values.append((i, block_mse))
    print(f'Block {i} MSE: {block_mse}')

# 处理最后一个不完整的块（如果有的话）
if len(samples1) % block_size != 0:
    start_idx = num_blocks * block_size
    last_block_mse = np.mean((samples1[start_idx:] - samples2[start_idx:])**2)
    mse_values.append(('Last', last_block_mse))
    print(f'Last block MSE: {last_block_mse}')

# 打印总体MSE
total_mse = np.mean((samples1 - samples2)**2)
print(f'\nTotal MSE: {total_mse}')

# 打印前五个最大的MSE
print('\nTop 5 largest MSE values:')
sorted_mse = sorted(mse_values, key=lambda x: x[1], reverse=True)
for i, (block_num, mse) in enumerate(sorted_mse[:5]):
    print(f'Block {block_num}: MSE = {mse}')

print(sorted_mse[-10:])
print(f'Block 0: MSE = {mse_values[0]}')
print(f'Block 2: MSE = {mse_values[2]}')
print(f'Block 75: MSE = {mse_values[75]}')
print(f'Block 149: MSE = {mse_values[149]}')
