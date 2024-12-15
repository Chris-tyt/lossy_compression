#!/usr/bin/env python

import subprocess
import numpy as np
import wave
import struct
import os
import time

def calculate_block_mse(samples1, samples2, block_num, block_size):
    start_idx = block_num * block_size
    end_idx = (block_num + 1) * block_size
    return np.mean((samples1[start_idx:end_idx] - samples2[start_idx:end_idx])**2)

def read_wav_file(filename):
    fin = wave.open(filename, 'r')
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = fin.getparams()
    inbytes = fin.readframes(nframes)
    fin.close()

    samplesint = [struct.unpack('<h',inbytes[2*i:2*i+2]) for i in range(nframes)]
    samplesfloat = [float(x[0])/(2**15) for x in samplesint]
    return np.array(samplesfloat)

file_name = 'optimization_results.txt'

# 清空或创建新的结果文件
with open(file_name, 'w') as f:
    f.write("Block_Num,Num_Waves,MSE\n")  # 写入CSV表头

# 设定参数
block_size = 2048
min_waves = 250
max_waves = 700
wave_step = 20

# 读取原始信号
original_samples = read_wav_file('step.wav')
num_blocks = len(original_samples) // block_size

# 在主循环开始前添加计数器
less_than_threshold = 0
greater_than_threshold = 0

# 记录开始时间
total_start_time = time.time()

# 主循环
for block_num in range(num_blocks):
    block_start_time = time.time()  # 记录每个块的开始时间
    print(f"Processing block {block_num}")
    # with open('optimization_results.txt', 'a') as f:
    #     f.write(f"Processing block {block_num}\n")
    #     f.write("Block_Num,Num_Waves,MSE\n")
    if True:
        # 先用默认参数运行一次来检查初始MSE
        subprocess.run(['python', 'compress_opt.py'], 
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        subprocess.run(['python', 'decompress.py'], 
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists('out.wav'):
            decompressed_samples = read_wav_file('out.wav')
            initial_mse = calculate_block_mse(original_samples, decompressed_samples, block_num, block_size)
            
            # 根据初始MSE决定搜索范围
            if initial_mse < 4e-5:
                wave_range = range(439, min_waves, -wave_step)
                less_than_threshold += 1
            else:
                wave_range = range(429, max_waves, wave_step)
                greater_than_threshold += 1
            
            found_valid_result = False
            valid_results_count = 0
            
            # 在确定的范围内搜索
            for num_waves in wave_range:
                print(f"Processing block {block_num} with {num_waves} waves")
                subprocess.run(['python', 'compress_opt.py', str(block_num), str(num_waves)], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                subprocess.run(['python', 'decompress.py'], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if os.path.exists('out.wav'):
                    decompressed_samples = read_wav_file('out.wav')
                    mse = calculate_block_mse(original_samples, decompressed_samples, block_num, block_size)
                    
                    if 2.5e-5 < mse < 8e-5:
                        found_valid_result = True
                        with open(file_name, 'a') as f:
                            f.write(f"{block_num},{num_waves},{mse}\n")
                        print(f"Found valid result - Block: {block_num}, Waves: {num_waves}, MSE: {mse}")
                        valid_results_count += 1
                        
                        if valid_results_count >= 10:
                            break
                    elif found_valid_result:
                        # 如果之前找到过有效结果，但现在不满足条件，则退出循环
                        break
            
            # 如果没有找到任何有效结果，使用极值
            if not found_valid_result:
                final_num_waves = min_waves if initial_mse < 4e-5 else max_waves
                subprocess.run(['python', 'compress_opt.py', str(block_num), str(final_num_waves)], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                subprocess.run(['python', 'decompress.py'], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if os.path.exists('out.wav'):
                    decompressed_samples = read_wav_file('out.wav')
                    mse = calculate_block_mse(original_samples, decompressed_samples, block_num, block_size)
                    with open(file_name, 'a') as f:
                        f.write(f"{block_num},{final_num_waves},{mse}\n")

    # 在每个块处理完成后计算并打印该块的处理时间
    block_end_time = time.time()
    block_duration = block_end_time - block_start_time
    print(f"Block {block_num} processing time: {block_duration:.2f} seconds")

# 计算并打印总运行时间
total_end_time = time.time()
total_duration = total_end_time - total_start_time

print(f"\nStatistics:")
print(f"Blocks with MSE < 4e-5: {less_than_threshold}")
print(f"Blocks with MSE >= 4e-5: {greater_than_threshold}")
print(f"Total blocks processed: {less_than_threshold + greater_than_threshold}")
print(f"Total running time: {total_duration:.2f} seconds")
print(f"Average time per block: {total_duration/num_blocks:.2f} seconds")
print("\nOptimization complete. Results saved to optimization_results.txt")
