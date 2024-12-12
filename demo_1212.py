import wave
import struct
import numpy as np
import scipy.fftpack as fftpack
import heapq

# ======================= 参数配置 =======================
AUDIO_FILE = 'step.wav'
BLOCK_SIZE = 1024             # 分块大小
TOTAL_BITRATE = 64.0          # 总比特率资源(示例值)
SILENCE_BLOCK_SEARCH = 100    # 用于从前多少帧中选能量最低帧作为静音估计
# ========================================================


# ========== 读取音频 ==========
fin = wave.open(AUDIO_FILE, 'r')
(nchannels, sampwidth, framerate, nframes, comptype, compname) = fin.getparams()
inbytes = fin.readframes(nframes)
fin.close()

# 将字节数据转换为整数样本（假设16bit PCM）
samplesint = [struct.unpack('<h', inbytes[2*i:2*i+2])[0] for i in range(nframes)]
samplesfloat = np.array([float(x)/(2**15) for x in samplesint], dtype=np.float64)

# ========== 自动选择静音区间 ==========
# 将前SILENCE_BLOCK_SEARCH个块计算能量，选最低能量的块作为静音
num_frames_silence_search = min(len(samplesfloat)//BLOCK_SIZE, SILENCE_BLOCK_SEARCH)
frame_energies = []
for i in range(num_frames_silence_search):
    frame = samplesfloat[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE]
    energy = np.mean(frame**2)
    frame_energies.append((energy, i))
frame_energies.sort(key=lambda x: x[0])
lowest_energy_frame = frame_energies[0][1]

silence_start = lowest_energy_frame * BLOCK_SIZE
silence_end = silence_start + BLOCK_SIZE
noise_samples = samplesfloat[silence_start:silence_end]
noise_power = np.mean(noise_samples**2)

# ========== 分块处理 ==========
num_blocks = len(samplesfloat)//BLOCK_SIZE
blocks = samplesfloat[:num_blocks*BLOCK_SIZE].reshape(num_blocks, BLOCK_SIZE)

# ========== 对每个块进行DFT ==========
freq_blocks = fftpack.fft(blocks, axis=1)
half_size = BLOCK_SIZE // 2
freq_blocks = freq_blocks[:, :half_size]

# 能量和SNR
energies = np.abs(freq_blocks)**2
SNR = energies / noise_power

# ========== 注水法分配比特率 ==========

def waterfilling_alloc(snr_values, total_rate, max_iter=100, W_low=0, W_high=100):
    def total_rate_given_W(W):
        rates = np.maximum(0, W - 1.0/snr_values)
        return np.sum(rates)
    for _ in range(max_iter):
        W_mid = (W_low + W_high)/2
        current_rate = total_rate_given_W(W_mid)
        if current_rate < total_rate:
            W_low = W_mid
        else:
            W_high = W_mid
    W_final = (W_low + W_high)/2
    final_rates = np.maximum(0, W_final - 1.0/snr_values)
    return final_rates

bitrate_allocation = []
# 简单假设每个块都有相同的总比特率资源（可根据需要改进分配策略）
for b in range(num_blocks):
    snr_block = SNR[b, :]
    rates = waterfilling_alloc(snr_block, TOTAL_BITRATE)
    bitrate_allocation.append(rates)
bitrate_allocation = np.array(bitrate_allocation)

# ========== 量化 ==========
# 简化量化：根据比特率决定量化级数 L = 2^R
# 假定幅值范围，以当前块的频率分量最大值为参考。
quantized_blocks = []
for b in range(num_blocks):
    block_freq = freq_blocks[b, :]
    rates = bitrate_allocation[b, :]
    block_max = np.max(np.abs(block_freq))
    if block_max == 0:
        block_max = 1e-9
    quant_block = []
    for i, val in enumerate(block_freq):
        r = rates[i]
        if r <= 0:
            # 不分配比特，置0
            q_val = 0
        else:
            L = int(2**r)
            if L < 2:
                L = 2
            step = (2*block_max)/L
            q_index = int(np.floor((val + block_max)/step))
            if q_index < 0:
                q_index = 0
            elif q_index >= L:
                q_index = L-1
            # 将索引中心化，以0为中心
            q_val = q_index - (L/2.0)
        quant_block.append(q_val)
    quantized_blocks.append(quant_block)

quantized_blocks = np.array(quantized_blocks)

# ========== 霍夫曼编码 ==========

class HuffmanNode:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(symbols, freqs):
    pq = []
    for s, f in zip(symbols, freqs):
        heapq.heappush(pq, HuffmanNode(freq=f, symbol=s))

    while len(pq) > 1:
        node1 = heapq.heappop(pq)
        node2 = heapq.heappop(pq)
        merged = HuffmanNode(freq=node1.freq+node2.freq, left=node1, right=node2)
        heapq.heappush(pq, merged)

    return pq[0]

def build_huffman_codes(root):
    codes = {}
    def traverse(node, prefix=""):
        if node.symbol is not None:
            codes[node.symbol] = prefix
            return
        if node.left:
            traverse(node.left, prefix + "0")
        if node.right:
            traverse(node.right, prefix + "1")
    traverse(root)
    return codes

# 将量化结果整合成一维
all_quant_vals = quantized_blocks.flatten()

# 霍夫曼树构建前，需要将可能为浮点的量化值变成可哈希的符号（转成字符串或四舍五入为整数）
# 这里简单转为字符串以便作为字典key
all_quant_vals_str = all_quant_vals.astype(str)

unique_vals, counts = np.unique(all_quant_vals_str, return_counts=True)
root = build_huffman_tree(unique_vals, counts)
codes_dict = build_huffman_codes(root)

encoded_stream = "".join([codes_dict[val] for val in all_quant_vals_str])

print("Noise Power:", noise_power)
print("Encoded bitstream length (bits):", len(encoded_stream))
print("Done.")
