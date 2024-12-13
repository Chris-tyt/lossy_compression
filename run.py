# run.py
import subprocess

NUMCUT = 200
# NUMCUT = int(sys.argv[1]) if len(sys.argv) > 1 else 35000
compress_file = "base_1212.py"
decompress_file = "de_base_1212.py"

# 压缩音频
print("Compressing audio...")
subprocess.run(["python", compress_file, str(NUMCUT)])

# 解压音频
print("Decompressing audio...")
subprocess.run(["python", decompress_file, str(NUMCUT)])

# 计算均方误差
print("Computing MSE...")
subprocess.run(["python", "computemse.py"])

print("Process completed.")