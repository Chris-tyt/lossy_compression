# run.py
import subprocess

NUMCUT = 2048
# NUMCUT = int(sys.argv[1]) if len(sys.argv) > 1 else 35000
compress_file = "compress.py"
decompress_file = "decompress.py"

# Compress audio
print("Compressing audio...")
subprocess.run(["python", compress_file, str(NUMCUT)])

# Decompress audio
print("Decompressing audio...")
subprocess.run(["python", decompress_file, str(NUMCUT)])

# Compute Mean Squared Error
print("Computing MSE...")
subprocess.run(["python", "computemse.py"])

print("Process completed.")