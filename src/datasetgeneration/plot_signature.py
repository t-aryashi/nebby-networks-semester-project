import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Pick one CSV for each class to compare
targets = ['cubic', 'reno', 'bbr']
plt.figure(figsize=(12, 6))

for cc in targets:
    # Find the first file that matches the CC
    files = glob.glob(f"../candidates-measurements/cc-{cc}_*")
    if not files: continue
    
    df = pd.read_csv(files[0])
    
        # More accurate BiF: (Current Seq + Len) - (Current Max Ack)
    df['max_ack_so_far'] = df['tcp.ack'].cummax()
    df['bif_actual'] = (df['tcp.seq'] + df['tcp.len']) - df['max_ack_so_far']
    # Smooth it to see the "shape"
    df['bif_smoothed'] = df['bif_actual'].rolling(window=50).mean()

    plt.plot(df['frame.time_relative'], df['bif_smoothed'], label=cc)

plt.title("Comparison of CCA BiF Signatures")
plt.xlabel("Time (s)")
plt.ylabel("Bytes in Flight")
plt.legend()
plt.grid(True)
plt.savefig("bif_comparison.png")
print("Saved comparison plot to bif_comparison.png")