import csv
import matplotlib.pyplot as plt

# =========================
# FILE PATHS
# =========================
# Replace these with your actual generated CSV filenames
file_cubic = "../candidates-measurements/cubic3_tcp.csv"
file_reno  = "../candidates-measurements/reno3_tcp.csv" 
file_bbr   = "../candidates-measurements/bbr3_tcp.csv"

# =========================
# Compute BIF (Bytes In Flight)
# =========================
def compute_bif(file):
    time = []
    bif = []
    max_seq = 0
    max_ack = 0

    try:
        with open(file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    t = float(row["frame.time_relative"])
                    seq = row["tcp.seq"]
                    ack = row["tcp.ack"]

                    if seq: max_seq = max(max_seq, int(seq))
                    if ack: max_ack = max(max_ack, int(ack))

                    # BIF = Data sent but not yet acknowledged
                    current_bif = max_seq - max_ack
                    time.append(t)
                    bif.append(current_bif)
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        print(f"Warning: {file} not found. Skipping.")
        return [], []

    return time, bif

# # =========================
# # Smoothing Function
# # =========================
# def smooth(data, window=20):
#     if not data: return []
#     smoothed = []
#     for i in range(len(data)):
#         start = max(0, i - window)
#         avg = sum(data[start:i+1]) / (i - start + 1)
#         smoothed.append(avg)
#     return smoothed

# =========================
# Load and Process Data
# =========================
data_map = {
    "CUBIC": compute_bif(file_cubic),
    "RENO":  compute_bif(file_reno),
    "BBR":   compute_bif(file_bbr)
}

# =========================
# Plotting
# =========================
plt.figure(figsize=(14, 7))

colors = {"CUBIC": "tab:blue", "RENO": "tab:red", "BBR": "tab:green"}

for label, (t, b) in data_map.items():
    if t:
        # b_smoothed = smooth(b, window=30)
        # plt.plot(t, b_smoothed, label=label, color=colors[label], linewidth=2)
        plt.plot(t, b, label=label, color=colors[label], linewidth=1)

plt.xlabel("Time (seconds)")
plt.ylabel("Bytes In Flight (BIF)")
plt.title("TCP Congestion Control Comparison: CUBIC vs RENO vs BBR")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.show()