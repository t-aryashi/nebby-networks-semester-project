import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob, os, re
from scipy.signal import butter, filtfilt

def lowpass_filter(signal, cutoff_hz, fs_hz, order=4):
    """Remove noise faster than 1/RTT as the paper describes."""
    nyq = 0.5 * fs_hz
    normal_cutoff = cutoff_hz / nyq
    normal_cutoff = min(normal_cutoff, 0.99)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

files = glob.glob("../candidates-measurements/*_tcp.csv")
fig, ax = plt.subplots(figsize=(14, 5))
colors = {'cubic': 'steelblue', 'reno': 'tomato', 'bbr': 'seagreen'}
styles = {'cubic': '-',  'reno': '--', 'bbr': ':'}


fig2, ax2 = plt.subplots(figsize=(14, 5))


SERVER_IP = "10.0.0.1"   # Mahimahi host is always this

for f in files:
    cc_match = re.search(r'cc-(\w+)_', f)
    if not cc_match:
        continue
    cc = cc_match.group(1)

    df = pd.read_csv(f)
    df.columns = df.columns.str.strip()

    # Convert to numeric safely
    for col in ['tcp.len', 'tcp.seq', 'tcp.ack', 'frame.time_relative']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['frame.time_relative', 'ip.src'])
    df = df.sort_values('frame.time_relative').reset_index(drop=True)

    # --- PROBLEM 1 FIX: separate directions using ip.src ---
    # Data packets: server → client (ip.src == SERVER_IP, tcp.len > 0)
    data = df[(df['ip.src'] == SERVER_IP) & (df['tcp.len'] > 0)].copy()
    # ACK packets: client → server (ip.src != SERVER_IP)
    acks = df[df['ip.src'] != SERVER_IP].copy()

    if data.empty or acks.empty:
        print(f"WARNING: {f} - missing one direction. Check SERVER_IP={SERVER_IP}")
        continue

    # --- PROBLEM 3 FIX: use seq + len, not just seq ---
    data['seq_end'] = data['tcp.seq'] + data['tcp.len']


    # --- RTT COMPUTATION ---
    rtt_times = []
    rtt_values = []

    ack_idx = 0
    acks_sorted = acks.sort_values('frame.time_relative').reset_index(drop=True)

    for i, row in data.iterrows():
        seq_end = row['seq_end']
        t_data_pkt = row['frame.time_relative']

        # Move forward in ACKs to find matching ACK
        while ack_idx < len(acks_sorted) and acks_sorted.loc[ack_idx, 'tcp.ack'] < seq_end:
            ack_idx += 1

        if ack_idx >= len(acks_sorted):
            break

        t_ack_pkt = acks_sorted.loc[ack_idx, 'frame.time_relative']
        rtt = t_ack_pkt - t_data_pkt

        if rtt > 0:
            rtt_times.append(t_data_pkt)
            rtt_values.append(rtt)

    
    # --- PROBLEM 2 FIX: running max for both seq_end and ack ---
    data['max_seq_end'] = data['seq_end'].cummax()

    acks['max_ack'] = acks['tcp.ack'].cummax()

    # Interpolate max_ack at each data packet's timestamp
    # Merge both series on a common time index
    t_data = data['frame.time_relative'].values
    t_acks = acks['frame.time_relative'].values
    v_acks = acks['max_ack'].values

    # Forward-fill max_ack at data packet times
    max_ack_at_data = np.interp(t_data, t_acks, v_acks,
                                 left=v_acks[0], right=v_acks[-1])

    bif_raw = np.maximum(data['max_seq_end'].values - max_ack_at_data, 0)

    # --- PROBLEM 5 FIX: low-pass filter at 1/RTT ---
    # Estimate RTT from filename (delay param is one-way, RTT = 2x)
    delay_match = re.search(r'delay(\d+)', f) or re.search(r'_(\d+)ms', f)
    rtt_s = 0.1  # default 100ms RTT
    
    # Resample to uniform time grid for filtering
    t_uniform = np.linspace(t_data[0], t_data[-1], len(t_data))
    bif_uniform = np.interp(t_uniform, t_data, bif_raw)

    # Sampling frequency estimate
    dt = np.median(np.diff(t_uniform))
    if dt > 0:
        fs = 1.0 / dt
        cutoff = 1.0 / rtt_s   # remove everything faster than 1 RTT
        try:
            bif_smooth = lowpass_filter(bif_uniform, cutoff, fs)
        except Exception:
            bif_smooth = bif_uniform
    else:
        bif_smooth = bif_uniform

    label = os.path.basename(f).replace('_tcp.csv', '')
    ax.plot(t_uniform, bif_smooth / 1024,   # convert to KB
            color=colors.get(cc, 'gray'),
            linestyle=styles.get(cc, '-'),
            linewidth=1.4, alpha=0.85, label=label)
    
    ax2.plot(rtt_times, rtt_values,
         color=colors.get(cc, 'gray'),
         linestyle=styles.get(cc, '-'),
         linewidth=1.2, alpha=0.8,
         label=label)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Bytes in Flight (KB)")
ax.set_title("BiF traces by CCA — Nebby methodology")
ax.legend(fontsize=7, loc='upper right', ncol=2)
plt.tight_layout()
plt.savefig("bif_traces.png", dpi=150)
plt.show()

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("RTT (s)")
ax2.set_title("RTT vs Time")
ax2.legend(fontsize=7)

plt.tight_layout()
plt.savefig("rtt_traces.png", dpi=150)
plt.show()