import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob, os, re
from scipy.signal import butter, filtfilt

def lowpass_filter(signal, cutoff_hz, fs_hz, order=4):
    nyq = 0.5 * fs_hz
    normal_cutoff = min(cutoff_hz / nyq, 0.99)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

def detect_directions(df):
    # Heuristic: sender has more tcp.len > 0 packets
    send_counts = df[df['tcp.len'] > 0]['ip.src'].value_counts()
    if len(send_counts) < 1:
        return None, None

    sender_ip = send_counts.idxmax()
    receiver_ip = [ip for ip in df['ip.src'].unique() if ip != sender_ip]

    if not receiver_ip:
        return None, None

    return sender_ip, receiver_ip[0]

def compute_rtt(data, acks):
    rtt_times, rtt_values = [], []

    acks = acks.sort_values('frame.time_relative').reset_index(drop=True)
    ack_idx = 0

    for _, row in data.iterrows():
        seq_end = row['tcp.seq'] + row['tcp.len']
        t_data = row['frame.time_relative']

        while ack_idx < len(acks) and acks.loc[ack_idx, 'tcp.ack'] < seq_end:
            ack_idx += 1

        if ack_idx >= len(acks):
            break

        t_ack = acks.loc[ack_idx, 'frame.time_relative']
        rtt = t_ack - t_data

        if rtt > 0:
            rtt_times.append(t_data)
            rtt_values.append(rtt)

    return np.array(rtt_times), np.array(rtt_values)

def compute_bif(data, acks):
    data['seq_end'] = data['tcp.seq'] + data['tcp.len']
    data['max_seq_end'] = data['seq_end'].cummax()

    acks['max_ack'] = acks['tcp.ack'].cummax()

    t_data = data['frame.time_relative'].values
    t_acks = acks['frame.time_relative'].values
    v_acks = acks['max_ack'].values

    max_ack_interp = np.interp(t_data, t_acks, v_acks,
                              left=v_acks[0], right=v_acks[-1])

    bif = np.maximum(data['max_seq_end'].values - max_ack_interp, 0)

    return t_data, bif

def smooth_signal(t, signal, rtt_s=0.1):
    t_uniform = np.linspace(t[0], t[-1], len(t))
    sig_uniform = np.interp(t_uniform, t, signal)

    dt = np.median(np.diff(t_uniform))
    if dt <= 0:
        return t_uniform, sig_uniform

    fs = 1.0 / dt
    cutoff = 1.0 / rtt_s

    try:
        sig_smooth = lowpass_filter(sig_uniform, cutoff, fs)
    except:
        sig_smooth = sig_uniform

    return t_uniform, sig_smooth

# ---------------- MAIN ----------------

files = glob.glob("../candidates-measurements/*_tcp.csv")

fig_bif, ax_bif = plt.subplots(figsize=(14, 5))
fig_rtt, ax_rtt = plt.subplots(figsize=(14, 5))

for f in files:
    print(f"Processing: {f}")

    df = pd.read_csv(f)
    df.columns = df.columns.str.strip()

    for col in ['tcp.len', 'tcp.seq', 'tcp.ack', 'frame.time_relative']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['frame.time_relative', 'ip.src'])
    df = df.sort_values('frame.time_relative').reset_index(drop=True)

    sender_ip, receiver_ip = detect_directions(df)
    if sender_ip is None:
        continue

    data = df[(df['ip.src'] == sender_ip) & (df['tcp.len'] > 0)].copy()
    acks = df[df['ip.src'] == receiver_ip].copy()

    if data.empty or acks.empty:
        continue

    # ---- RTT ----
    rtt_t, rtt_v = compute_rtt(data, acks)

    # ---- BiF ----
    t_data, bif = compute_bif(data, acks)

    # ---- Smooth ----
    t_s, bif_s = smooth_signal(t_data, bif)

    label = os.path.basename(f).replace('_tcp.csv', '')

    # Plot ALL in same figure
    ax_bif.plot(t_s, bif_s / 1024, linewidth=1.3, label=label)
    ax_rtt.plot(rtt_t, rtt_v, linewidth=1.1, label=label)

# ---- Final formatting (ONLY ONCE) ----

ax_bif.set_xlabel("Time (s)")
ax_bif.set_ylabel("Bytes in Flight (KB)")
ax_bif.set_title("BiF — All Traces")
ax_bif.legend(fontsize=7)
ax_bif.grid(True)

ax_rtt.set_xlabel("Time (s)")
ax_rtt.set_ylabel("RTT (s)")
ax_rtt.set_title("RTT — All Traces")
ax_rtt.legend(fontsize=7)
ax_rtt.grid(True)

plt.tight_layout()

fig_bif.savefig("all_bif.png", dpi=150)
fig_rtt.savefig("all_rtt.png", dpi=150)

plt.show()