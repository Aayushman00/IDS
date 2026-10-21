"""pcap (ns-3 gateway tap) + labels.csv -> CICIoT2023-style feature rows.

CICIoT2023 computes its 46 features over windows of 10 packets. We do the same per HOST:
a window = 10 consecutive IP packets sent BY one host to the server. Replies are dropped: in real
CICIoT2023 Drate==0 and Srate==Rate (checked on 166k rows), i.e. only the sender's direction is measured.
The original extractor is not public in detail, so every feature below is an APPROXIMATION of
its documented meaning (see FEATURES). Expect domain shift vs. the real dataset; known weak spots:
  * IAT: CICIoT2023 values are huge/odd (~1e8); here it is the mean inter-arrival time in seconds.
  * ARP/DHCP/LLC/HTTP(S)/DNS/SSH: derived from ports/EtherType only; ns-3 apps never speak them.
  * Duration (CICIoT2023: TTL) = mean IP TTL.
usage: python extract_features.py results/gw-0-0.pcap results/labels.csv results/features.csv
"""
import csv
import socket
import sys
from collections import defaultdict

import dpkt
import numpy as np

WIN = 10
SERVER = "10.1.3.2"
# order = outputs/cache/meta.json feature_names (what the scaler/model expect)
FEATURES = ['flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate',
            'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
            'ack_count', 'syn_count', 'fin_count', 'urg_count', 'rst_count', 'HTTP', 'HTTPS', 'DNS', 'SSH',
            'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std',
            'Tot size', 'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight']
F = dpkt.tcp


def read_packets(path):
    """-> {host: [pkt dict]} in capture order."""
    per_host = defaultdict(list)
    with open(path, "rb") as fh:
        for ts, buf in dpkt.pcap.Reader(fh):
            eth = dpkt.ethernet.Ethernet(buf)
            ipk = eth.data
            if not isinstance(ipk, dpkt.ip.IP):
                continue  # ARP etc. dropped
            src, dst = socket.inet_ntoa(ipk.src), socket.inet_ntoa(ipk.dst)
            l4, sport, dport, flags, l4hl = ipk.data, 0, 0, 0, 8  # icmp header = 8
            if isinstance(l4, dpkt.tcp.TCP):
                sport, dport, flags, l4hl = l4.sport, l4.dport, l4.flags, l4.off * 4
            elif isinstance(l4, dpkt.udp.UDP):
                sport, dport, l4hl = l4.sport, l4.dport, 8
            if dst != SERVER:
                continue
            out, host = True, src
            # +1: tap is behind the gateway, which decremented TTL; real captures see the sender's initial TTL
            per_host[host].append(dict(ts=ts, len=14 + ipk.len, hdr=ipk.hl * 4 + l4hl, proto=ipk.p, ttl=ipk.ttl + 1,
                                       ports=(sport, dport), flags=flags, out=out))
    return per_host


def window_features(w):
    ts = np.array([p["ts"] for p in w])
    ln = np.array([p["len"] for p in w], float)
    dur = ts[-1] - ts[0]
    protos = [p["proto"] for p in w]
    flag = lambda m: float(np.mean([bool(p["flags"] & m) for p in w]))
    cnt = lambda m: float(sum(bool(p["flags"] & m) for p in w))
    port = lambda *ps: float(any(x in ps for p in w for x in p["ports"]))
    return {
        "flow_duration": dur, "Header_Length": np.mean([p["hdr"] for p in w]),
        "Protocol Type": max(set(protos), key=protos.count), "Duration": np.mean([p["ttl"] for p in w]),
        "Rate": len(w) / dur if dur else 0.0, "Srate": len(w) / dur if dur else 0.0,
        "Drate": 0.0,
        "fin_flag_number": flag(F.TH_FIN), "syn_flag_number": flag(F.TH_SYN), "rst_flag_number": flag(F.TH_RST),
        "psh_flag_number": flag(F.TH_PUSH), "ack_flag_number": flag(F.TH_ACK),
        "ack_count": cnt(F.TH_ACK), "syn_count": cnt(F.TH_SYN), "fin_count": cnt(F.TH_FIN),
        "urg_count": cnt(F.TH_URG), "rst_count": cnt(F.TH_RST),
        "HTTP": port(80), "HTTPS": port(443), "DNS": port(53), "SSH": port(22),
        "TCP": float(6 in protos), "UDP": float(17 in protos), "DHCP": port(67, 68), "ARP": 0.0,
        "ICMP": float(1 in protos), "IPv": 1.0, "LLC": 1.0,
        "Tot sum": ln.sum(), "Min": ln.min(), "Max": ln.max(), "AVG": ln.mean(), "Std": ln.std(),
        "Tot size": ln[-1], "IAT": float(np.diff(ts).mean()), "Number": float(len(w)),
        # Direction-pair features: in real CICIoT2023 both "directions" are the same packet list, verified on 166k rows:
        # Magnitue == sqrt(2*AVG) and Radius == sqrt(2)*Std (median rel. err 0), Covariance ~ Std^2,
        # Variance ~ 0.95 if there is size variance else 0, Weight ~ 14.9*Number (dataset constant; we use n^2).
        "Magnitue": float(np.sqrt(2 * ln.mean())), "Radius": float(np.sqrt(2) * ln.std()),
        "Covariance": float(ln.var()), "Variance": 0.95 if ln.var() > 0 else 0.0, "Weight": float(len(w) ** 2)}


GRACE = 5.0  # attack traffic still drains from device queues a few seconds after the app stops


def label_of(host, t, labels):
    for lab, a, b in labels.get(host, []):
        if a <= t <= b + GRACE:
            return lab
    return "Benign"


def main(pcap, labels_csv, out_csv):
    labels = defaultdict(list)
    for r in csv.DictReader(open(labels_csv)):
        labels[r["ip"]].append((r["label"], float(r["start"]), float(r["stop"])))
    rows = []
    for host, pk in read_packets(pcap).items():
        for i in range(0, len(pk) - WIN + 1, WIN):
            w = pk[i:i + WIN]
            rows.append({"host": host, "t": w[0]["ts"], "label": label_of(host, w[0]["ts"], labels),
                         **window_features(w)})
    rows.sort(key=lambda r: r["t"])
    with open(out_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, ["host", "t", "label"] + FEATURES)
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} windows -> {out_csv}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
