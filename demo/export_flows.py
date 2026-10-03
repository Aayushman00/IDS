"""Sample held-out CICIoT2023 test flows + cached CNN-LSTM scores -> demo/flows.js (+flows.json).
Run: python demo/export_flows.py   (numpy + joblib only; scores come from outputs/cache/proba_CNN-LSTM.npy,
produced by the trained CNN-LSTM on this exact X_test)."""
import json, pathlib, warnings
import numpy as np, joblib
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
C = ROOT / "outputs/cache"
meta = json.load(open(C / "meta.json"))
X = np.load(C / "X_test.npy"); y = np.load(C / "y_test.npy")
sub = np.load(C / "sub_test_codes.npy"); p = np.load(C / "proba_CNN-LSTM.npy")[:, 1]
names = meta["sub_test_names"]; feats = meta["feature_names"]
raw = joblib.load(ROOT / "outputs/models/scaler.joblib").inverse_transform(X)

def fam(n):
    for k, f in (("DDoS", "DDoS"), ("DoS", "DoS"), ("Mirai", "Mirai"), ("Recon", "Recon"), ("Vulnerability", "Recon"),
                 ("Spoof", "Spoofing"), ("MITM", "Spoofing"), ("BruteForce", "BruteForce"), ("Backdoor", "Backdoor")):
        if k in n: return f
    return "Benign" if n == "BenignTraffic" else "Web"

KEEP = ["Header_Length", "Protocol Type", "Rate", "Srate", "Drate", "syn_flag_number", "ack_flag_number", "rst_flag_number",
        "psh_flag_number", "fin_flag_number", "syn_count", "ack_count", "TCP", "UDP", "ICMP", "ARP", "HTTP", "HTTPS", "DNS",
        "Tot size", "AVG", "IAT", "Number"]
idx = [feats.index(k) for k in KEEP]
fams = np.array([fam(names[c]) for c in sub])
rng = np.random.default_rng(7)
out = []
for f in sorted(set(fams)):
    pool = np.flatnonzero(fams == f)
    for i in rng.choice(pool, min(300 if f == "Benign" else 150, len(pool)), replace=False):
        out.append({"fam": f, "sub": names[sub[i]], "y": int(y[i]), "p": round(float(p[i]), 4),
                    "f": {k: round(float(raw[i, j]), 3) for k, j in zip(KEEP, idx)}})
rng.shuffle(out)
assert out and 0 <= min(o["p"] for o in out) and max(o["p"] for o in out) <= 1
need = {"Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce", "Backdoor"}
assert need <= {o["fam"] for o in out}, need - {o["fam"] for o in out}
d = ROOT / "demo"
(d / "flows.json").write_text(json.dumps(out))
(d / "flows.js").write_text("window.FLOWS=" + json.dumps(out) + ";")
print(len(out), "flows;", {f: sum(o["fam"] == f for o in out) for f in sorted(need)})
