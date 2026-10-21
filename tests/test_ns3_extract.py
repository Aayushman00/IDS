"""Smallest check: a synthetic 10-packet SYN burst yields the expected features.
Run (WSL): ~/venvs/ids/bin/python -m pytest tests/test_ns3_extract.py"""
import socket

import dpkt

import extract_features as ef  # ns3_sim/ is on pytest's pythonpath (pytest.ini)


def pkt(src, dst, flags, n=0):
    tcp = dpkt.tcp.TCP(sport=40000 + n, dport=9, flags=flags)
    ip = dpkt.ip.IP(src=socket.inet_aton(src), dst=socket.inet_aton(dst), p=6, ttl=64, data=tcp)
    ip.len += len(tcp)
    return bytes(dpkt.ethernet.Ethernet(type=dpkt.ethernet.ETH_TYPE_IP, data=ip))


def test_syn_burst(tmp_path):
    p = tmp_path / "t.pcap"
    with open(p, "wb") as fh:
        w = dpkt.pcap.Writer(fh)
        for i in range(10):
            w.writepkt(pkt("10.1.2.2", ef.SERVER, dpkt.tcp.TH_SYN, i), ts=i * 0.01)
        w.writepkt(pkt(ef.SERVER, "10.1.9.9", dpkt.tcp.TH_RST), ts=1)  # other host: must not mix in
    pk = ef.read_packets(p)["10.1.2.2"]
    f = ef.window_features(pk[:10])
    assert len(pk) == 10
    assert f["Number"] == 10 and f["syn_count"] == 10 and f["TCP"] == 1 and f["UDP"] == 0
    assert abs(f["Rate"] - 10 / 0.09) < 1e-6 and f["Drate"] == 0 and f["Weight"] == 100 and f["Variance"] == 0
    assert set(ef.FEATURES) <= set(f)
