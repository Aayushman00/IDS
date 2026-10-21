// IoT IDS traffic scenario for ns-3 (copied to ns-3/scratch/ by run_sim.sh).
//
//   benign IoT devices --csma--+                       +--csma-- server
//   (10.1.1.0/24)              +-- gateway (pcap tap) -+          (10.1.3.2)
//   attackers          --csma--+   <- IDS sits here
//   (10.1.2.0/24)
//
// Timeline (duration D split in 6 slots): benign only | UDP flood | ICMP flood | SYN flood | port scan | benign only
// Outputs in --out: gw-*.pcap (gateway, server side) and labels.csv (ip,label,start,stop).
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-apps-module.h"

#include "ns3/internet-module.h"
#include "ns3/netanim-module.h"
#include "ns3/network-module.h"

#include <fstream>
#include <sstream>

using namespace ns3;

// Opens a fresh TCP connection every `interval`. portStep=0: hammer one port (SYN flood);
// portStep=1: walk ports upward (scan). Closed port -> server answers RST, as in a real stack.
class ConnApp : public Application
{
  public:
    void Setup(Ipv4Address dst, uint16_t port, uint16_t step, Time interval)
    {
        m_dst = dst; m_port = port; m_step = step; m_interval = interval;
    }

  private:
    void StartApplication() override { Fire(); }
    void StopApplication() override { m_ev.Cancel(); }
    void Fire()
    {
        Ptr<Socket> s = Socket::CreateSocket(GetNode(), TcpSocketFactory::GetTypeId());
        s->Bind();
        s->Connect(InetSocketAddress(m_dst, m_port));
        // no Close(): server's RST ends the attempt; an explicit close would add an RST from us (real floods have none)
        m_port += m_step;
        m_ev = Simulator::Schedule(m_interval, &ConnApp::Fire, this);
    }
    Ipv4Address m_dst;
    uint16_t m_port{9}, m_step{0};
    Time m_interval;
    EventId m_ev;
};

int
main(int argc, char* argv[])
{
    uint32_t nBenign = 20, nAtk = 3, seed = 1;
    double duration = 120;
    std::string out = "out", anim = "";
    double udpKbps = 320;
    CommandLine cmd(__FILE__);
    cmd.AddValue("nBenign", "benign IoT devices", nBenign);
    cmd.AddValue("nAtk", "attackers per attack type", nAtk);
    cmd.AddValue("duration", "sim seconds", duration);
    cmd.AddValue("seed", "RNG run number", seed);
    cmd.AddValue("out", "output dir (must exist)", out);
    cmd.AddValue("anim", "NetAnim XML path (empty = off). Use with a low --udpKbps, UDP flood makes huge traces", anim);
    cmd.AddValue("udpKbps", "UDP flood rate per attacker (320 = real-data rate)", udpKbps);
    cmd.Parse(argc, argv);
    RngSeedManager::SetRun(seed);

    NodeContainer gw, server, dev, atk;
    gw.Create(1); server.Create(1); dev.Create(nBenign); atk.Create(4 * nAtk);
    NodeContainer iotLan(gw, dev), atkLan(gw, atk), srvLan(gw, server);

    CsmaHelper csma;
    csma.SetChannelAttribute("DataRate", StringValue("100Mbps"));
    csma.SetChannelAttribute("Delay", TimeValue(MicroSeconds(50)));
    NetDeviceContainer dIot = csma.Install(iotLan), dAtk = csma.Install(atkLan), dSrv = csma.Install(srvLan);

    InternetStackHelper().Install(NodeContainer(gw, server, dev, atk));
    Ipv4AddressHelper ip;
    ip.SetBase("10.1.1.0", "255.255.255.0"); Ipv4InterfaceContainer iIot = ip.Assign(dIot);
    ip.SetBase("10.1.2.0", "255.255.255.0"); Ipv4InterfaceContainer iAtk = ip.Assign(dAtk);
    ip.SetBase("10.1.3.0", "255.255.255.0"); Ipv4InterfaceContainer iSrv = ip.Assign(dSrv);
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
    Ipv4Address srv = iSrv.GetAddress(1);

    csma.EnablePcap(out + "/gw", dSrv.Get(0), true); // gateway's server-side port, promiscuous

    // server listeners: CoAP-like UDP 5683, HTTPS-like TCP 443
    PacketSinkHelper udpSink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), 5683));
    PacketSinkHelper tcpSink("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), 443));
    ApplicationContainer sinks = udpSink.Install(server);
    sinks.Add(tcpSink.Install(server));
    sinks.Start(Seconds(0)); sinks.Stop(Seconds(duration));

    std::ofstream lab(out + "/labels.csv");
    lab << "ip,label,start,stop\n";
    Ptr<UniformRandomVariable> u = CreateObject<UniformRandomVariable>();
    auto ipOf = [](Ipv4InterfaceContainer& c, uint32_t i) { std::ostringstream s; c.GetAddress(i).Print(s); return s.str(); };

    // benign: periodic sensor telemetry (UDP or TCP), plus occasional ping
    for (uint32_t i = 0; i < nBenign; i++)
    {
        bool tcp = i % 2;
        OnOffHelper app(tcp ? "ns3::TcpSocketFactory" : "ns3::UdpSocketFactory",
                        InetSocketAddress(srv, tcp ? 443 : 5683));
        app.SetAttribute("DataRate", StringValue("40kbps"));
        app.SetAttribute("PacketSize", UintegerValue(u->GetInteger(60, 1000)));
        app.SetAttribute("OnTime", StringValue("ns3::ExponentialRandomVariable[Mean=2]"));
        app.SetAttribute("OffTime", StringValue("ns3::ExponentialRandomVariable[Mean=3]"));
        ApplicationContainer a = app.Install(dev.Get(i));
        a.Start(Seconds(u->GetValue(0, 2))); a.Stop(Seconds(duration));
        if (i % 5 == 0)
        {
            PingHelper p(srv);
            p.SetAttribute("VerboseMode", StringValue("Silent"));
            p.SetAttribute("Interval", TimeValue(Seconds(1)));
            ApplicationContainer pa = p.Install(dev.Get(i));
            pa.Start(Seconds(u->GetValue(0, 1))); pa.Stop(Seconds(duration));
        }
        lab << ipOf(iIot, i + 1) << ",Benign,0," << duration << "\n";
    }

    // attackers: group g=0..3 active in slot g+1
    const char* names[4] = {"DDoS-UDP_Flood", "DDoS-ICMP_Flood", "DDoS-SYN_Flood", "Recon-PortScan"};
    double slot = duration / 6;
    for (uint32_t g = 0; g < 4; g++)
    {
        double t0 = slot * (g + 1), t1 = t0 + slot;
        for (uint32_t k = 0; k < nAtk; k++)
        {
            Ptr<Node> n = atk.Get(g * nAtk + k);
            ApplicationContainer a;
            if (g == 0)
            {
                OnOffHelper f("ns3::UdpSocketFactory", InetSocketAddress(srv, 5683));
                f.SetAttribute("DataRate", StringValue(std::to_string(udpKbps) + "kbps")); // 8 B payload; 320 tuned so measured Rate ~7600 pps/host like real UDP-flood rows
                f.SetAttribute("PacketSize", UintegerValue(8));
                f.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
                f.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
                a = f.Install(n);
            }
            else if (g == 1)
            {
                PingHelper p(srv);
                p.SetAttribute("VerboseMode", StringValue("Silent"));
                p.SetAttribute("Interval", TimeValue(MilliSeconds(100))); // real ICMP flood rows: ~10 pps/host
                p.SetAttribute("Size", UintegerValue(16));
                a = p.Install(n);
            }
            else
            {
                Ptr<ConnApp> c = CreateObject<ConnApp>();
                if (g == 2) c->Setup(srv, 9, 0, MilliSeconds(90));
                else c->Setup(srv, 1, 1, MilliSeconds(30));
                n->AddApplication(c);
                a.Add(c);
            }
            a.Start(Seconds(t0 + u->GetValue(0, 0.5))); a.Stop(Seconds(t1));
            lab << ipOf(iAtk, g * nAtk + k + 1) << "," << names[g] << "," << t0 << "," << t1 << "\n";
        }
    }
    lab.close();

    // NetAnim layout (100x100 canvas): attackers grouped by type along the top, IoT devices in a grid bottom-left,
    // gateway (IDS tap) centre-right, server far right. Attackers are grey until their slot, then take the attack colour.
    // Positions are set BEFORE AnimationInterface exists: its constructor assigns random positions to unplaced nodes.
    std::unique_ptr<AnimationInterface> ai;
    if (!anim.empty())
    {
        struct Rgb { uint8_t r, g, b; };
        const Rgb atkColor[4] = {{255, 140, 0}, {235, 200, 0}, {230, 30, 30}, {200, 40, 200}};
        const char* atkShort[4] = {"UDP flood", "ICMP flood", "SYN flood", "Port scan"};
        AnimationInterface::SetConstantPosition(gw.Get(0), 62, 45);
        AnimationInterface::SetConstantPosition(server.Get(0), 92, 45);
        for (uint32_t i = 0; i < nBenign; i++)
            AnimationInterface::SetConstantPosition(dev.Get(i), 6 + 11.0 * (i % 5), 58 + 11.0 * (i / 5));
        for (uint32_t g = 0; g < 4; g++)
            for (uint32_t k = 0; k < nAtk; k++)
                AnimationInterface::SetConstantPosition(atk.Get(g * nAtk + k), 10 + 20.0 * g, 8 + 9.0 * k);

        ai = std::make_unique<AnimationInterface>(anim);
        ai->SetMaxPktsPerTraceFile(50000000);
        auto style = [&](Ptr<Node> n, Rgb c, double size, const std::string& d) {
            ai->UpdateNodeColor(n, c.r, c.g, c.b);
            ai->UpdateNodeSize(n, size, size);
            ai->UpdateNodeDescription(n, d);
        };
        style(gw.Get(0), {30, 100, 255}, 6, "GATEWAY + IDS");
        style(server.Get(0), {150, 60, 200}, 6, "SERVER");
        for (uint32_t i = 0; i < nBenign; i++)
            style(dev.Get(i), {40, 200, 80}, 3.5, "IoT " + ipOf(iIot, i + 1).substr(5)); // "1.1.5"
        for (uint32_t g = 0; g < 4; g++)
            for (uint32_t k = 0; k < nAtk; k++)
            {
                Ptr<Node> n = atk.Get(g * nAtk + k);
                Rgb c = atkColor[g];
                style(n, {150, 150, 150}, 3.5, std::string(atkShort[g]) + " " + ipOf(iAtk, g * nAtk + k + 1).substr(5));
                Simulator::Schedule(Seconds(slot * (g + 1)), [&ai, n, c]() { ai->UpdateNodeColor(n, c.r, c.g, c.b); });
                Simulator::Schedule(Seconds(slot * (g + 2)), [&ai, n]() { ai->UpdateNodeColor(n, 150, 150, 150); });
            }
    }

    Simulator::Stop(Seconds(duration));
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
