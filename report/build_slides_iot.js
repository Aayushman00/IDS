// IoT-oriented deck: threat -> gateway IDS -> dataset -> results -> edge feasibility.
// usage: node build_slides_iot.js   (writes IoT_IDS_IoT_slides.pptx)
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "report_data.json"), "utf8"));
const FIG = (n) => path.join(__dirname, "..", "outputs", "figures", n);
const ANIM = (n) => path.join(__dirname, "..", "outputs", "animations", n);

const NAVY = "0F1B2D", INK = "1F2937", MUTED = "5B6472", TEAL = "0E8C80", RED = "C0392B",
  AMBER = "B8660B", CARD = "F1F5F8", WHITE = "FFFFFF", LINE = "D5DCE3", MINT = "9FE3DA", PALE = "C9D3DD",
  REDBG = "FBE9E7";
const HEAD = "Arial", BODY = "Calibri";

const M = D.models, CL = M["CNN-LSTM"], MAJ = M["Majority (always malicious)"], RF = M.RandomForest,
  CNN = M.CNN, RI = D.run_info, P = D.paper;
const pct = (x, d = 2) => (100 * x).toFixed(d) + "%";
const num = (x) => Number(x).toLocaleString("en-US");
const e8 = D.edge.find((r) => r.variant.includes("int8"));
const e32 = D.edge.find((r) => r.variant === "TFLite float32");
const k32 = D.edge.find((r) => r.variant.startsWith("Keras"));

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "Securing IoT Networks with a Lightweight Gateway IDS";
pres.author = "Aayushman, Ashutosh Kumar, Sahil Mengji";

function title(s, text, sub) {
  s.addText(text, { x: 0.5, y: 0.3, w: 9, h: 0.6, fontFace: HEAD, fontSize: 26, bold: true, color: INK,
    margin: 0, isTextBox: true });
  if (sub) s.addText(sub, { x: 0.5, y: 0.88, w: 9, h: 0.35, fontFace: BODY, fontSize: 13, color: MUTED,
    margin: 0, isTextBox: true });
}
function card(s, x, y, w, h, fill = CARD) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08 });
}
function badge(s, x, y, label, color = TEAL) {
  s.addShape(pres.shapes.OVAL, { x, y, w: 0.42, h: 0.42, fill: { color }, line: { color } });
  s.addText(label, { x, y, w: 0.42, h: 0.42, fontFace: HEAD, fontSize: 13, bold: true, color: WHITE,
    align: "center", valign: "middle", margin: 0, isTextBox: true });
}
function stat(s, x, y, w, value, label, color = TEAL, size = 30) {
  s.addText(value, { x, y, w, h: 0.6, fontFace: HEAD, fontSize: size, bold: true, color, margin: 0, isTextBox: true });
  s.addText(label, { x, y: y + 0.6, w, h: 0.45, fontFace: BODY, fontSize: 11.5, color: MUTED, margin: 0,
    valign: "top", isTextBox: true });
}
function img(s, file, x, y, w, ratio) { s.addImage({ path: file, x, y, w, h: w / ratio }); }
function source(s, text) {
  s.addText(text, { x: 0.5, y: 5.22, w: 9, h: 0.25, fontFace: BODY, fontSize: 9, color: MUTED, margin: 0, isTextBox: true });
}
function box(s, x, y, w, h, text, fill, color, size = 12, bold = true) {
  card(s, x, y, w, h, fill);
  s.addText(text, { x, y, w, h, fontFace: BODY, fontSize: size, bold, color, align: "center", valign: "middle",
    margin: 3, isTextBox: true });
}
function arrow(s, x1, y1, x2, y2, color = MUTED) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, flipV: y2 < y1, line: { color, width: 2, endArrowType: "triangle" } });
}

// 1 Title ---------------------------------------------------------------------
let s = pres.addSlide();
s.background = { color: NAVY };
s.addText("Securing IoT Networks", { x: 0.6, y: 1.0, w: 8.8, h: 0.8, fontFace: HEAD, fontSize: 38, bold: true,
  color: WHITE, margin: 0, isTextBox: true });
s.addText("A lightweight, flow-based intrusion detector for the IoT gateway", { x: 0.6, y: 1.85, w: 8.8, h: 0.5,
  fontFace: BODY, fontSize: 20, color: MINT, margin: 0, isTextBox: true });
s.addText("Built by reproducing Gueriani, Kheddar & Mazari, \"Enhancing IoT Security with CNN and LSTM-Based " +
  "Intrusion Detection Systems\", IEEE PAIS 2024, on the CICIoT2023 IoT testbed dataset.", { x: 0.6, y: 2.6, w: 8.4,
  h: 0.7, fontFace: BODY, fontSize: 12, color: PALE, margin: 0, isTextBox: true });
s.addText([
  { text: "Aayushman (231CS105)", options: { breakLine: true } },
  { text: "Ashutosh Kumar (231CS113)", options: { breakLine: true } },
  { text: "Sahil Mengji (231CS151)" }], { x: 0.6, y: 3.6, w: 5, h: 1.1, fontFace: BODY, fontSize: 15, color: WHITE,
  margin: 0, paraSpaceAfter: 4, isTextBox: true });
s.addText([
  { text: "Internet of Things", options: { breakLine: true } },
  { text: "Instructor: Mr. Sridhar Sanshi" }], { x: 5.8, y: 3.6, w: 3.6, h: 1.1, fontFace: BODY, fontSize: 13,
  color: PALE, align: "right", margin: 0, paraSpaceAfter: 4, isTextBox: true });
s.addNotes("Our topic is IoT security. IoT devices cannot defend themselves, so we look at a small detector that " +
  "sits at the gateway and watches network traffic. The detector comes from a published paper, which we reproduced.");

// 2 Why IoT is hard to secure ---------------------------------------------------
s = pres.addSlide();
title(s, "Why IoT devices are easy targets", "They cannot protect themselves, so the network has to");
const why = [
  ["Tiny hardware", "Little CPU and RAM: no room for antivirus or heavy security software"],
  ["Weak defaults", "Default or hard-coded passwords, rarely patched firmware"],
  ["Always on, always connected", "Cameras, plugs and sensors stay online 24/7 and are reachable from the internet"],
  ["Many device types", "Different vendors and protocols make one-size-fits-all security hard"]];
why.forEach(([h, d], i) => {
  const x = 0.5 + (i % 2) * 4.6, y = 1.5 + Math.floor(i / 2) * 1.6;
  card(s, x, y, 4.4, 1.4);
  badge(s, x + 0.2, y + 0.2, String(i + 1));
  s.addText(h, { x: x + 0.75, y: y + 0.18, w: 3.5, h: 0.45, fontFace: HEAD, fontSize: 15, bold: true, color: INK,
    valign: "middle", margin: 0, isTextBox: true });
  s.addText(d, { x: x + 0.2, y: y + 0.72, w: 4.0, h: 0.6, fontFace: BODY, fontSize: 12.5, color: MUTED, margin: 0,
    valign: "top", isTextBox: true });
});
s.addText("Result: one hacked camera can join a botnet, flood other sites, or scan the whole home network.", { x: 0.5,
  y: 4.75, w: 9, h: 0.4, fontFace: BODY, fontSize: 13, bold: true, color: RED, margin: 0, isTextBox: true });
s.addNotes("IoT devices are constrained and often ship with weak credentials. They cannot run an antivirus. " +
  "A compromised device is usually silent: the owner sees nothing, but the device attacks others.");

// 3 Attack families -----------------------------------------------------------------
s = pres.addSlide();
title(s, "What attacks hit IoT networks", "The 7 attack families in CICIoT2023 (33 attack types in total)");
const fam = [
  ["DDoS", "Many hacked devices flood one target (ICMP, UDP, SYN floods)", RED],
  ["DoS", "A single source floods a target until it stops responding", RED],
  ["Mirai", "Botnet malware traffic: greeth, UDP and similar floods", RED],
  ["Recon", "Port scan, OS scan, vulnerability scan: mapping the network", AMBER],
  ["Spoofing", "ARP and DNS spoofing: man-in-the-middle on the LAN", AMBER],
  ["Web", "SQL injection, XSS, command injection, backdoor, upload attacks", AMBER],
  ["Brute force", "Dictionary attack on device or service passwords", AMBER],
  ["Benign", "Normal device traffic: what the IDS must NOT flag", TEAL]];
fam.forEach(([h, d, c], i) => {
  const x = 0.5 + (i % 4) * 2.28, y = 1.45 + Math.floor(i / 4) * 1.75;
  card(s, x, y, 2.12, 1.55);
  s.addShape(pres.shapes.OVAL, { x: x + 0.15, y: y + 0.15, w: 0.2, h: 0.2, fill: { color: c }, line: { color: c } });
  s.addText(h, { x: x + 0.45, y: y + 0.1, w: 1.6, h: 0.3, fontFace: HEAD, fontSize: 14, bold: true, color: INK,
    valign: "middle", margin: 0, isTextBox: true });
  s.addText(d, { x: x + 0.15, y: y + 0.52, w: 1.85, h: 0.95, fontFace: BODY, fontSize: 11.5, color: MUTED, margin: 0,
    valign: "top", isTextBox: true });
});
s.addText("Red = high-volume floods.  Amber = quiet attacks that look like normal traffic.", { x: 0.5, y: 4.95, w: 9,
  h: 0.3, fontFace: BODY, fontSize: 11, color: MUTED, margin: 0, isTextBox: true });
s.addNotes("Floods are loud and easy to see. Reconnaissance, spoofing and web attacks are quiet and are the hard cases, " +
  "as our error analysis will show later.");

// 4 Where the IDS sits (architecture) ---------------------------------------------------
s = pres.addSlide();
title(s, "Our idea: an IDS at the IoT gateway", "Every device's traffic passes through the gateway, so one detector protects them all");
const devs = ["IP camera", "Smart plug", "Sensor"];
devs.forEach((d, i) => {
  const y = 1.55 + i * 0.95;
  box(s, 0.5, y, 1.7, 0.7, d, CARD, INK, 12);
  arrow(s, 2.2, y + 0.35, 3.3, 2.5 + (i - 1) * 0.15);
});
box(s, 3.3, 1.55, 3.0, 2.65, "", NAVY, WHITE);
s.addText("IoT gateway", { x: 3.3, y: 1.65, w: 3.0, h: 0.4, fontFace: HEAD, fontSize: 15, bold: true, color: WHITE,
  align: "center", margin: 0, isTextBox: true });
box(s, 3.55, 2.15, 2.5, 0.5, "1  Group packets into flows", "1E2F47", WHITE, 11, false);
box(s, 3.55, 2.75, 2.5, 0.5, "2  Compute flow features", "1E2F47", WHITE, 11, false);
box(s, 3.55, 3.35, 2.5, 0.65, "3  Lightweight ML model: benign or attack?", TEAL, WHITE, 11);
arrow(s, 6.3, 2.5, 7.2, 2.5);
box(s, 7.2, 1.95, 2.3, 1.1, "Internet / cloud", CARD, INK, 13);
arrow(s, 4.8, 4.2, 4.8, 4.65, RED);
box(s, 3.3, 4.65, 3.0, 0.5, "ALERT / block the device", REDBG, RED, 12);
s.addNotes("The detector does not run on the tiny devices. It runs on the gateway, where all traffic is visible. " +
  "This slide is the proposed deployment. In this project we evaluated the detector on a recorded dataset; " +
  "running it live on a gateway is future work.");

// 5 Dataset ------------------------------------------------------------------------------
s = pres.addSlide();
title(s, "Data: CICIoT2023, captured from a real IoT lab", "Canadian Institute for Cybersecurity (UNB): attacks launched against real devices");
const ds = [["105", "real IoT devices in the testbed"], ["33", "attack types, 7 families"],
  [(RI.population_rows / 1e6).toFixed(1) + "M", "network flows in the files we used"],
  [pct(MAJ.accuracy, 1), "of the flows are attacks"]];
ds.forEach(([v, l], i) => stat(s, 0.5 + i * 2.3, 1.35, 2.1, v, l, i === 3 ? RED : TEAL));
img(s, FIG("01_class_distribution.png"), 0.5, 2.55, 9, 3.556);
source(s, "Fig 1 - benign vs malicious counts: full data, our sample, and the training split before/after class weighting.");
s.addNotes("The data is recorded traffic from a real IoT lab with 105 devices. We use this recorded traffic; we did not " +
  "capture our own. Only about 2.7% of flows are benign, which will matter when we read the results.");

// 6 From packets to features ----------------------------------------------------------
s = pres.addSlide();
title(s, "From raw traffic to a decision", "The detector sees flows, not individual packets");
const chain = [["Packets", "raw network\ntraffic"], ["Flows", "one connection =\none flow"], ["46 features", "per flow"],
  ["Model", "benign or\nmalicious"], ["Alert", "if malicious"]];
chain.forEach(([h, d], i) => {
  const x = 0.5 + i * 1.85;
  card(s, x, 1.45, 1.6, 1.15, i === 3 ? NAVY : i === 4 ? REDBG : CARD);
  s.addText(h, { x, y: 1.52, w: 1.6, h: 0.4, fontFace: HEAD, fontSize: 14, bold: true,
    color: i === 3 ? WHITE : i === 4 ? RED : TEAL, align: "center", margin: 0, isTextBox: true });
  s.addText(d, { x, y: 1.95, w: 1.6, h: 0.6, fontFace: BODY, fontSize: 11, color: i === 3 ? WHITE : INK,
    align: "center", valign: "top", margin: 0, isTextBox: true });
  if (i < chain.length - 1) s.addText(">", { x: x + 1.6, y: 1.8, w: 0.25, h: 0.4, fontFace: HEAD, fontSize: 18,
    bold: true, color: MUTED, align: "center", margin: 0, isTextBox: true });
});
s.addText("Example flow features", { x: 0.5, y: 2.95, w: 4.4, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true,
  color: INK, margin: 0, isTextBox: true });
s.addText([
  { text: "Flow duration, packet rate", options: { bullet: true, breakLine: true } },
  { text: "TCP flag counts: SYN, ACK, FIN, RST", options: { bullet: true, breakLine: true } },
  { text: "Protocol: TCP, UDP, ICMP, DNS, HTTP, ARP", options: { bullet: true, breakLine: true } },
  { text: "Packet size and inter-arrival time statistics", options: { bullet: true } }],
  { x: 0.5, y: 3.35, w: 4.4, h: 1.7, fontFace: BODY, fontSize: 12.5, color: INK, paraSpaceAfter: 5, margin: 0, valign: "top", isTextBox: true });
card(s, 5.3, 2.95, 4.2, 2.05);
s.addText("Why this works for IoT", { x: 5.5, y: 3.05, w: 3.8, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: TEAL,
  margin: 0, isTextBox: true });
s.addText("A flood has a huge packet rate and many SYN flags; a scan sends many tiny probes; IoT devices normally " +
  "talk in simple, repetitive patterns. Attacks break those patterns, and the flow features capture that.",
  { x: 5.5, y: 3.4, w: 3.8, h: 1.5, fontFace: BODY, fontSize: 12, color: INK, margin: 0, valign: "top", isTextBox: true });
s.addNotes("CICIoT2023 already provides flow-level features, which is what we train on. In a live system, a flow " +
  "extractor such as CICFlowMeter would have to produce the same features on the gateway.");

// 7 The detector (kept simple) ------------------------------------------------------------
s = pres.addSlide();
title(s, "The detector: a small neural network", "Design goal: accurate enough, and small enough for an edge device");
const trunk = [`${RI.n_features_used} flow\nfeatures`, "CNN layers", "Dense layers", "LSTM branch", "Benign /\nMalicious"];
trunk.forEach((t, i) => {
  const x = 0.5 + i * 1.85;
  box(s, x, 1.6, 1.6, 0.95, t, i === 4 ? NAVY : CARD, i === 4 ? WHITE : INK, 12);
  if (i < trunk.length - 1) s.addText(">", { x: x + 1.6, y: 1.85, w: 0.25, h: 0.4, fontFace: HEAD, fontSize: 18,
    bold: true, color: MUTED, align: "center", margin: 0, isTextBox: true });
});
stat(s, 0.5, 3.0, 2.8, num(CL.total_params), "parameters in total", TEAL, 30);
stat(s, 3.5, 3.0, 2.8, CL.model_size_mb.toFixed(1) + " MB", "model file (Keras)", TEAL, 30);
stat(s, 6.5, 3.0, 3.0, "25 epochs", `${(RI.train_time_s / 60).toFixed(0)} min on a laptop GPU`, TEAL, 30);
s.addText("CNN-LSTM exactly as in the paper's Fig. 2. Unstated details (batch size, learning rate, pooling) are " +
  "documented assumptions. The deep-learning internals are in our report; here we care about what it means for IoT.",
  { x: 0.5, y: 4.4, w: 9, h: 0.7, fontFace: BODY, fontSize: 12, color: MUTED, margin: 0, isTextBox: true });
s.addNotes("We keep the model description short on purpose. It is a convolutional layer stack followed by an LSTM branch, " +
  "about 80 thousand parameters. The important number for IoT is the size: around one megabyte.");

// 8 Results, IoT framing ----------------------------------------------------------------------
s = pres.addSlide();
title(s, "Result: it catches almost every attack", `Held-out test set: ${num(RI.n_test)} flows the model never saw`);
stat(s, 0.5, 1.4, 2.3, pct(CL.accuracy), `accuracy (paper: ${P.accuracy}%)`, RED, 32);
stat(s, 2.9, 1.4, 2.2, pct(CL.recall, 1), "of attacks detected", TEAL, 32);
stat(s, 0.5, 2.7, 2.3, num(CL.fp), `false alarms on benign traffic (${pct(CL.fpr)})`, TEAL, 32);
stat(s, 2.9, 2.7, 2.2, num(CL.fn), `attacks missed (${pct(CL.fnr)})`, AMBER, 32);
s.addText(`False alarms matter for IoT: a wrongly blocked device (a door lock, a camera) is a real outage. ` +
  `Here only ${CL.fp} benign flows were flagged.`, { x: 0.5, y: 3.85, w: 4.6, h: 0.65, fontFace: BODY, fontSize: 12,
  color: INK, margin: 0, valign: "top", isTextBox: true });
img(s, FIG("10_confusion_matrix_normalized.png"), 5.7, 1.35, 3.8, 1.167);
s.addText(`Caution: ${pct(MAJ.accuracy, 1)} of flows are attacks, so even "always say attack" scores ${pct(MAJ.accuracy, 1)} ` +
  `accuracy. That is why we also report false alarms and MCC (${CL.mcc.toFixed(2)}).`, { x: 0.5, y: 4.55, w: 4.8, h: 0.65,
  fontFace: BODY, fontSize: 11, color: MUTED, margin: 0, valign: "top", isTextBox: true });
s.addNotes(`Accuracy is ${pct(CL.accuracy)}, ${(100 * CL.accuracy - P.accuracy).toFixed(2)} points above the paper's ${P.accuracy}%. ` +
  "But accuracy alone is misleading on this imbalanced data, so we focus on attacks caught and false alarms.");

// 9 Which attacks are missed ---------------------------------------------------------------------
s = pres.addSlide();
title(s, "Which IoT attacks does it miss?", "Share of test flows misclassified, per original traffic type");
img(s, FIG("20_error_by_attack_type.png"), 0.5, 1.3, 5.4, 1.429);
const eb = Object.fromEntries(D.error_by_type.map((r) => [r.sub, r]));
const er = (k) => eb[k] ? pct(+eb[k].error_rate, 0) : "n/a";
s.addText([
  { text: "Floods: easy", options: { bold: true, color: TEAL, breakLine: true } },
  { text: `DDoS and DoS are almost never missed (DDoS-SYN ${er("DDoS-SYN_Flood")}).`, options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Quiet attacks: hard", options: { bold: true, color: RED, breakLine: true } },
  { text: `Recon-OSScan ${er("Recon-OSScan")}, DNS spoofing ${er("DNS_Spoofing")}, ARP spoofing ${er("MITM-ArpSpoofing")} missed.`, options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Web attacks: mostly missed", options: { bold: true, color: RED, breakLine: true } },
  { text: "SQL injection, XSS, backdoor: few samples and they look like normal traffic at flow level." }],
  { x: 6.2, y: 1.35, w: 3.3, h: 3.8, fontFace: BODY, fontSize: 12.5, color: INK, margin: 0, valign: "top", isTextBox: true });
s.addText("Implication: a flow-level IDS is one layer. Pair it with device hardening and application-level checks.",
  { x: 6.2, y: 4.15, w: 3.3, h: 0.95, fontFace: BODY, fontSize: 12, bold: true, color: INK, margin: 0, valign: "top", isTextBox: true });
s.addNotes("The detector is excellent on loud attacks and weak on quiet ones. That is an honest limit of looking at flow " +
  "statistics only, and it is why we describe the IDS as one security layer, not the whole solution.");

// 10 Edge feasibility ----------------------------------------------------------------------------
s = pres.addSlide();
title(s, "Small and fast enough for a gateway", "TensorFlow Lite export, measured on a laptop CPU, one flow at a time");
s.addChart(pres.charts.BAR, [{ name: "Model size (MB)", labels: ["Keras", "TFLite float32", "TFLite int8"],
  values: [+k32.size_mb.toFixed(3), +e32.size_mb.toFixed(3), +e8.size_mb.toFixed(3)] }],
  { x: 0.5, y: 1.35, w: 4.6, h: 3.4, barDir: "col", chartColors: [TEAL], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.00", dataLabelFontSize: 12, showLegend: false, catAxisLabelFontSize: 12,
    valAxisLabelFontSize: 10, valAxisMinVal: 0, valGridLine: { color: "E5E7EB", size: 0.5 },
    catGridLine: { style: "none" }, showTitle: true, title: "Model size (MB)", titleFontSize: 12 });
stat(s, 5.5, 1.4, 4.0, `${e8.size_mb.toFixed(2)} MB`, `int8 model: ${(k32.size_mb / e8.size_mb).toFixed(0)}x smaller than Keras`, TEAL, 30);
stat(s, 5.5, 2.55, 4.0, `${e8.ms_per_sample.toFixed(3)} ms`, "to classify one flow on a CPU", TEAL, 30);
stat(s, 5.5, 3.7, 4.0, e8.accuracy_drop_pp <= 0.005 ? "No loss" : `${e8.accuracy_drop_pp.toFixed(2)} pp`,
  "of accuracy after int8 quantization", TEAL, 30);
s.addText("Not yet tested on real IoT hardware (e.g. Raspberry Pi): that is our next step.", { x: 0.5, y: 4.85, w: 9, h: 0.3,
  fontFace: BODY, fontSize: 11.5, italic: true, color: AMBER, margin: 0, isTextBox: true });
s.addNotes(`Quantized to int8 the model is ${e8.size_mb.toFixed(2)} MB and takes ${e8.ms_per_sample.toFixed(3)} ms per flow ` +
  "on a CPU, with no accuracy loss. This suggests it would fit a gateway like a Raspberry Pi, but we have only measured " +
  "on a laptop CPU, so we say 'suggests', not 'proves'.");

// 11 Is deep learning needed? -----------------------------------------------------------------------
s = pres.addSlide();
title(s, "Do we even need the CNN-LSTM?", "An honest finding for resource-limited IoT: simpler models do as well");
const hdr = ["Model", "Accuracy", "False alarms", "Missed attacks", "MCC", "Size (MB)"].map((t) =>
  ({ text: t, options: { bold: true, fill: { color: NAVY }, color: WHITE } }));
const mrow = (n, m, hl) => [n, pct(m.accuracy), num(m.fp), num(m.fn), m.mcc.toFixed(3), m.model_size_mb.toFixed(2)]
  .map((t, j) => ({ text: t, options: { bold: j === 0 || hl, fill: { color: hl ? "E6F4F1" : WHITE } } }));
s.addTable([hdr, mrow("CNN-LSTM (paper)", CL, false), mrow("CNN only", CNN, false), mrow("Random Forest", RF, true)],
  { x: 0.5, y: 1.45, w: 9, colW: [2.4, 1.2, 1.4, 1.6, 1.1, 1.3], fontFace: BODY, fontSize: 12.5, color: INK,
    border: { type: "solid", pt: 0.5, color: LINE }, rowH: 0.45 });
s.addText([
  { text: `The LSTM branch adds nothing: CNN-only MCC ${CNN.mcc.toFixed(3)} vs CNN-LSTM ${CL.mcc.toFixed(3)}.`, options: { bullet: true, breakLine: true } },
  { text: `A Random Forest trained on only 300,000 flows is the strongest (accuracy ${pct(RF.accuracy)}, MCC ${RF.mcc.toFixed(2)}).`, options: { bullet: true, breakLine: true } },
  { text: "The paper did not compare against classical models. For constrained IoT gateways, a simple model is attractive: easier to deploy and to audit.", options: { bullet: true } }],
  { x: 0.5, y: 3.6, w: 9, h: 1.5, fontFace: BODY, fontSize: 13, color: INK, paraSpaceAfter: 6, margin: 0, isTextBox: true });
s.addNotes("This is our main critical finding. The paper's architecture works, but its LSTM branch gives no measurable " +
  "benefit, and a plain Random Forest beats it. For IoT, where simplicity matters, that is a useful result.");

// 12 Reproduction vs paper -----------------------------------------------------------------------------
s = pres.addSlide();
title(s, "Reproduction check against the paper", "Paper values from the published abstract");
s.addChart(pres.charts.BAR, [
  { name: "Paper", labels: ["Accuracy", "F1 (weighted)"], values: [P.accuracy, P.f1] },
  { name: "Ours", labels: ["Accuracy", "F1 (weighted)"], values: [+(100 * CL.accuracy).toFixed(2), +(100 * CL.f1_weighted).toFixed(2)] }],
  { x: 0.5, y: 1.4, w: 5.2, h: 3.5, barDir: "col", barGrouping: "clustered", chartColors: ["5B6472", RED],
    valAxisMinVal: 97, valAxisMaxVal: 100, valAxisLabelFormatCode: "0.0", showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.00", dataLabelFontSize: 11, showLegend: true, legendPos: "b", catAxisLabelFontSize: 12,
    valAxisLabelFontSize: 10, valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" },
    showTitle: true, title: "Score (%)", titleFontSize: 12 });
const cmp = [["Accuracy", `${P.accuracy}%`, pct(CL.accuracy)], ["F1", `${P.f1}%`, pct(CL.f1_weighted)],
  ["False pos. rate", `${P.fpr}%`, pct(CL.fpr)]];
const rows = [["Metric", "Paper", "Ours"].map((t) => ({ text: t, options: { bold: true, fill: { color: NAVY }, color: WHITE } }))]
  .concat(cmp.map((r) => r.map((t, j) => ({ text: t, options: { bold: j === 0 } }))));
s.addTable(rows, { x: 6.0, y: 1.5, w: 3.5, colW: [1.4, 1.0, 1.1], fontFace: BODY, fontSize: 12, color: INK,
  border: { type: "solid", pt: 0.5, color: LINE }, rowH: 0.4 });
const nb = D.nobalance;
s.addText(`Our false positive rate is far lower because we weighted the rare benign class. Without class weights it is ` +
  `${nb ? pct(nb.model.fpr) : "n/a"}, close to the paper's ${P.fpr}%. Data subset, split and hyperparameters also differ.`,
  { x: 6.0, y: 3.3, w: 3.5, h: 1.6, fontFace: BODY, fontSize: 12, color: MUTED, margin: 0, valign: "top", isTextBox: true });
s.addNotes("The headline claim reproduces. Differences come from the data subset, the split, removed duplicates and " +
  "unstated hyperparameters; all are listed in our report.");

// 13 Live replay ------------------------------------------------------------------------------------------
s = pres.addSlide();
title(s, "Demo: replaying real IoT flows through the detector", "Real held-out test flows, real model output (traffic_replay.mp4)");
img(s, ANIM("traffic_replay_final_16x9.png"), 0.5, 1.4, 5.6, 1.778);
s.addText([
  { text: "Benign flows stay far below the alert threshold", options: { bullet: true, breakLine: true } },
  { text: "A DDoS burst pushes the score to ~1.0 and fires the alert", options: { bullet: true, breakLine: true } },
  { text: "Quiet recon and spoofing flows are sometimes missed", options: { bullet: true } }],
  { x: 6.4, y: 1.45, w: 3.1, h: 2.4, fontFace: BODY, fontSize: 12.5, color: INK, paraSpaceAfter: 6, margin: 0, valign: "top", isTextBox: true });
s.addText("This is a replay of recorded flows, not live network traffic. The stream is re-sampled to be watchable, so its " +
  "running numbers are not the reported results.", { x: 0.5, y: 4.7, w: 9, h: 0.6, fontFace: BODY, fontSize: 11.5,
  italic: true, color: MUTED, margin: 0, valign: "top", isTextBox: true });
s.addNotes("Play traffic_replay.mp4 here. Stress that this is recorded data replayed, not a live network.");

// 14 Limits + future ----------------------------------------------------------------------------------------
s = pres.addSlide();
title(s, "Limits and next steps toward a real IoT deployment", "What we did, and what is still missing");
card(s, 0.5, 1.45, 4.4, 2.9);
s.addText("Done", { x: 0.7, y: 1.55, w: 4.0, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: TEAL, margin: 0, isTextBox: true });
s.addText([
  { text: "Detector trained on real IoT lab traffic", options: { bullet: true, breakLine: true } },
  { text: "98.7% accuracy, only 8 false alarms in 183,857 flows", options: { bullet: true, breakLine: true } },
  { text: "5-fold CV: stable at 98.1%", options: { bullet: true, breakLine: true } },
  { text: `Exported to ${e8.size_mb.toFixed(2)} MB TFLite model`, options: { bullet: true } }],
  { x: 0.7, y: 2.0, w: 4.0, h: 2.7, fontFace: BODY, fontSize: 13, color: INK, paraSpaceAfter: 6, margin: 0, valign: "top", isTextBox: true });
card(s, 5.1, 1.45, 4.4, 2.9, REDBG);
s.addText("Not done yet", { x: 5.3, y: 1.55, w: 4.0, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: RED, margin: 0, isTextBox: true });
s.addText([
  { text: "Run on real gateway hardware (Raspberry Pi)", options: { bullet: true, breakLine: true } },
  { text: "Live packet capture and flow extraction with the same 46 features", options: { bullet: true, breakLine: true } },
  { text: "Detect unseen attacks and new devices", options: { bullet: true, breakLine: true } },
  { text: "Better coverage of quiet attacks (recon, spoofing, web)", options: { bullet: true } }],
  { x: 5.3, y: 2.0, w: 4.0, h: 2.7, fontFace: BODY, fontSize: 13, color: INK, paraSpaceAfter: 6, margin: 0, valign: "top", isTextBox: true });
s.addNotes("Be upfront: we did not deploy on hardware and we did not capture live traffic. The biggest practical gap is " +
  "live flow-feature extraction that matches the dataset's feature definitions exactly.");

// 15 Conclusion ------------------------------------------------------------------------------------------------
s = pres.addSlide();
s.background = { color: NAVY };
s.addText("Takeaways", { x: 0.6, y: 0.45, w: 8.8, h: 0.7, fontFace: HEAD, fontSize: 34, bold: true, color: WHITE, margin: 0, isTextBox: true });
const concl = [
  ["IoT devices cannot defend themselves", "A gateway IDS protects all of them without loading the devices."],
  [`It works: ${pct(CL.accuracy)} accuracy, ${CL.fp} false alarms`, "Reproduces the paper's 98.42% on real IoT lab traffic."],
  ["Small enough for the edge", `${e8.size_mb.toFixed(2)} MB and ${e8.ms_per_sample.toFixed(3)} ms per flow (laptop CPU, no accuracy loss).`],
  ["Honest limits", "Quiet attacks are missed, a Random Forest does as well, and live hardware deployment is future work."]];
concl.forEach(([h, d], i) => {
  const y = 1.45 + i * 0.95;
  badge(s, 0.6, y + 0.05, String(i + 1), TEAL);
  s.addText(h, { x: 1.2, y, w: 8.2, h: 0.38, fontFace: HEAD, fontSize: 16, bold: true, color: WHITE, margin: 0, isTextBox: true });
  s.addText(d, { x: 1.2, y: y + 0.38, w: 8.2, h: 0.38, fontFace: BODY, fontSize: 13, color: PALE, margin: 0, isTextBox: true });
});
s.addNotes("Close on the IoT message: protect many weak devices from one place, with a model small enough for the edge.");

pres.writeFile({ fileName: path.join(__dirname, "IoT_IDS_IoT_slides.pptx") }).then((f) => console.log("written", f));
