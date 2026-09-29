// Build IoT_IDS_CNN_LSTM_slides.pptx from report_data.json + the pipeline figures.
// usage: node build_slides.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "report_data.json"), "utf8"));
const FIG = (n) => path.join(__dirname, "..", "outputs", "figures", n);
const ANIM = (n) => path.join(__dirname, "..", "outputs", "animations", n);

// ---- palette / type -------------------------------------------------------
const NAVY = "0F1B2D", INK = "1F2937", MUTED = "5B6472", TEAL = "0E8C80", RED = "C0392B",
  AMBER = "B8660B", CARD = "F1F5F8", WHITE = "FFFFFF", LINE = "D5DCE3";
const HEAD = "Arial", BODY = "Calibri";

const M = D.models, CL = M["CNN-LSTM"], MAJ = M["Majority (always malicious)"], RI = D.run_info;
const pct = (x, d = 2) => (100 * x).toFixed(d) + "%";
const num = (x) => Number(x).toLocaleString("en-US");
const P = D.paper;

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "IoT Network Intrusion Detection - CNN-LSTM IDS reproduction";
pres.author = "Aayushman, Ashutosh Kumar, Sahil Mengji";

// ---- helpers ----------------------------------------------------------------
function title(s, text, sub) {
  s.addText(text, { x: 0.5, y: 0.3, w: 9, h: 0.6, fontFace: HEAD, fontSize: 26, bold: true,
    color: INK, margin: 0, isTextBox: true });
  if (sub) s.addText(sub, { x: 0.5, y: 0.88, w: 9, h: 0.35, fontFace: BODY, fontSize: 13,
    color: MUTED, margin: 0, isTextBox: true });
}
function card(s, x, y, w, h, fill = CARD) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill },
    rectRadius: 0.08 });
}
function badge(s, x, y, label, color = TEAL) {
  s.addShape(pres.shapes.OVAL, { x, y, w: 0.42, h: 0.42, fill: { color }, line: { color } });
  s.addText(label, { x, y, w: 0.42, h: 0.42, fontFace: HEAD, fontSize: 13, bold: true, color: WHITE,
    align: "center", valign: "middle", margin: 0, isTextBox: true });
}
function stat(s, x, y, w, value, label, color = TEAL, size = 30) {
  s.addText(value, { x, y, w, h: 0.6, fontFace: HEAD, fontSize: size, bold: true, color, margin: 0,
    isTextBox: true });
  s.addText(label, { x, y: y + 0.6, w, h: 0.45, fontFace: BODY, fontSize: 11.5, color: MUTED,
    margin: 0, valign: "top", isTextBox: true });
}
function img(s, file, x, y, w, ratio) {
  s.addImage({ path: file, x, y, w, h: w / ratio });
}
function source(s, text) {
  s.addText(text, { x: 0.5, y: 5.22, w: 9, h: 0.25, fontFace: BODY, fontSize: 9, color: MUTED,
    margin: 0, isTextBox: true });
}

// ============================================================================
// 1 Title
let s = pres.addSlide();
s.background = { color: NAVY };
s.addText("IoT Network Intrusion Detection", { x: 0.6, y: 1.05, w: 8.8, h: 0.8, fontFace: HEAD,
  fontSize: 36, bold: true, color: WHITE, margin: 0, isTextBox: true });
s.addText("Reproducing a CNN-LSTM intrusion detection system on CICIoT2023", { x: 0.6, y: 1.85,
  w: 8.8, h: 0.5, fontFace: BODY, fontSize: 20, color: "9FE3DA", margin: 0, isTextBox: true });
s.addText("Paper: Gueriani, Kheddar & Mazari, \"Enhancing IoT Security with CNN and LSTM-Based Intrusion " +
  "Detection Systems\", IEEE PAIS 2024 (DOI 10.1109/PAIS62114.2024.10541178)", { x: 0.6, y: 2.6, w: 8.6,
  h: 0.6, fontFace: BODY, fontSize: 12, color: "C9D3DD", margin: 0, isTextBox: true });
s.addText([
  { text: "Aayushman (231CS105)", options: { breakLine: true } },
  { text: "Ashutosh Kumar (231CS113)", options: { breakLine: true } },
  { text: "Sahil Mengji (231CS151)" }], { x: 0.6, y: 3.55, w: 5, h: 1.1, fontFace: BODY, fontSize: 15,
  color: WHITE, margin: 0, paraSpaceAfter: 4, isTextBox: true });
s.addNotes("Our project reproduces a published CNN-LSTM intrusion detection system for IoT networks and " +
  "checks whether its reported 98.42% accuracy holds up on the CICIoT2023 dataset.");

// 2 Goal
s = pres.addSlide();
title(s, "What we set out to do", "Scientific reproduction and validation, not a new method");
const goals = [
  ["1", "Rebuild the architecture", "CNN-LSTM exactly as drawn in the paper's Fig. 2"],
  ["2", "Train on real IoT traffic", "CICIoT2023: 105 devices, benign traffic + 33 attack types"],
  ["3", "Detect benign vs malicious", "Binary classification of network flows"],
  ["4", "Compare with the paper", "Accuracy, precision, recall, F1 vs the reported 98.42%"],
  ["5", "Document every deviation", "What differs, and why the numbers might differ"]];
goals.forEach(([n, h, d], i) => {
  const y = 1.45 + i * 0.72;
  badge(s, 0.5, y + 0.02, n);
  s.addText(h, { x: 1.1, y, w: 4.6, h: 0.3, fontFace: HEAD, fontSize: 14, bold: true, color: INK, margin: 0, isTextBox: true });
  s.addText(d, { x: 1.1, y: y + 0.3, w: 4.6, h: 0.3, fontFace: BODY, fontSize: 12, color: MUTED, margin: 0, isTextBox: true });
});
card(s, 6.2, 1.45, 3.3, 3.4, NAVY);
s.addText(P.accuracy.toFixed(2) + "%", { x: 6.4, y: 1.85, w: 2.9, h: 0.9, fontFace: HEAD, fontSize: 48,
  bold: true, color: "9FE3DA", align: "center", margin: 0, isTextBox: true });
s.addText("accuracy reported by the paper on CICIoT2023 (binary)", { x: 6.45, y: 2.8, w: 2.8, h: 0.7,
  fontFace: BODY, fontSize: 13, color: WHITE, align: "center", margin: 0, isTextBox: true });
s.addText(`Published abstract also reports F1 ${P.f1}%, FPR ${P.fpr}%, loss ${P.loss}`, { x: 6.45, y: 3.65,
  w: 2.8, h: 0.8, fontFace: BODY, fontSize: 11, color: "C9D3DD", align: "center", margin: 0, isTextBox: true });
s.addNotes("The paper claims 98.42% accuracy for binary benign-vs-malicious detection. Our five goals: " +
  "rebuild the model, train it on the real dataset, get a working detector, compare metrics with the " +
  "paper, and document every difference.");

// 3 Dataset
s = pres.addSlide();
title(s, "CICIoT2023: heavily skewed toward attacks", "Canadian Institute for Cybersecurity, UNB - original 46-feature release");
const ds = [["105", "IoT devices in the testbed"], ["33", "attack types in 7 families"],
  [(RI.population_rows / 1e6).toFixed(2) + "M", "flows in the files we used"],
  [pct(MAJ.accuracy, 1), "of test flows are attacks"]];
ds.forEach(([v, l], i) => stat(s, 0.5 + i * 2.3, 1.35, 2.1, v, l, i === 3 ? RED : TEAL));
img(s, FIG("01_class_distribution.png"), 0.5, 2.55, 9, 3.556);
source(s, "Fig 1 - benign vs malicious counts in the full data, our sample, and the train split before/after class weighting.");
s.addNotes(`Only about ${pct(1 - MAJ.accuracy, 1)} of flows are benign. That matters: a model that always says ` +
  `'malicious' already gets ${pct(MAJ.accuracy)} accuracy, so accuracy alone cannot tell us much.`);

// 4 Pipeline
s = pres.addSlide();
title(s, "A leakage-safe, resumable pipeline", "Each stage saves its outputs and can be re-run on its own");
const stages = [["prepare", "sample, clean,\nsplit, scale"], ["train", "CNN-LSTM\n25 epochs"],
  ["baselines", "CNN, LSTM,\nMLP, RF"], ["cv", "5-fold\nstability"], ["compare", "vs paper"],
  ["edge", "TFLite\nbenchmark"]];
stages.forEach(([h, d], i) => {
  const x = 0.5 + i * 1.53;
  card(s, x, 1.45, 1.35, 1.25, i === 1 ? NAVY : CARD);
  s.addText(h, { x, y: 1.55, w: 1.35, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true,
    color: i === 1 ? WHITE : TEAL, align: "center", margin: 0, isTextBox: true });
  s.addText(d, { x, y: 1.9, w: 1.35, h: 0.7, fontFace: BODY, fontSize: 11, color: i === 1 ? WHITE : INK,
    align: "center", valign: "top", margin: 0, isTextBox: true });
  if (i < stages.length - 1) s.addText(">", { x: x + 1.35, y: 1.85, w: 0.18, h: 0.4, fontFace: HEAD,
    fontSize: 16, bold: true, color: MUTED, align: "center", margin: 0, isTextBox: true });
});
s.addText([
  { text: `${num(RI.rows_sampled)} flows sampled (stratified) from ${num(RI.population_rows)}`, options: { bullet: true, breakLine: true } },
  { text: `${num(RI.duplicates_removed)} exact duplicates removed before splitting`, options: { bullet: true, breakLine: true } },
  { text: `Split 70/15/15: ${num(RI.n_train)} train / ${num(RI.n_val)} val / ${num(RI.n_test)} test`, options: { bullet: true, breakLine: true } },
  { text: `${RI.n_features_used} of ${RI.n_features_raw} features (constant columns dropped)`, options: { bullet: true } }],
  { x: 0.5, y: 3.05, w: 4.6, h: 1.9, fontFace: BODY, fontSize: 13, color: INK, paraSpaceAfter: 6, margin: 0, isTextBox: true });
card(s, 5.4, 3.05, 4.1, 1.85);
s.addText("No data leakage", { x: 5.6, y: 3.15, w: 3.7, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: TEAL, margin: 0, isTextBox: true });
s.addText("Scaler, imputer and class weights are fitted on the training split only. Validation and test keep " +
  "their natural class mix. Automated tests check all of this.", { x: 5.6, y: 3.5, w: 3.7, h: 1.3,
  fontFace: BODY, fontSize: 12, color: INK, margin: 0, valign: "top", isTextBox: true });
s.addNotes("Duplicates were dropped before the split so the same flow can never appear in both training and " +
  "test data. Every statistic the model learns from is computed on training data only.");

// 5 Architecture
s = pres.addSlide();
title(s, "The CNN-LSTM model (paper Fig. 2)", `${num(CL.total_params)} parameters; two heads merged into one decision`);
const trunk = [`Input ${RI.n_features_used}x1`, "Conv1D 64, k3", "BatchNorm + AvgPool", "Conv1D 64, k3",
  "AvgPool + Flatten", "Dense 32", "Dense 16"];
trunk.forEach((t, i) => {
  const x = 0.5 + i * 1.3;
  card(s, x, 1.5, 1.18, 0.65, i === 0 ? LINE : CARD);
  s.addText(t, { x, y: 1.5, w: 1.18, h: 0.65, fontFace: BODY, fontSize: 11, bold: true, color: INK,
    align: "center", valign: "middle", margin: 2, isTextBox: true });
});
s.addText("CNN part: finds local patterns across neighbouring flow features", { x: 0.5, y: 2.2, w: 9, h: 0.3,
  fontFace: BODY, fontSize: 11, italic: true, color: MUTED, margin: 0, isTextBox: true });
card(s, 0.5, 2.7, 3.3, 0.7);
s.addText("Head A: Dense 2 (softmax)", { x: 0.5, y: 2.7, w: 3.3, h: 0.7, fontFace: BODY, fontSize: 12, bold: true,
  color: INK, align: "center", valign: "middle", margin: 0, isTextBox: true });
card(s, 0.5, 3.6, 6.1, 0.7, "FBE9E7");
s.addText("Head B: Reshape 16x1 > LSTM 64 > LSTM 64 > Dense 2 (sigmoid)", { x: 0.5, y: 3.6, w: 6.1, h: 0.7,
  fontFace: BODY, fontSize: 12, bold: true, color: RED, align: "center", valign: "middle", margin: 0, isTextBox: true });
card(s, 7.0, 2.95, 2.5, 1.1, NAVY);
s.addText("Concatenate > Dense 2 (softmax)\nbenign / malicious", { x: 7.0, y: 2.95, w: 2.5, h: 1.1, fontFace: BODY,
  fontSize: 12, bold: true, color: WHITE, align: "center", valign: "middle", margin: 4, isTextBox: true });
s.addText("Assumed (not stated in the paper): pooling size 2, ReLU, batch 256, learning rate 1e-3, " +
  "early stopping + LR reduction. Paper: Adam, 25 epochs.", { x: 0.5, y: 4.55, w: 9, h: 0.5, fontFace: BODY,
  fontSize: 11, color: MUTED, margin: 0, isTextBox: true });
s.addNotes("We read the layer shapes from the paper's architecture figure. The LSTM does not see time here: " +
  "it reads the 16 numbers produced by the CNN as a short sequence. Hyperparameters the paper does not give " +
  "are marked as assumptions in config.py.");

// 6 Training
s = pres.addSlide();
title(s, "Training on the RTX 4060", `${RI.epochs_run} epochs, best epoch ${RI.best_epoch}, ` +
  `${(RI.train_time_s / 60).toFixed(1)} min total (about 42 s per epoch)`);
img(s, FIG("slides/06_training_accuracy.png"), 0.5, 1.45, 4.4, 1.778);
img(s, FIG("slides/07_training_loss.png"), 5.1, 1.45, 4.4, 1.778);
s.addText("Validation curves track training closely: no sign of overfitting. The learning rate was halved " +
  "automatically when validation loss stalled.", { x: 0.5, y: 4.15, w: 9, h: 0.6, fontFace: BODY, fontSize: 13,
  color: INK, margin: 0, isTextBox: true });
s.addNotes("Training ran in WSL2 on the laptop GPU. The run is resumable: after an earlier crash we added " +
  "per-epoch checkpoints and memory fixes, and peak RAM stayed below 3 GB.");

// 7 Headline
s = pres.addSlide();
title(s, "Result: 98.7% accuracy, matching the paper", `Held-out test set, ${num(RI.n_test)} flows`);
stat(s, 0.5, 1.4, 2.3, pct(CL.accuracy), `accuracy (paper ${P.accuracy}%)`, RED, 32);
stat(s, 2.9, 1.4, 2.0, CL.mcc.toFixed(3), "MCC (0 = no skill)", TEAL, 32);
stat(s, 0.5, 2.75, 2.3, pct(CL.fpr), `false positive rate (paper ${P.fpr}%)`, TEAL, 32);
stat(s, 2.9, 2.75, 2.0, pct(CL.fnr), "attacks missed (FNR)", AMBER, 32);
s.addText(`Errors: ${num(CL.fn)} attacks missed vs only ${num(CL.fp)} false alarms.`, { x: 0.5, y: 4.15, w: 4.5,
  h: 0.6, fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0, isTextBox: true });
img(s, FIG("10_confusion_matrix_normalized.png"), 5.3, 1.3, 4.2, 1.167);
s.addNotes(`Our accuracy is ${(100 * CL.accuracy - P.accuracy).toFixed(2)} points above the paper. ` +
  `Our false positive rate is far lower than the paper's ${P.fpr}%, because we weighted the rare benign class; ` +
  "the price is that most of our errors are missed attacks.");

// 8 Paper comparison (native chart)
s = pres.addSlide();
title(s, "Our reproduction vs the paper", "Paper values from the published abstract; precision/recall only in the full text");
s.addChart(pres.charts.BAR, [
  { name: "Paper", labels: ["Accuracy", "F1 (weighted)"], values: [P.accuracy, P.f1] },
  { name: "Ours", labels: ["Accuracy", "F1 (weighted)"], values: [+(100 * CL.accuracy).toFixed(2), +(100 * CL.f1_weighted).toFixed(2)] }],
  { x: 0.5, y: 1.4, w: 5.2, h: 3.6, barDir: "col", barGrouping: "clustered", chartColors: ["5B6472", RED],
    valAxisMinVal: 97, valAxisMaxVal: 100, valAxisLabelFormatCode: "0.0", showValue: true,
    dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 11, showLegend: true,
    legendPos: "b", catAxisLabelFontSize: 12, valAxisLabelFontSize: 10, valGridLine: { color: "E5E7EB", size: 0.5 },
    catGridLine: { style: "none" }, showTitle: true, title: "Score (%)", titleFontSize: 12 });
const cmp = [["Accuracy", `${P.accuracy}%`, pct(CL.accuracy)], ["F1", `${P.f1}%`, pct(CL.f1_weighted)],
  ["FPR", `${P.fpr}%`, pct(CL.fpr)], ["Loss", `${P.loss}`, CL.log_loss.toFixed(4)]];
const rows = [[{ text: "Metric", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
  { text: "Paper", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
  { text: "Ours", options: { bold: true, fill: { color: NAVY }, color: WHITE } }]]
  .concat(cmp.map((r) => r.map((t, j) => ({ text: t, options: { bold: j === 0 } }))));
s.addTable(rows, { x: 6.0, y: 1.5, w: 3.5, colW: [1.2, 1.1, 1.2], fontFace: BODY, fontSize: 12, color: INK,
  border: { type: "solid", pt: 0.5, color: LINE }, rowH: 0.38 });
s.addText(D.nobalance ? `FPR gap: the paper flags about 1 in 11 benign flows. Our class-weighted model almost never does; ` +
  `retrained without class weights our FPR is ${pct(D.nobalance.model.fpr)}, close to the paper's.` :
  "FPR gap: the paper flags about 1 in 11 benign flows; our class weighting almost never does.",
  { x: 6.0, y: 3.55, w: 3.5, h: 1.3, fontFace: BODY, fontSize: 12, color: MUTED, margin: 0, valign: "top", isTextBox: true });
s.addNotes("Accuracy and F1 land within a third of a point of the paper. Precision and recall are only in the " +
  "paper's full-text table, which we could not access, so we do not compare them.");

// 9 Accuracy is misleading + baselines
s = pres.addSlide();
title(s, "Accuracy hides the real story", "MCC is robust to class imbalance - and the ranking changes");
const order = ["RandomForest", "CNN", "MLP", "CNN-LSTM", "LSTM", "Majority (always malicious)"];
const lab = { "RandomForest": "Random Forest", "Majority (always malicious)": "Always malicious" };
s.addChart(pres.charts.BAR, [{ name: "MCC", labels: order.map((k) => lab[k] || k),
  values: order.map((k) => +M[k].mcc.toFixed(3)) }],
  { x: 0.5, y: 1.35, w: 5.4, h: 3.7, barDir: "bar", chartColors: [TEAL], valAxisMinVal: 0, valAxisMaxVal: 1,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 11,
    showLegend: false, catAxisLabelFontSize: 12, valAxisLabelFontSize: 10, catAxisOrientation: "maxMin",
    valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" }, showTitle: true,
    title: "Matthews correlation coefficient (test set)", titleFontSize: 12 });
const rf = M.RandomForest, cnn = M.CNN;
const pts = [[pct(MAJ.accuracy), "accuracy for 'always malicious' - with MCC 0"],
  [`${CL.mcc.toFixed(3)} vs ${cnn.mcc.toFixed(3)}`, "MCC of CNN-LSTM vs CNN-only: the LSTM branch adds nothing"],
  [pct(rf.accuracy), `Random Forest accuracy, MCC ${rf.mcc.toFixed(3)} - best model`]];
pts.forEach(([v, l], i) => stat(s, 6.3, 1.35 + i * 1.25, 3.2, v, l, i === 2 ? RED : TEAL, 26));
s.addNotes("Always predicting malicious scores 97% accuracy but MCC zero. By MCC, the LSTM branch adds nothing " +
  "over the CNN alone on this data, and a plain Random Forest trained on 300k rows beats every neural model. " +
  "The paper did not compare against tree models.");

// 10 Error analysis
s = pres.addSlide();
title(s, "Where the model fails", "Share of test flows misclassified, per original traffic type");
img(s, FIG("20_error_by_attack_type.png"), 0.5, 1.3, 5.4, 1.429);
const eb = Object.fromEntries(D.error_by_type.map((r) => [r.sub, r]));
const er = (k) => eb[k] ? pct(+eb[k].error_rate, 0) : "n/a";
s.addText([
  { text: "Floods are easy", options: { bold: true, color: TEAL, breakLine: true } },
  { text: `DDoS / DoS: close to 0% missed (e.g. DDoS-SYN ${er("DDoS-SYN_Flood")}).`, options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Quiet attacks are hard", options: { bold: true, color: RED, breakLine: true } },
  { text: `Recon-OSScan ${er("Recon-OSScan")}, DNS spoofing ${er("DNS_Spoofing")}, ARP spoofing ${er("MITM-ArpSpoofing")} missed.`, options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Web attacks: all missed", options: { bold: true, color: RED, breakLine: true } },
  { text: "SQL injection, XSS, backdoor: only 6-27 test flows each; at flow level they look like normal traffic." }],
  { x: 6.2, y: 1.35, w: 3.3, h: 3.8, fontFace: BODY, fontSize: 12.5, color: INK, margin: 0, valign: "top", isTextBox: true });
s.addNotes("The model is excellent on high-volume floods and weak on low-volume attacks. These are rare in the " +
  "data and their flow statistics resemble benign traffic, so class weighting pushes them to 'benign'.");

// 11 Trade-off experiments
s = pres.addSlide();
title(s, "Tuning the trade-off: missed attacks vs false alarms", "Threshold chosen on validation data only; test set untouched");
const T = D.threshold, NB = D.nobalance;
const trow = (name, m) => [name, pct(m.accuracy), m.mcc.toFixed(3), num(m.fn), num(m.fp), pct(m.fpr)];
const body = [trow("Class weights, threshold 0.5 (main)", T.test_at_0_5 || T["test_at_0.5"]),
  trow(`Class weights, threshold ${T.threshold_tuned.toFixed(3)}`, T.test_at_tuned)];
if (NB) body.push(trow("No class weights, threshold 0.5", NB.model));
if (NB && NB.threshold) body.push(trow(`No class weights, threshold ${NB.threshold.threshold_tuned.toFixed(3)}`, NB.threshold.test_at_tuned));
const hdr = ["Setting", "Accuracy", "MCC", "Missed attacks", "False alarms", "FPR"].map((t) =>
  ({ text: t, options: { bold: true, fill: { color: NAVY }, color: WHITE } }));
s.addTable([hdr].concat(body.map((r, i) => r.map((t, j) => ({ text: t, options: { bold: j === 0 || i === 0 && j === 0,
  fill: { color: i === 0 ? "FBE9E7" : WHITE } } })))),
  { x: 0.5, y: 1.45, w: 9, colW: [3.3, 1.1, 0.9, 1.3, 1.2, 1.2], fontFace: BODY, fontSize: 12, color: INK,
    border: { type: "solid", pt: 0.5, color: LINE }, rowH: 0.42 });
s.addText("There is no free lunch: every setting moves errors between the two columns. For an IDS the right " +
  "choice depends on whether analysts can tolerate false alarms or not.", { x: 0.5, y: 3.95, w: 9, h: 0.8,
  fontFace: BODY, fontSize: 13, color: INK, margin: 0, isTextBox: true });
s.addNotes("Two extra experiments: a decision threshold chosen on validation data, and training without class " +
  "weights, which is probably closer to the paper's setup. Both cut missed attacks sharply at the cost of more " +
  "false alarms. The unweighted model has the best MCC, and its false positive rate is close to the paper's 9.17%.");

// 12 Stability + edge
s = pres.addSlide();
title(s, "Stable, and small enough for the edge", "5-fold cross-validation and TensorFlow Lite export");
const cv = D.cv, e8 = D.edge.find((r) => r.variant.includes("int8")), e32 = D.edge.find((r) => r.variant === "TFLite float32");
stat(s, 0.5, 1.4, 2.2, `${pct(cv.accuracy.mean)}`, `CV accuracy, +/- ${(100 * cv.accuracy.std).toFixed(2)} pp over ${cv.folds} folds`, TEAL, 28);
stat(s, 0.5, 2.6, 2.2, `${e8.size_mb.toFixed(2)} MB`, "int8 TFLite model size", TEAL, 28);
stat(s, 2.8, 2.6, 2.2, `${e8.ms_per_sample.toFixed(3)} ms`, "per flow on CPU, batch 1", TEAL, 28);
stat(s, 2.8, 1.4, 2.2, e8.accuracy_drop_pp <= 0.005 ? "None" : `${e8.accuracy_drop_pp.toFixed(2)} pp`, "accuracy lost by int8 quantization", TEAL, 28);
s.addText("A Raspberry Pi could run this model on live flow features.", { x: 0.5, y: 3.85, w: 4.5, h: 0.6,
  fontFace: BODY, fontSize: 13, color: INK, margin: 0, isTextBox: true });
img(s, ANIM("traffic_replay_final_16x9.png"), 5.3, 1.4, 4.2, 1.778);
s.addText("Live-replay demo: real test flows, real model output (see traffic_replay.mp4)", { x: 5.3, y: 3.8, w: 4.2,
  h: 0.5, fontFace: BODY, fontSize: 10.5, color: MUTED, margin: 0, isTextBox: true });
s.addNotes(`Fold-to-fold spread is tiny. Quantized to int8, the model is ${e8.size_mb.toFixed(2)} MB and ` +
  `classifies a flow in ${e8.ms_per_sample.toFixed(3)} ms on a CPU with no accuracy loss (float32 TFLite: ` +
  `${e32.size_mb.toFixed(2)} MB).`);

// 13 Deviations
s = pres.addSlide();
title(s, "Why our numbers may differ from the paper", "Documented in deviation_analysis.txt");
const dev = [["Sample", `${(RI.rows_sampled / 1e6).toFixed(1)}M-row random sample vs paper's own 1.19M subset`],
  ["Split", "70/15/15 stratified vs paper's 80/20 + separate test files"],
  ["Duplicates", `${num(RI.duplicates_removed)} removed before splitting (paper: not mentioned)`],
  ["Hyperparameters", "batch size, learning rate, pooling assumed"],
  ["Class weights", "we weight the rare benign class; paper does not say"],
  ["Features", `${RI.n_features_used} used vs 45 in the paper`]];
dev.forEach(([h, d], i) => {
  const x = 0.5 + (i % 3) * 3.05, y = 1.45 + Math.floor(i / 3) * 1.75;
  card(s, x, y, 2.85, 1.5);
  badge(s, x + 0.2, y + 0.2, String(i + 1));
  s.addText(h, { x: x + 0.75, y: y + 0.2, w: 1.95, h: 0.42, fontFace: HEAD, fontSize: 14, bold: true, color: INK,
    valign: "middle", margin: 0, isTextBox: true });
  s.addText(d, { x: x + 0.2, y: y + 0.72, w: 2.5, h: 0.7, fontFace: BODY, fontSize: 11.5, color: MUTED, margin: 0,
    valign: "top", isTextBox: true });
});
s.addNotes("None of these is a mistake in either work; they are choices the paper does not specify. Removing " +
  "duplicates before splitting is the one we consider most important for an honest test score.");

// 14 Conclusion
s = pres.addSlide();
s.background = { color: NAVY };
s.addText("Conclusions", { x: 0.6, y: 0.45, w: 8.8, h: 0.7, fontFace: HEAD, fontSize: 34, bold: true, color: WHITE,
  margin: 0, isTextBox: true });
const concl = [
  [`Reproduced: ${pct(CL.accuracy)} accuracy vs the paper's ${P.accuracy}%`, "The CNN-LSTM works as described on real CICIoT2023 traffic."],
  ["Accuracy is not enough", `97% of flows are attacks; MCC ${CL.mcc.toFixed(3)} and per-attack errors tell the real story.`],
  ["Simpler models do as well or better", "No gain from the LSTM branch; Random Forest is strongest here."],
  ["Next steps", "Multiclass detection, rare-attack handling, and a live Raspberry Pi deployment."]];
concl.forEach(([h, d], i) => {
  const y = 1.45 + i * 0.95;
  badge(s, 0.6, y + 0.05, String(i + 1), "0E8C80");
  s.addText(h, { x: 1.2, y, w: 8.2, h: 0.38, fontFace: HEAD, fontSize: 16, bold: true, color: WHITE, margin: 0, isTextBox: true });
  s.addText(d, { x: 1.2, y: y + 0.38, w: 8.2, h: 0.38, fontFace: BODY, fontSize: 13, color: "C9D3DD", margin: 0, isTextBox: true });
});
s.addNotes("We reproduced the paper's headline result, showed why accuracy alone is misleading on this data, " +
  "and found that simpler models are at least as good. Code, figures and results are all in the repository.");

pres.writeFile({ fileName: path.join(__dirname, "IoT_IDS_CNN_LSTM_slides.pptx") })
  .then((f) => console.log("written", f));
