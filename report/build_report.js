// Build IoT_IDS_CNN_LSTM_report.docx from report_data.json + pipeline figures.
// usage: node build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, PageBreak, LevelFormat,
  Footer, PageNumber, Header,
} = require("docx");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "report_data.json"), "utf8"));
const FIG = (n) => path.join(__dirname, "..", "outputs", "figures", n);
const M = D.models, CL = M["CNN-LSTM"], MAJ = M["Majority (always malicious)"], RI = D.run_info, P = D.paper;
const pct = (x, d = 2) => (100 * x).toFixed(d) + "%";
const num = (x) => Number(x).toLocaleString("en-US");
const FONT = "Calibri", NAVY = "0F1B2D", LINE = "BFC7D0";
const PAGE_W = 11906, MARGIN = 1134, CONTENT_W = PAGE_W - 2 * MARGIN; // A4, 2 cm margins

// ---- helpers ----------------------------------------------------------------
const p = (text, opts = {}) => new Paragraph({ spacing: { after: 120, line: 276 }, ...opts,
  children: (Array.isArray(text) ? text : [text]).map((t) => typeof t === "string" ? new TextRun(t) : t) });
const b = (t) => new TextRun({ text: t, bold: true });
const i_ = (t) => new TextRun({ text: t, italics: true });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)], spacing: { before: 280, after: 140 } });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)], spacing: { before: 200, after: 100 } });
const bullet = (t) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60 },
  children: (Array.isArray(t) ? t : [t]).map((x) => typeof x === "string" ? new TextRun(x) : x) });
let listNo = 0; // each numbered list restarts at 1: bump with newList()
const newList = () => { listNo += 1; };
const numbered = (t) => new Paragraph({ numbering: { reference: "num", level: 0, instance: listNo }, spacing: { after: 60 },
  children: (Array.isArray(t) ? t : [t]).map((x) => typeof x === "string" ? new TextRun(x) : x) });

function figure(file, widthIn, ratio, caption) {
  const w = Math.round(widthIn * 96), h = Math.round((widthIn / ratio) * 96);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: w, height: h },
        altText: { title: caption, description: caption, name: path.basename(file) } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: caption, italics: true, size: 18, color: "555555" })] }),
  ];
}

function table(header, rows, widths, caption) {
  const total = widths.reduce((a, c) => a + c, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: LINE };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (t, w, head, shade) => new TableCell({ width: { size: w, type: WidthType.DXA }, borders,
    shading: head ? { type: ShadingType.CLEAR, fill: NAVY, color: "auto" } : shade ? { type: ShadingType.CLEAR, fill: shade, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text: String(t), bold: head, color: head ? "FFFFFF" : "1F2937", size: 19 })] })] });
  const out = [new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((t, k) => cell(t, widths[k], true)) })]
      .concat(rows.map((r) => new TableRow({ children: r.cells.map((t, k) => cell(t, widths[k], false, r.shade)) }))) })];
  if (caption) out.push(new Paragraph({ spacing: { before: 80, after: 200 },
    children: [new TextRun({ text: caption, italics: true, size: 18, color: "555555" })] }));
  return out;
}
const R = (cells, shade) => ({ cells, shade });

// ---- content ------------------------------------------------------------------
const children = [];
// Title page
children.push(
  new Paragraph({ spacing: { before: 2400, after: 240 }, children: [new TextRun({ text: "IoT Network Intrusion Detection",
    bold: true, size: 52, color: NAVY, font: "Arial" })] }),
  new Paragraph({ spacing: { after: 480 }, children: [new TextRun({ text: "Implementation and reproduction of a CNN-LSTM based Intrusion Detection System on CICIoT2023",
    size: 30, color: "0E8C80" })] }),
  p([i_("Reproduced paper: "), new TextRun("A. Gueriani, H. Kheddar and A. C. Mazari, \"Enhancing IoT Security with CNN and LSTM-Based Intrusion Detection Systems\", 2024 6th International Conference on Pattern Analysis and Intelligent Systems (PAIS), IEEE, 2024. DOI 10.1109/PAIS62114.2024.10541178.")]),
  new Paragraph({ spacing: { before: 600, after: 80 }, children: [b("Team")] }),
  p("Aayushman (231CS105)", { spacing: { after: 40 } }),
  p("Ashutosh Kumar (231CS113)", { spacing: { after: 40 } }),
  p("Sahil Mengji (231CS151)", { spacing: { after: 40 } }),
  new Paragraph({ children: [new PageBreak()] }),
);

// Abstract
children.push(h1("Abstract"),
  p(`We reproduce the CNN-LSTM intrusion detection system of Gueriani et al. (IEEE PAIS 2024), which reports ${P.accuracy}% accuracy for binary benign-versus-malicious classification of IoT network flows on the CICIoT2023 dataset. We rebuilt the two-branch architecture from the paper's figure, trained it on a ${num(RI.rows_sampled)}-row stratified sample of the original 46-feature release (${num(RI.rows_after_cleaning)} flows after removing exact duplicates), and evaluated it on a held-out test split of ${num(RI.n_test)} flows. Our model reaches ${pct(CL.accuracy)} accuracy and ${pct(CL.f1_weighted)} weighted F1, against the paper's ${P.accuracy}% and ${P.f1}%, so the headline result reproduces. Because ${pct(MAJ.accuracy)} of test flows are attacks, we also report imbalance-robust metrics: MCC ${CL.mcc.toFixed(3)}, benign recall ${pct(CL.benign_recall)}, false positive rate ${pct(CL.fpr)} and false negative rate ${pct(CL.fnr)}. Baselines show that the LSTM branch gives no measurable gain over the CNN alone, and that a Random Forest outperforms every neural model (MCC ${M.RandomForest.mcc.toFixed(3)}). We document all deviations from the paper's setup, a decision-threshold study, cross-validation, and a TensorFlow Lite deployment benchmark.`));

// 1 Introduction
newList();
children.push(h1("1. Introduction and objectives"),
  p("IoT devices are numerous, heterogeneous and often poorly secured, which makes network-based intrusion detection an important line of defence. Deep-learning IDSs that combine convolutional and recurrent layers are a popular choice in recent work. This project does not propose a new method: its goal is a careful scientific reproduction of one such system and an honest assessment of whether its reported performance holds."),
  p("Our objectives were:"),
  numbered("Reproduce the CNN-LSTM methodology described in the paper."),
  numbered("Train the model on the CICIoT2023 dataset."),
  numbered("Build a working pipeline that distinguishes benign from malicious traffic."),
  numbered(`Compare accuracy, precision, recall and F1 with the paper's reported ${P.accuracy}% accuracy.`),
  numbered("Document every deviation between our setup and the paper's."));

// 2 Paper summary
children.push(h1("2. The reproduced paper"),
  p("Gueriani et al. propose a hybrid model in which 1D convolutional layers extract local patterns from the flow features and LSTM layers model dependencies between the learned features. The output is a binary decision: benign or malicious. They train on a subset of CICIoT2023 and additionally test on CICIDS2017."),
  p([b("Reported results. "), new TextRun(`The published abstract reports accuracy ${P.accuracy}%, loss ${P.loss}, false positive rate ${P.fpr}% and F1-score ${P.f1}%. Precision and recall are given only in a full-text table that we could not access, so we do not use them. The abstract does not say whether F1 is weighted over both classes or computed for the attack class; we therefore report both.`)]),
  p([b("Architecture (paper Fig. 2). "), new TextRun("Input sequence of flow features > Conv1D(64, kernel 3) > BatchNormalization > AveragePooling1D > Conv1D(64, kernel 3) > AveragePooling1D > Flatten > Dense(32) > Dense(16). From there two heads: (a) Dense(2, softmax); (b) Reshape(16, 1) > LSTM(64) > LSTM(64) > Dense(2, sigmoid). The heads are concatenated and passed to a final Dense(2, softmax). The paper trains for 25 epochs with Adam.")]));

// 3 Data
children.push(h1("3. Dataset and preprocessing"),
  h2("3.1 CICIoT2023"),
  p(`CICIoT2023 was captured by the Canadian Institute for Cybersecurity on a testbed of 105 IoT devices and contains benign traffic and 33 attacks grouped into 7 families (DDoS, DoS, Mirai, Recon, Spoofing, Web-based, Brute force). We used the original release with 46 flow-level features, which is the release the paper used. The files available to us contain ${num(RI.population_rows)} flows. The official download page requires a registration form, so the files were obtained from a public mirror of the official CIC release.`),
  ...figure(FIG("01_class_distribution.png"), 6.6, 3.556, "Figure 1 - Benign vs malicious flows in the full data, our sample and the training split (before and after class weighting)."),
  h2("3.2 Pipeline"),
  bullet([b("Sampling: "), new TextRun(`${num(RI.rows_sampled)} rows were drawn with the same keep-probability for every class, so class ratios match the full data.`)]),
  bullet([b("Cleaning: "), new TextRun(`${num(RI.duplicates_removed)} exact duplicate rows were removed before splitting, so no identical flow can appear in both training and test data. No missing or infinite values were present.`)]),
  bullet([b("Split: "), new TextRun(`stratified 70/15/15 into ${num(RI.n_train)} training, ${num(RI.n_val)} validation and ${num(RI.n_test)} test flows (seed ${RI.seed}).`)]),
  bullet([b("Features: "), new TextRun(`${RI.n_features_used} of ${RI.n_features_raw} features were kept; ${RI.dropped_constant.join(", ")} are constant on the training split and were dropped.`)]),
  bullet([b("Scaling and imbalance: "), new TextRun(`Min-max scaling and 'balanced' class weights (benign ${Number(RI.class_weight["0"]).toFixed(2)}, malicious ${Number(RI.class_weight["1"]).toFixed(3)}) were fitted on the training split only.`)]),
  p("Automated tests verify that the splits are disjoint and that the scaler and class weights see training data only."));

// 4 Setup
const st = D.stage_runtime;
children.push(h1("4. Experimental setup"),
  ...table(["Item", "Value", "Source"], [
    R(["Epochs", `${RI.epochs_max} (all run; best epoch ${RI.best_epoch})`, "paper"]),
    R(["Optimiser", `Adam, learning rate ${RI.learning_rate}`, "paper (lr assumed)"]),
    R(["Batch size", String(RI.batch_size), "assumed"]),
    R(["Pooling size / activations", "2 / ReLU", "assumed (consistent with Fig. 2 shapes)"]),
    R(["Callbacks", "early stopping (patience 5), LR halving on plateau, best checkpoint", "assumed"]),
    R(["Parameters", num(CL.total_params), "measured"]),
    R(["Hardware", "NVIDIA RTX 4060 Laptop GPU, TensorFlow 2.20 in WSL2", "-"]),
    R(["Training time", `${(RI.train_time_s / 60).toFixed(1)} min (about 42 s per epoch)`, "measured"]),
  ], [2600, 4400, 2638], "Table 1 - Training configuration. 'Assumed' values are not stated in the paper."),
  p(`The pipeline is split into resumable stages (prepare, train, baselines, cross-validation, comparison, edge export). An early full run crashed when the host ran out of memory; we fixed this by caching prepared arrays as memory-mapped files, feeding batches directly from them, subsampling for the Random Forest and running jobs detached. Peak memory of the full run stayed below ${Math.max(...Object.values(st).map((v) => v.peak_rss_gb)).toFixed(1)} GB.`),
  ...figure(FIG("06_training_accuracy.png"), 5.2, 1.6, "Figure 2 - Training and validation accuracy per epoch."),
  ...figure(FIG("07_training_loss.png"), 5.2, 1.6, "Figure 3 - Training and validation loss per epoch."));

// 5 Results
children.push(h1("5. Results"),
  h2("5.1 Test-set performance"),
  p(`All numbers are computed once on the held-out test split. Positive class = malicious. The 'always malicious' row shows what a model with no skill scores on the same data.`),
  ...table(["Model", "Acc.", "F1 (mal.)", "Benign P", "Benign R", "FPR", "FNR", "MCC", "ROC-AUC"],
    ["CNN-LSTM", "CNN", "LSTM", "MLP", "RandomForest", "Majority (always malicious)"].map((k) => R([
      k === "Majority (always malicious)" ? "Always malicious" : k === "RandomForest" ? "Random Forest" : k,
      pct(M[k].accuracy), pct(M[k].f1), pct(M[k].benign_precision), pct(M[k].benign_recall), pct(M[k].fpr),
      pct(M[k].fnr), M[k].mcc.toFixed(3), M[k].roc_auc.toFixed(4)], k === "CNN-LSTM" ? "FBE9E7" : undefined))
      .concat([R(["Paper (CNN-LSTM)", `${P.accuracy}%`, "n/a", "n/a", "n/a", `${P.fpr}%`, "n/a", "n/a", "n/a"], "EEF2F6")]),
    [2100, 900, 1000, 950, 950, 850, 850, 900, 1138],
    "Table 2 - Test-set metrics (n = " + num(RI.n_test) + "). Paper values from the published abstract."),
  p(`The CNN-LSTM reaches ${pct(CL.accuracy)} accuracy, ${(100 * CL.accuracy - P.accuracy).toFixed(2)} percentage points above the paper. Its weighted F1 of ${pct(CL.f1_weighted)} is close to the paper's ${P.f1}%. However, always predicting 'malicious' already scores ${pct(MAJ.accuracy)}, so accuracy says little on its own. The MCC of ${CL.mcc.toFixed(3)} and the confusion matrix give a clearer picture: the model raises almost no false alarms (${num(CL.fp)} of ${num(CL.tn + CL.fp)} benign flows) but misses ${num(CL.fn)} attacks (${pct(CL.fnr)}).`),
  ...figure(FIG("10_confusion_matrix_normalized.png"), 3.6, 1.167, "Figure 4 - Normalised confusion matrix of the CNN-LSTM on the test set."),
  ...figure(FIG("11_roc_curve.png"), 3.6, 1.167, "Figure 5 - ROC curves of all models."),
  h2("5.2 Comparison with the paper"),
  ...table(["Metric", "Paper", "Ours", "Difference"], [
    R(["Accuracy", `${P.accuracy}%`, pct(CL.accuracy), `${(100 * CL.accuracy - P.accuracy >= 0 ? "+" : "")}${(100 * CL.accuracy - P.accuracy).toFixed(2)} pp`]),
    R(["F1 (weighted)", `${P.f1}%`, pct(CL.f1_weighted), `${(100 * CL.f1_weighted - P.f1 >= 0 ? "+" : "")}${(100 * CL.f1_weighted - P.f1).toFixed(2)} pp`]),
    R(["False positive rate", `${P.fpr}%`, pct(CL.fpr), `${(100 * CL.fpr - P.fpr).toFixed(2)} pp`]),
    R(["Loss (cross-entropy)", String(P.loss), CL.log_loss.toFixed(4), (CL.log_loss - P.loss).toFixed(4)]),
    R(["Precision / recall", "not available", `${pct(CL.precision_weighted)} / ${pct(CL.recall_weighted)} (weighted)`, "-"]),
  ], [2600, 2000, 2800, 2238], "Table 3 - Our CNN-LSTM against the paper's published figures."),
  ...figure(FIG("18_paper_comparison.png"), 6.4, 1.897, "Figure 6 - Our results next to the paper's (dashed line: paper accuracy)."),
  p(`Accuracy and F1 reproduce within about a third of a percentage point. The largest difference is the false positive rate: the paper flags about one in eleven benign flows as malicious, while our class-weighted model flags almost none. Our errors instead fall on the attack side. Section 6 shows that this balance can be shifted` +
    (D.nobalance ? `: retrained without class weights, the same architecture has an FPR of ${pct(D.nobalance.model.fpr)} and a loss of ${D.nobalance.model.log_loss.toFixed(4)}, much closer to the paper's ${P.fpr}% and ${P.loss}, which suggests the paper did not re-weight the classes.` : ".")));

// 5.3 baselines
children.push(h2("5.3 Baselines and ablation"),
  p(`To see what each part of the model contributes, we trained a CNN-only model (the paper's CNN part without the LSTM branch), an LSTM-only model, an MLP and a Random Forest on the same data (the Random Forest on a ${num(M.RandomForest.train_rows)}-row stratified subsample).`),
  ...figure(FIG("17_model_comparison.png"), 6.4, 2.0, "Figure 7 - Accuracy, precision, recall and F1 of all models."),
  bullet([b("The LSTM branch does not help here. "), new TextRun(`CNN-only scores MCC ${M.CNN.mcc.toFixed(3)} against ${CL.mcc.toFixed(3)} for the CNN-LSTM. The flows are independent records, not time series, so there is little sequential structure for the LSTM to exploit.`)]),
  bullet([b("All neural models are close. "), new TextRun(`Accuracy ranges from ${pct(M.LSTM.accuracy)} to ${pct(M.CNN.accuracy)}.`)]),
  bullet([b("Random Forest is strongest. "), new TextRun(`${pct(M.RandomForest.accuracy)} accuracy and MCC ${M.RandomForest.mcc.toFixed(3)}, with far fewer missed attacks (FNR ${pct(M.RandomForest.fnr)}) at a higher FPR (${pct(M.RandomForest.fpr)}). The paper does not compare against tree ensembles.`)]),
  ...figure(FIG("19_inference_cost_comparison.png"), 6.4, 2.6, "Figure 8 - Inference time per sample and model size."));

// 5.4 error analysis, CV, edge
const cv = D.cv, e8 = D.edge.find((r) => r.variant.includes("int8")), e32 = D.edge.find((r) => r.variant === "TFLite float32");
children.push(h2("5.4 Where the model errs"),
  ...figure(FIG("20_error_by_attack_type.png"), 5.6, 1.429, "Figure 9 - Share of test flows misclassified per original traffic type."),
  p("High-volume floods (DDoS, DoS, Mirai) are detected almost perfectly. The missed attacks are low-volume ones - reconnaissance scans, DNS and ARP spoofing, dictionary brute force - and the web attacks, which have only 6 to 27 test flows each. At the flow level these look much like normal traffic, and class weighting pushes borderline flows towards 'benign'."),
  h2("5.5 Stability (cross-validation)"),
  p(`Five-fold stratified cross-validation on a ${num(cv.rows)}-row subset of the training and validation data (scaler re-fitted per fold, ${cv.epochs_per_fold} epochs per fold) gives accuracy ${pct(cv.accuracy.mean)} +/- ${(100 * cv.accuracy.std).toFixed(2)} pp and benign recall ${pct(cv.benign_recall.mean)} +/- ${(100 * cv.benign_recall.std).toFixed(2)} pp. The result does not depend on a lucky split.`),
  ...figure(FIG("22_cross_validation.png"), 4.8, 1.6, "Figure 10 - Metric spread across the five folds."),
  h2("5.6 Edge deployment"),
  ...table(["Variant", "Size (MB)", "Latency (ms/flow)", "Accuracy"], D.edge.map((r) => R([r.variant, r.size_mb.toFixed(3),
    r.ms_per_sample.toFixed(3), pct(r.accuracy)])), [3600, 1800, 2200, 2038],
    `Table 4 - TensorFlow Lite export, benchmarked one flow at a time on ${num(20000)} test flows (Keras row: GPU).`),
  p(`The dynamic-range quantised model is ${e8.size_mb.toFixed(2)} MB (float32: ${e32.size_mb.toFixed(2)} MB) and classifies a flow in ${e8.ms_per_sample.toFixed(3)} ms on a CPU with no loss of accuracy, which makes a Raspberry Pi-class edge IDS feasible.`));

// 6 Additional experiments
const T = D.threshold, NB = D.nobalance, t05 = T["test_at_0.5"];
const expRows = [R(["Class weights, threshold 0.5 (main)", pct(t05.accuracy), t05.mcc.toFixed(3), num(t05.fn), num(t05.fp), pct(t05.fpr)], "FBE9E7"),
  R([`Class weights, threshold ${T.threshold_tuned.toFixed(3)}`, pct(T.test_at_tuned.accuracy), T.test_at_tuned.mcc.toFixed(3), num(T.test_at_tuned.fn), num(T.test_at_tuned.fp), pct(T.test_at_tuned.fpr)])];
if (NB) {
  expRows.push(R(["No class weights, threshold 0.5", pct(NB.model.accuracy), NB.model.mcc.toFixed(3), num(NB.model.fn), num(NB.model.fp), pct(NB.model.fpr)]));
  if (NB.threshold) expRows.push(R([`No class weights, threshold ${NB.threshold.threshold_tuned.toFixed(3)}`, pct(NB.threshold.test_at_tuned.accuracy),
    NB.threshold.test_at_tuned.mcc.toFixed(3), num(NB.threshold.test_at_tuned.fn), num(NB.threshold.test_at_tuned.fp), pct(NB.threshold.test_at_tuned.fpr)]));
}
children.push(h1("6. Additional experiments: the error trade-off"),
  p("Our main model misses attacks rather than raising false alarms, the opposite of the paper's error profile. We ran two experiments to see how much of this is a choice rather than a property of the architecture."),
  bullet([b("Decision threshold. "), new TextRun("Instead of 0.5, the threshold on P(malicious) was chosen to maximise MCC on the validation split. The test split was not used for the choice.")]),
  bullet([b("No class weights. "), new TextRun("The model was retrained with identical settings but without class weights, which is closer to the paper's (unstated) setup.")]),
  ...table(["Setting", "Accuracy", "MCC", "Missed attacks", "False alarms", "FPR"], expRows, [3500, 1200, 1000, 1350, 1300, 1288],
    "Table 5 - Test-set error trade-off under different settings."),
  ...figure(FIG("24_threshold_tuning.png"), 5.4, 1.8, "Figure 11 - Validation MCC, FPR and FNR as a function of the decision threshold."),
  p(NB ? `Lowering the threshold to ${T.threshold_tuned.toFixed(3)} cuts missed attacks from ${num(t05.fn)} to ${num(T.test_at_tuned.fn)} and raises MCC to ${T.test_at_tuned.mcc.toFixed(3)}, at the cost of ${num(T.test_at_tuned.fp)} false alarms. Without class weights the model reaches ${pct(NB.model.accuracy)} accuracy and MCC ${NB.model.mcc.toFixed(3)} with ${num(NB.model.fn)} missed attacks and ${num(NB.model.fp)} false alarms (FPR ${pct(NB.model.fpr)}). The operating point is a deployment decision: it depends on how many false alarms the analysts can handle.`
    : `Lowering the threshold to ${T.threshold_tuned.toFixed(3)} cuts missed attacks from ${num(t05.fn)} to ${num(T.test_at_tuned.fn)} and raises MCC to ${T.test_at_tuned.mcc.toFixed(3)}, at the cost of ${num(T.test_at_tuned.fp)} false alarms.`));

// 7 Deviations
newList();
children.push(h1("7. Deviations from the paper"),
  numbered([b("Sample size: "), new TextRun(`we used a ${num(RI.rows_sampled)}-row random sample of ${num(RI.population_rows)} rows; the paper used a 1,191,264-row train/validation subset and a separate 1,175,692-row test subset.`)]),
  numbered([b("Split: "), new TextRun("stratified 70/15/15 instead of 80/20 plus a separate test subset, so the test sets are not identical.")]),
  numbered([b("Duplicates: "), new TextRun(`${num(RI.duplicates_removed)} exact duplicates were removed before splitting; the paper does not mention de-duplication, and duplicates shared across splits usually inflate test scores.`)]),
  numbered([b("Hyperparameters: "), new TextRun("batch size, learning rate, pooling size, activations and callbacks are not given in the paper and were assumed.")]),
  numbered([b("Class imbalance: "), new TextRun("we used class weights; the paper does not mention re-balancing, which is consistent with its much higher FPR.")]),
  numbered([b("Features: "), new TextRun(`${RI.n_features_used} features after dropping constant columns; the paper states 45 without saying which feature it removed.`)]),
  numbered([b("Dataset copy and seed: "), new TextRun(`files from a public mirror of the official release; seed ${RI.seed}. GPU kernels are not bit-exact, so re-runs can differ by a few hundredths of a point.`)]),
  numbered([b("Metric convention: "), new TextRun("the abstract does not say how F1 is averaged; we report weighted and attack-class values.")]));

// 8 Conclusion
children.push(h1("8. Conclusion and future work"),
  p(`We reproduced the CNN-LSTM IDS of Gueriani et al. on CICIoT2023 and obtained ${pct(CL.accuracy)} accuracy and ${pct(CL.f1_weighted)} weighted F1, in line with the published ${P.accuracy}% and ${P.f1}%. The reproduction therefore supports the paper's headline claim. Our analysis adds three points the paper does not make: accuracy is a weak measure on this heavily skewed dataset; the LSTM branch adds nothing measurable over the CNN alone; and a Random Forest is the strongest model we tested. The model's weaknesses are low-volume attacks, which future work could address with multiclass training, rare-class oversampling or per-host temporal features. A live deployment on a Raspberry Pi, for which the TFLite model is already small and fast enough, is a natural next step.`));

// References
newList();
children.push(h1("References"),
  numbered("A. Gueriani, H. Kheddar, A. C. Mazari, \"Enhancing IoT Security with CNN and LSTM-Based Intrusion Detection Systems,\" 2024 6th Int. Conf. on Pattern Analysis and Intelligent Systems (PAIS), IEEE, 2024. DOI: 10.1109/PAIS62114.2024.10541178. Preprint: arXiv:2405.18624."),
  numbered("E. C. P. Neto et al., \"CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment,\" Sensors, 23(13), 5941, 2023. Dataset: https://www.unb.ca/cic/datasets/iotdataset-2023.html."));

// Appendix
children.push(h1("Appendix: reproducing the results"),
  p("All code, figures and metrics are in the project repository. To rerun:"),
  bullet("pip install -r requirements.txt  (GPU training on Windows requires WSL2)"),
  bullet("python download_data.py --release original"),
  bullet("python -m pytest -q test_leakage.py test_pipeline.py"),
  bullet("python main.py --stage prepare,train,baselines,cv,compare,edge --sample 1400000 --epochs 25"),
  bullet("python tools/tune_threshold.py; python simulation.py  (threshold study and demo animation)"));

// ---- document -----------------------------------------------------------------
const doc = new Document({
  creator: "Aayushman, Ashutosh Kumar, Sahil Mengji",
  title: "IoT Network Intrusion Detection - CNN-LSTM IDS reproduction",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: NAVY }, paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0, keepNext: true, keepLines: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: "0E8C80" }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1, keepNext: true, keepLines: true } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "CNN-LSTM IDS reproduction on CICIoT2023", size: 16, color: "888888" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "888888" })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "IoT_IDS_CNN_LSTM_report.docx");
  fs.writeFileSync(out, buf);
  console.log("written", out);
});
