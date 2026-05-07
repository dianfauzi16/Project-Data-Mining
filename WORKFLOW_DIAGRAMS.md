# 📊 PROJECT WORKFLOW & ARCHITECTURE DIAGRAMS

## Diagram 1: Overall Project Pipeline

```mermaid
graph TD
    A["📥 01: DATA ACQUISITION"] --> B["📊 02: EDA"]
    B --> C["🧹 03: PREPROCESSING"]
    C --> D["✂️ 04: TRAIN-TEST SPLIT<br/>80-20 Stratified"]
    D --> E["80% Training Data"]
    D --> F["20% Test Data<br/>Held-Out"]
    E --> G["⚖️ 05: MLSMOTE BALANCING<br/>Handle Imbalance"]
    G --> H["🎯 06: MFO FEATURE SELECTION<br/>50+ features → 31 features"]
    H --> I["🔄 06b: ABLATION STUDY<br/>Transfer Function Comparison"]
    H --> J["🤖 07: HYPERPARAMETER TUNING<br/>XGBoost + ClassifierChain"]
    J --> K["🏆 07a-07g: BENCHMARKING STUDIES<br/>6 ablation studies"]
    J --> L["✅ 08: FINAL EVALUATION<br/>on Test Set"]
    L --> M["🎚️ 08b: THRESHOLD OPTIMIZATION<br/>Per-label tuning"]
    L --> N["📈 09: 5-FOLD CROSS-VALIDATION<br/>Robustness Check"]
    M --> O["🔍 10: SHAP EXPLAINABILITY<br/>Feature Importance Analysis"]
    O --> P["🚀 11: DEPLOYMENT<br/>Production-Ready Model"]
    F --> L
    F --> N
    P --> Q["📦 Final Output<br/>model.pkl + thresholds"]

    style A fill:#e1f5ff
    style B fill:#e1f5ff
    style C fill:#f3e5f5
    style D fill:#f3e5f5
    style E fill:#fff3e0
    style F fill:#fff3e0
    style G fill:#fff3e0
    style H fill:#c8e6c9
    style I fill:#c8e6c9
    style J fill:#c8e6c9
    style K fill:#ffe0b2
    style L fill:#ffccbc
    style M fill:#ffccbc
    style N fill:#ffccbc
    style O fill:#d1c4e9
    style P fill:#b2dfdb
    style Q fill:#a5d6a7
```

---

## Diagram 2: Data Flow Through Preprocessing

```mermaid
graph LR
    A["Raw Dataset<br/>DASS-42"] -->|Age filter: 15-24| B["Filtered Data"]
    B -->|Remove outliers<br/>age=1998| C["Cleaned Data"]
    C -->|Feature Engineering<br/>DASS scores| D["Scored Features"]
    D -->|Prevent leakage<br/>Remove original Q's| E["Clean Features"]
    E -->|Stratified 80-20| F["Train: 80%"]
    E -->|Stratified 80-20| G["Test: 20%<br/>Hold-out"]
    F -->|MLSMOTE| H["Balanced Train<br/>50-50 minority-majority"]
    H -->|MFO Selection| I["Selected Features<br/>31/50+"]
    I -->|Ready for modeling| J["Training Phase"]
    G -->|Final Evaluation| K["Testing Phase"]

    style A fill:#fff3e0
    style B fill:#ffe0b2
    style C fill:#ffcc80
    style D fill:#ffb74d
    style E fill:#ffa726
    style F fill:#ff9800
    style G fill:#f57c00
    style H fill:#e65100
    style I fill:#c8e6c9
    style J fill:#81c784
    style K fill:#558b2f
```

---

## Diagram 3: Multi-Label Classification Strategy

```mermaid
graph TD
    A["Input Features<br/>31 selected features"] --> B["ClassifierChain<br/>Random Order"]
    B --> C1["Model 1:<br/>XGBoost<br/>→ Predict risk_depression"]
    B --> C2["Model 2:<br/>XGBoost<br/>→ Predict risk_anxiety<br/>+ uses pred_depression"]
    B --> C3["Model 3:<br/>XGBoost<br/>→ Predict risk_stress<br/>+ uses pred_depression & anxiety"]

    C1 --> D["Output 1:<br/>Probability depression"]
    C2 --> E["Output 2:<br/>Probability anxiety"]
    C3 --> F["Output 3:<br/>Probability stress"]

    D --> G["Apply Threshold 1: 0.41"]
    E --> H["Apply Threshold 2: 0.37"]
    F --> I["Apply Threshold 3: 0.38"]

    G --> J["Binary pred:<br/>risk_depression"]
    H --> K["Binary pred:<br/>risk_anxiety"]
    I --> L["Binary pred:<br/>risk_stress"]

    J --> M["Final Output<br/>3 binary labels"]
    K --> M
    L --> M

    style A fill:#bbdefb
    style B fill:#90caf9
    style C1 fill:#64b5f6
    style C2 fill:#64b5f6
    style C3 fill:#64b5f6
    style D fill:#42a5f5
    style E fill:#42a5f5
    style F fill:#42a5f5
    style G fill:#2196f3
    style H fill:#2196f3
    style I fill:#2196f3
    style J fill:#1976d2
    style K fill:#1976d2
    style L fill:#1976d2
    style M fill:#0d47a1
```

---

## Diagram 4: MFO Feature Selection Process

```mermaid
graph TD
    A["Initialize<br/>30 moths<br/>Binary positions"] --> B["Iteration 1-20"]
    B --> C["For each moth:<br/>Evaluate Fitness"]
    C --> D["Fitness = Cost<br/>cost = 1-F1 + alpha*features_ratio"]
    D --> E["Update Flames<br/>Best solutions"]
    E --> F["Update Moth Positions<br/>Using Spiral + Transfer"]
    F --> G{" Conv?"}
    G -->|No| B
    G -->|Yes| H["Return Best<br/>Feature Mask"]
    H --> I["31 Optimal Features<br/>Selected!"]

    F --> F1["Logarithmic Spiral:<br/>pos = distance × exp × cos + flame_pos"]
    F1 --> F2["S-Shaped Transfer:<br/>sigmoid = 1/(1+exp(-x))"]
    F2 --> F3["Binary Convert:<br/>if sigmoid > 0.5 then 1 else 0"]

    style A fill:#c8e6c9
    style B fill:#a5d6a7
    style C fill:#81c784
    style D fill:#66bb6a
    style E fill:#4caf50
    style F fill:#43a047
    style G fill:#2e7d32
    style H fill:#1b5e20
    style I fill:#558b2f
    style F1 fill:#fff9c4
    style F2 fill:#fff9c4
    style F3 fill:#fff9c4
```

---

## Diagram 5: Benchmarking Studies (Phase 5)

```mermaid
graph TD
    A["07: XGBoost Tuned"] --> B["07a: MFO Effect<br/>WITH vs WITHOUT<br/>MFO selection"]
    A --> C["07b: Algorithm Comparison<br/>RF vs XGBoost<br/>Fair tuning"]
    A --> D["07c: EPIC Trinity<br/>RF 🌲 vs XGBoost 🚀<br/>vs CatBoost 🐱"]
    A --> E["07d: Comprehensive<br/>Baseline<br/>RF+Chain, SVM,<br/>Stacking, etc"]
    A --> F["07e: XGBoost NO MFO<br/>All 50+ features<br/>vs 31 selected"]
    A --> G["07f: BPSO vs MFO<br/>Feature selection<br/>methods"]
    A --> H["07h: GA vs MFO<br/>Feature selection<br/>methods"]

    B --> I["Result: MFO improves<br/>performance +X%"]
    C --> J["Result: Algorithm<br/>ranking"]
    D --> K["Winner: XGBoost 🚀"]
    E --> L["Result: Baseline<br/>comparison"]
    F --> M["Result: Feature<br/>reduction value"]
    G --> N["Result: MFO > BPSO"]
    H --> O["Result: MFO > GA"]

    style A fill:#ff9800
    style B fill:#ffb74d
    style C fill:#ffb74d
    style D fill:#ffb74d
    style E fill:#ffb74d
    style F fill:#ffb74d
    style G fill:#ffb74d
    style H fill:#ffb74d
    style K fill:#00bcd4
```

---

## Diagram 6: Evaluation & Validation Pipeline

```mermaid
graph TD
    A["Final Tuned Model"] --> B["08: FINAL TEST<br/>Evaluate on test set"]
    A --> C["09: K-FOLD CV<br/>5-fold validation"]

    B --> D["Test Metrics"]
    D --> D1["Exact Match: X%"]
    D --> D2["F1 Macro: 0.XXXX"]
    D --> D3["F1 Micro: 0.XXXX"]
    D --> D4["Precision: 0.XXXX"]
    D --> D5["Recall: 0.XXXX"]
    D --> D6["Hamming Loss: X%"]

    C --> E["K-Fold Metrics"]
    E --> E1["Fold 1 F1: 0.XXXX"]
    E --> E2["Fold 2 F1: 0.XXXX"]
    E --> E3["Fold 3 F1: 0.XXXX"]
    E --> E4["Fold 4 F1: 0.XXXX"]
    E --> E5["Fold 5 F1: 0.XXXX"]
    E --> E6["Mean: 0.XXXX ± CI"]

    B --> F["08b: THRESHOLD OPT<br/>Find per-label threshold"]
    F --> G["Optimize for F1"]
    G --> G1["Depression: 0.41"]
    G --> G2["Anxiety: 0.37"]
    G --> G3["Stress: 0.38"]

    D1 --> H["✅ Robust Model"]
    E6 --> H
    G1 --> H

    style A fill:#ff9800
    style B fill:#ffb74d
    style C fill:#ffb74d
    style F fill:#ffb74d
    style H fill:#4caf50
    style D fill:#fff9c4
    style E fill:#fff9c4
    style G fill:#fff9c4
```

---

## Diagram 7: SHAP Explainability & Deployment

```mermaid
graph TD
    A["Trained Model"] --> B["10: SHAP ANALYSIS<br/>Feature Importance"]
    B --> C["Force Plots<br/>Per sample"]
    B --> D["Summary Plots<br/>Feature ranking"]
    B --> E["Dependence Plots<br/>Feature relationships"]

    C --> F["Understanding<br/>Individual predictions"]
    D --> G["Global Feature<br/>Importance"]
    E --> H["Feature Value<br/>Impact Analysis"]

    F --> I["Clinical Insights"]
    G --> I
    H --> I

    A --> J["11: DEPLOYMENT<br/>Package for production"]
    J --> K["Load Best Model<br/>Load Thresholds"]
    K --> L["Validation Check<br/>on test set"]
    L --> M["Create Prediction<br/>Pipeline"]
    M --> N["Save Artifacts"]
    N --> N1["multilabel_xgboost_mfo.pkl"]
    N --> N2["optimal_thresholds.json"]
    N --> N3["feature_names.json"]
    N --> N4["README_DEPLOYMENT.md"]

    N1 --> O["🚀 Production Ready<br/>Ready for deployment"]
    N2 --> O
    N3 --> O
    N4 --> O

    style A fill:#ff9800
    style B fill:#e1bee7
    style C fill:#ce93d8
    style D fill:#ce93d8
    style E fill:#ce93d8
    style J fill:#b3e5fc
    style K fill:#81d4fa
    style L fill:#81d4fa
    style M fill:#4fc3f7
    style N fill:#29b6f6
    style O fill:#0288d1
```

---

## Tabel Komparatif: Phase Outcomes

| Phase | Notebook | Input         | Output             | Key Metric          |
| ----- | -------- | ------------- | ------------------ | ------------------- |
| **1** | 01-02    | Raw data      | Statistics & plots | Data quality ✓      |
| **2** | 03-05    | Raw → Cleaned | Balanced training  | Imbalance ratio 1:1 |
| **3** | 06-06b   | 50+ features  | 31 features        | Reduction -38%      |
| **4** | 07       | Balanced data | Tuned model        | F1 macro ~0.XXX     |
| **5** | 07a-07g  | Tuned model   | Rankings           | Ablation proven     |
| **6** | 08-09    | Model         | Metrics & CV       | F1 mean ± CI        |
| **7** | 10-11    | Model         | Production         | Explainability ✓    |

---

## Design Decision Tree

```mermaid
graph TD
    A["Problem: Multi-label<br/>Mental Health<br/>Classification"] --> B{Multi-label<br/>Strategy?}
    B -->|Option 1| B1["Independent Classifiers<br/>Pros: Simple<br/>Cons: No dependencies"]
    B -->|Option 2| B2["ClassifierChain ✓<br/>Pros: Captures dependencies<br/>Cons: Error propagation"]
    B -->|Option 3| B3["Label Powerset<br/>Pros: Exact match good<br/>Cons: Complexity"]
    B2 --> C{Class Imbalance<br/>Handling?}
    C -->|Option 1| C1["Undersample only<br/>Lose information"]
    C -->|Option 2| C2["Oversample only<br/>Overfitting risk"]
    C -->|Option 3| C3["MLSMOTE ✓<br/>Synthetic + balance<br/>Multi-label aware"]
    C3 --> D{Feature Selection?}
    D -->|Option 1| D1["Domain knowledge<br/>Manual selection<br/>Slow"]
    D -->|Option 2| D2["Statistical tests<br/>Univariate<br/>Miss interactions"]
    D -->|Option 3| D3["B-MFO ✓<br/>Metaheuristic<br/>Catch interactions"]
    D3 --> E{Base Algorithm?}
    E -->|Option 1| E1["Random Forest<br/>Interpretable<br/>Moderate performance"]
    E -->|Option 2| E2["XGBoost ✓<br/>SOTA<br/>Good regularization"]
    E -->|Option 3| E3["CatBoost<br/>Categorical handling<br/>Good performance"]
    E2 --> F{Validation?}
    F -->|Option 1| F1["Train-test only<br/>Single estimate<br/>Biased"]
    F -->|Option 2| F2["5-fold CV ✓<br/>Multiple splits<br/>Robust"]
    F2 --> G["Final Decision: ✓<br/>ClassifierChain + MLSMOTE<br/>+ B-MFO + XGBoost<br/>+ 5-fold CV"]

    style B2 fill:#c8e6c9
    style C3 fill:#c8e6c9
    style D3 fill:#c8e6c9
    style E2 fill:#c8e6c9
    style F2 fill:#c8e6c9
    style G fill:#4caf50
```

---

## File I/O Summary

```
PROJECT STRUCTURE
│
├── 📥 INPUT
│   ├── Data/raw/Dataset.csv
│   └── notebooks/*.ipynb
│
├── 🔄 PROCESSING
│   ├── Data/processed/
│   │   ├── cleaned_data.csv (03)
│   │   ├── train_balanced_multilabel.csv (05)
│   │   └── train_selected_features.csv (06)
│   │
│   └── Data/split/
│       ├── train_data.csv (04)
│       └── test_data.csv (04)
│
├── 📊 OUTPUTS
│   ├── outputs/
│   │   ├── eda_figures/ (02)
│   │   ├── imbalance_figures/ (05)
│   │   ├── shap_figures/ (10)
│   │   ├── best_matrix/ (08)
│   │   ├── ablation_results_mfo_effect.csv (07a)
│   │   └── best_parameter/ (07)
│   │
│   └── notebooks/
│       ├── classification_report.txt (08)
│       ├── optimal_thresholds.txt (08b)
│       └── MFO_ABLATION_GUIDE.md
│
├── 🤖 MODELS
│   ├── mlruns/ (MLflow tracking)
│   └── models/
│       └── multilabel_xgboost_mfo.pkl (11)
│
└── 📋 DOCUMENTATION
    ├── README.md
    ├── METHODOLOGY.md (THIS!)
    ├── requirements.txt
    └── src/
        ├── mfo_optimizer.py
        ├── model_trainer.py
        └── preprocessing.py
```

---

**Diagram ini menyajikan visualisasi lengkap dari seluruh workflow dan arsitektur project.**
