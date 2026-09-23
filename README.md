# IndoShield — Indonesian Toxic Language Detection

IndoShield is a Natural Language Processing project that detects and grades abusive and hateful language in Indonesian text. Instead of a simple "toxic / not toxic" flag, it classifies a piece of text into one of four severity levels — from clean to strongly abusive — using a combination of classical machine learning models (Naive Bayes, Logistic Regression, SVM) and a fine-tuned Indonesian transformer model (IndoBERTweet).

The project was built to handle the way abusive language actually shows up on Indonesian social media: informal spelling, spaced-out letters (`a n j i n g`), leetspeak substitutions (`b4b1`, `t0l0l`), and regional slang. It includes a full preprocessing and training pipeline, four trained models served side by side, and a Streamlit web app where anyone can test the models on their own text.

This project was developed as a team NLP/Machine Learning coursework project focused on the Indonesian language.

---

## Live Demo

🚀 **Try the application:** [IndoShield on Hugging Face Spaces](https://huggingface.co/spaces/britod/swear-words-detector)

🎥 **Demo recording:** [Watch the Demo (Google Drive)](https://drive.google.com/drive/folders/17NRBhcIx0w8HfKoxa9bIJOY93ZJ0E-SC?usp=sharing)

---

## Dataset

IndoShield is trained on the **Indonesian Multi-label Hate Speech and Abusive Language Dataset** by Ibrohim & Budi (2019):

🔗 [github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection)

The original dataset is **multi-label**: each tweet is annotated with several binary columns (`HS`, `Abusive`, `HS_Individual`, `HS_Group`, `HS_Religion`, `HS_Race`, `HS_Physical`, `HS_Gender`, `HS_Other`, `HS_Weak`, `HS_Moderate`, `HS_Strong`). IndoShield **does not use the dataset as-is**. It collapses these labels into a single **ordinal 4-class severity label** using the following rule (`pipeline/libs/data_loader.py`):

| Condition (checked in order) | Resulting Level |
| --- | --- |
| `HS == 0` and `Abusive == 0` | Level 0 |
| `HS_Strong == 1` | Level 3 |
| `HS_Moderate == 1` | Level 2 |
| Otherwise (`HS_Weak == 1` or `Abusive == 1`) | Level 1 |

This means the original target/category/group information (who the hate speech is directed at, e.g. religion, race, gender) is **discarded** — only the severity dimension is kept.

Two auxiliary files from the same dataset source are also used:

| File | Role |
| --- | --- |
| `dataset/abusive.csv` | 125-word Indonesian abusive-word lexicon, used by the app to highlight/censor matched words |
| `dataset/new_kamusalay.csv` | 15,166-entry Indonesian slang-to-formal dictionary, used to normalize slang during preprocessing |

**Preprocessing before training** (`pipeline/preprocess_pipeline.py`): lowercasing, collapsing spaced-out letters, stripping stray punctuation, leetspeak substitution, and alphabetic filtering. For the classical models the text is additionally tokenized, slang-normalized against the *kamusalay* dictionary, and stemmed with Sastrawi. The IndoBERTweet branch skips tokenization/stemming and uses its own tokenizer instead.

**Split:** the data is divided **80% train / 10% validation / 10% test**, stratified by label, with `random_state=42` (`pipeline/libs/data_loader.py::make_splits`).

**Verified sample counts** (from `logs/classical_training.log`, after dropping empty rows during preprocessing):

| Split | Samples |
| --- | ------: |
| Train | 10,535 |
| Validation | 1,317 |
| Test | 1,317 |
| **Total** | **13,169** |

---

## Results

Metrics below come directly from `logs/classical_training.log`, evaluated on the held-out 10% test set (1,317 examples), for the last complete training run of the three classical models (Naive Bayes, Logistic Regression, SVM). All values are **macro-averaged** across the four severity levels, which matters here because the classes are heavily imbalanced (Level 0/1 make up the large majority of examples, Level 3 is a small minority).

| Model | Accuracy | Precision (macro) | Recall (macro) | F1-score (macro) |
| --- | -------: | -----------------: | --------------: | -----------------: |
| Logistic Regression | 0.78 | 0.72 | 0.76 | 0.74 |
| SVM (LinearSVC, calibrated) | 0.79 | 0.76 | 0.70 | 0.72 |
| Naive Bayes (ComplementNB) | 0.74 | 0.66 | 0.64 | 0.64 |

**What this means:** Logistic Regression and SVM perform closely and clearly outperform Naive Bayes. Looking at the per-class breakdown in the same log, all three models do well on Level 0 (clean) and Level 1 (mildly abusive), but **struggle on Level 2 (moderately abusive)** — recall drops to 0.33–0.60 depending on the model — likely because the boundary between "mild" and "moderate" abuse in the source annotations is itself subjective. Level 3 (strongly abusive) scores are based on only 47 test examples, so those numbers should be read with some caution.

**IndoBERTweet:** the app also serves a fine-tuned `indolem/indobertweet-base-uncased` model (see `pipeline/train_bert.py`), and `pipeline/evaluate.py` is designed to report its precision/recall/F1 alongside the classical models on the same test split. However, **no committed log file in this repository contains IndoBERTweet's test-set numbers**, so they are intentionally left out of the table above rather than estimated. Running `python -m pipeline.evaluate` after training the model will produce them.

No confusion-matrix image was found committed in the repository (`pipeline/evaluate.py` generates one at `outputs/confusion_matrices.png`, but that path is git-ignored).

---

## What It Does

1. The user types (or selects a sample) Indonesian text into the Streamlit app.
2. The app runs the shared preprocessing pipeline, producing two versions of the text: a stemmed/slang-normalized version for the classical models, and a lightly cleaned version for IndoBERTweet — matching exactly what each model saw during training.
3. All **four models run in parallel**: Naive Bayes, Logistic Regression, SVM, and IndoBERTweet each independently predict a severity level (0–3) along with per-class probabilities.
4. The app also checks the raw text against the abusive-word lexicon (`abusive.csv`) and produces a censored version of the input, independently of the model predictions.
5. Results are displayed side by side, along with a **majority-vote consensus** across the four models and a full preprocessing trace so the user can see how the text was transformed at each step.

---

## Example

Input:

> "Dasar bego lo, ga bisa ngapa-ngapain!"
> *(translation: "You're such an idiot, you can't do anything!")*

Preprocessing:

| Stage | Result |
| --- | --- |
| Lowercase | `dasar bego lo, ga bisa ngapa-ngapain!` |
| Cleaned (IndoBERTweet input) | `dasar bego lo ga bisa ngapa ngapain` |
| Slang-normalized + stemmed (classical input) | `dasar bego kamu enggak bisa ngapa ngapain` |

Prediction:

> The app shows one predicted level per model (Naive Bayes, Logistic Regression, SVM, IndoBERTweet), each with a class-probability bar for all four levels, plus a consensus label from majority voting.

Confidence:

> The app displays per-class confidence percentages from each model's `predict_proba` output. Actual values depend on the trained model weights (which are not shipped in this repository — see [Setup](#setup)), so no specific numbers are quoted here. Run the [live demo](https://huggingface.co/spaces/britod/swear-words-detector) to see real predictions and confidence scores.

---

## Levels / Classification Categories

IndoShield performs **4-class classification** (not binary). Each input text is assigned exactly one of the following levels, defined in `app/predictor.py::LEVEL_LABELS`:

| Level | Indonesian Label | English Label | Meaning |
| :---: | --- | --- | --- |
| 0 | Bersih | Clean | No hate speech or abusive content detected |
| 1 | Kasar Ringan | Mildly Abusive | Weak hate speech or general abusive language |
| 2 | Kasar Sedang | Moderately Abusive | Moderate-strength hate speech |
| 3 | Kasar Berat | Strongly Abusive | Strong/severe hate speech |

---

## Data Overview

| Property | Details |
| --- | --- |
| Dataset | Indonesian Multi-label Hate Speech and Abusive Language Dataset (Ibrohim & Budi, 2019) |
| Language | Indonesian (informal Twitter text) |
| Task | 4-class ordinal severity classification, derived from a multi-label dataset |
| Number of Samples | 13,169 tweets (after preprocessing) |
| Number of Classes | 4 (Level 0–3) |
| Input | Raw Indonesian tweet text |
| Target | `Level Kata Kasar` — an integer 0–3 derived from the original `HS`/`Abusive`/`HS_Weak`/`HS_Moderate`/`HS_Strong` columns |
| Train Split | 10,535 samples (80%) |
| Validation Split | 1,317 samples (10%) |
| Test Split | 1,317 samples (10%) |
| Random Seed | 42 (stratified split) |

---

## How It Works

```text
Raw Indonesian Text
        ↓
Text Preprocessing (lowercase → de-space letters → clean punctuation → leetspeak substitution → keep alphabetic only)
        ↓
        ├── Classical branch: tokenize → slang normalization (kamusalay) → Sastrawi stemming → TF-IDF (char n-grams, 2–5)
        └── Transformer branch: cleaned text → IndoBERTweet tokenizer
        ↓
Models: ComplementNB · Logistic Regression · Calibrated LinearSVC (classical) · fine-tuned IndoBERTweet (transformer)
        ↓
Predicted Severity Level (0–3) + class probabilities per model
        ↓
App displays per-model results, majority-vote consensus, and lexicon-based censoring
```

Techniques actually used in the code, confirmed by inspection:

| Technique | Used? | Where |
| --- | :---: | --- |
| Lowercasing, punctuation cleanup | ✅ | `pipeline/preprocess_pipeline.py` |
| Leetspeak substitution (`4→a`, `0→o`, etc.) | ✅ | `pipeline/preprocess_pipeline.py` |
| Slang normalization (kamusalay dictionary) | ✅ | `pipeline/preprocess_pipeline.py::normalize_slang` |
| Stemming (Sastrawi) | ✅ | classical models only |
| Stopword removal | ⚠️ Loaded but not applied to the model input pipeline (`stopword_list` is defined in `preprocess_pipeline.py` but not used in `preprocess()`) |
| TF-IDF (character n-grams, 2–5, 50k features) | ✅ | Naive Bayes, Logistic Regression, SVM |
| Bag-of-Words | ❌ Not used | — |
| Word embeddings (word2vec/GloVe) | ❌ Not used | — |
| Transformer tokenization | ✅ | IndoBERTweet branch |
| IndoBERT / IndoBERTweet fine-tuning | ✅ | `pipeline/train_bert.py` (`indolem/indobertweet-base-uncased`) |
| Naive Bayes | ✅ | `ComplementNB` |
| Logistic Regression | ✅ | `class_weight='balanced'` |
| SVM | ✅ | `LinearSVC` wrapped in `CalibratedClassifierCV` for probability estimates |
| Hyperparameter tuning | ✅ | Optuna, cross-validated macro-F1 for classical models; validation macro-F1 with pruning for IndoBERTweet |
| Class imbalance handling | ✅ | `class_weight='balanced'` for classical models (shipped version); random oversampling for IndoBERTweet training; SMOTE / SMOTEENN / undersampling variants exist as **comparison experiments** in `pipeline/alternatives/`, not the shipped models |

---

## Limitations

- **Level 2 (moderately abusive) is the weakest point.** Recall for this class ranges from 0.33 (Naive Bayes) to 0.60 (Logistic Regression) across the classical models. Moderate hate speech is frequently misclassified as mild.
- **Class imbalance.** Level 3 (strongly abusive) makes up a small fraction of the dataset (47 of 1,317 test examples), so its scores are based on a small sample and can shift noticeably with small prediction changes.
- **No context or sarcasm handling.** Each model scores a single piece of text in isolation, with no surrounding conversation, target information, or intent — sarcasm, quoted speech, and reclaimed language can all be misread as abusive.
- **Dataset is Twitter-specific and dated.** The training data comes from Indonesian Twitter around 2018–2019, so slang, spelling patterns, and platform norms may differ from more recent social media text.
- **The censoring feature is simpler than the classifiers.** It performs exact string matching against a 125-word lexicon on the *raw* input text, so it does not catch leetspeak or obfuscated spellings the way the trained models do.
- **IndoBERTweet's own test performance is not verifiable from this repository.** No committed log contains its precision/recall/F1, so any comparison between it and the classical models cannot currently be confirmed from the codebase alone.

---

## Ethical Use

IndoShield is an academic NLP/machine learning project, built to explore automated detection of abusive language in Indonesian text — not a production moderation system.

- Predictions are **probabilistic classifications**, not absolute judgments about a person or their intent.
- Context matters: sarcasm, reclaimed slurs, and jokes between friends can be misclassified as abusive, and genuinely harmful language can be missed.
- False positives and false negatives are expected, especially for the moderate-severity class (see [Limitations](#limitations)).
- The model should **not** be used as the sole basis for decisions that affect real people — such as banning, reporting, or profiling individuals — without human review.
- The training data may carry its own biases (in topic coverage, annotator judgment, and time period), which can propagate into the model's predictions.
- This project is best used to **support** human moderation or research, not to replace human judgment.

---

## Project Structure

```text
IndoShield/
├── app/
│   ├── app.py                  # Streamlit UI — text input, model predictions, censoring
│   ├── predictor.py            # Loads the 4 trained models and runs inference
│   └── preprocessing.py        # Serving-time preprocessing (mirrors the training pipeline)
├── pipeline/
│   ├── preprocess_pipeline.py  # 7-step text cleaning/normalization pipeline
│   ├── train_classical.py      # Trains Naive Bayes, Logistic Regression, SVM (shipped models)
│   ├── train_bert.py           # Fine-tunes IndoBERTweet
│   ├── evaluate.py             # Generates classification reports, confusion matrices, comparison plots
│   ├── alternatives/           # Resampling-strategy comparison experiments (SMOTE, undersampling, etc.)
│   └── libs/
│       ├── config.py           # File paths, hyperparameters, random seed
│       ├── data_loader.py      # Label mapping (map_to_level) and train/val/test splitting
│       └── lexicon.py          # Standalone fuzzy-matching lexicon utility (not used by the app)
├── dataset/
│   ├── data.csv                 # Source multi-label dataset (Ibrohim & Budi, 2019)
│   ├── abusive.csv              # 125-word abusive-word lexicon
│   └── new_kamusalay.csv        # 15,166-entry Indonesian slang dictionary
├── dataset_processed.csv       # Preprocessed dataset produced by preprocess_pipeline.py
├── logs/                       # Training logs (Optuna trials + classification reports)
├── reports/                    # Internal engineering audit notes
├── deploy/upload_models.py     # Uploads trained models to a Hugging Face Hub model repo
├── requirements.txt            # Runtime dependencies for the Streamlit app
└── LICENSE                     # MIT license (project code)
```

> Note: `saved_models/` (trained model weights, ~436 MB) and `outputs/` (generated plots) are excluded via `.gitignore` and are not present in this repository. They must be generated by running the training scripts, or downloaded from a Hugging Face Hub model repo at runtime (see [Setup](#setup)).

---

## Setup

**Requirements:** Python 3.11 (the deployed Hugging Face Space is pinned to this version; dependency wheels in `requirements.txt` are pinned to match).

```bash
git clone <repository-url>
cd IndoShield

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` only covers what the **Streamlit app** needs to run (it doubles as the Hugging Face Space's dependency manifest). To run the **training pipeline**, additional packages that are imported in the code but not listed there are also required:

```bash
pip install optuna matplotlib seaborn imbalanced-learn rapidfuzz
```

- `optuna` — hyperparameter tuning (`pipeline/libs/_train_core.py`, `pipeline/train_bert.py`)
- `matplotlib`, `seaborn` — plots in `pipeline/evaluate.py` and `pipeline/train_bert.py`
- `imbalanced-learn` — resampling experiments in `pipeline/alternatives/`
- `rapidfuzz` — only needed if running `pipeline/libs/lexicon.py` directly (not used by the app)

**Trained model weights are not included in this repository.** To run the app locally you need to either:
1. Train the models yourself (see [Usage](#usage)), which will populate `saved_models/`, or
2. Set the `MODEL_REPO_ID` environment variable to a Hugging Face Hub model repo containing the trained artifacts — `app/predictor.py` will download them automatically via `huggingface_hub.snapshot_download`.

---

## Usage

**1. Preprocess the dataset** (produces `dataset_processed.csv`, already committed in this repo):

```bash
python -m pipeline.preprocess_pipeline
```

**2. Train the classical models** (Naive Bayes, Logistic Regression, SVM):

```bash
python -m pipeline.train_classical
```

**3. Fine-tune IndoBERTweet:**

```bash
python -m pipeline.train_bert
```

**4. Evaluate all trained models** (classification reports, confusion matrices, comparison plots):

```bash
python -m pipeline.evaluate
```

**5. Run the Streamlit app locally:**

```bash
streamlit run app/app.py
```

**Notes:**
- Run pipeline scripts as modules (`python -m pipeline.…`) from the repository root — they rely on package-relative imports (`pipeline.libs.config`, etc.) that break if run as plain scripts from inside the `pipeline/` folder.
- `FORCE_RETRAIN = True` by default in `pipeline/libs/config.py`, so re-running the training scripts will overwrite any existing saved models.
- On Hugging Face Spaces, setting `MODEL_REPO_ID` makes the app pull model weights from the Hub instead of expecting a local `saved_models/` directory (`deploy/upload_models.py` is the corresponding upload script).

---

## Reproducibility

| Aspect | Status |
| --- | --- |
| Random seed | `RANDOM_STATE = 42` in `pipeline/libs/config.py`, used for the train/val/test split and the Optuna sampler |
| Train/test split | 80/10/10, stratified by label |
| Dependencies | Pinned in `requirements.txt` for the app; training-only dependencies are unpinned (see [Setup](#setup)) |
| Dataset source | Publicly available (Ibrohim & Budi, 2019) — linked above |
| Classical model configuration | TF-IDF (`char_wb`, n-grams 2–5, 50,000 features, `min_df=2`), Optuna-tuned hyperparameters logged in `logs/classical_training.log` |
| IndoBERTweet configuration | `indolem/indobertweet-base-uncased`, max length 128, Optuna-tuned learning rate/weight decay/warmup/batch size, 10 final training epochs |
| Trained weights | **Not committed** to the repository (`saved_models/` is git-ignored, ~436 MB). Must be regenerated via the training scripts, or fetched from a Hugging Face Hub model repo at runtime |
| Classical model metrics | Reproducible and already logged in `logs/classical_training.log` |
| IndoBERTweet metrics | **Not currently reproducible from committed artifacts** — no test-set results for it are saved in this repository; running `pipeline/evaluate.py` after training will generate them |

Because IndoBERTweet fine-tuning uses non-deterministic GPU kernels, exact metric reproduction for that model may vary slightly between runs even with the same seed.

---

## References

- Ibrohim, M. O., & Budi, I. (2019). *Multi-label Hate Speech and Abusive Language Detection in Indonesian Twitter.* Proceedings of the 3rd Workshop on Abusive Language Online (ALW3), pages 46–57. [Dataset repository](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection)
- Koto, F., Lau, J. H., & Baldwin, T. (2021). *IndoBERTweet: A Pretrained Language Model for Indonesian Twitter with Effective Domain-Specific Vocabulary Initialization.* EMNLP 2021. Model used: [`indolem/indobertweet-base-uncased`](https://huggingface.co/indolem/indobertweet-base-uncased)
- [scikit-learn](https://scikit-learn.org/) — TF-IDF vectorization, Logistic Regression, Naive Bayes, SVM, and model calibration
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) — IndoBERTweet fine-tuning and inference
- [Sastrawi](https://github.com/har07/PySastrawi) — Indonesian stemming
- [Optuna](https://optuna.org/) — hyperparameter tuning
- [Streamlit](https://streamlit.io/) — web application framework

---

## License

This repository's code is licensed under the **MIT License** — see [`LICENSE`](LICENSE).

The dataset, abusive-word lexicon, and slang dictionary in `dataset/` originate from Ibrohim & Budi's dataset repository and are governed by that repository's own terms, not by this project's MIT license. Refer to the [original dataset repository](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection) for its specific licensing terms. The `indolem/indobertweet-base-uncased` model carries its own license on the Hugging Face Hub — check its model card before commercial use.

---

## Team Members

| Student ID | Name |
| --- | --- |
| 2802403183 | Brian Nicholas Tedjo |
| 2802419446 | Jason Budiharjo |
| 2802402275 | Marvin Adriano Rusdianto |
