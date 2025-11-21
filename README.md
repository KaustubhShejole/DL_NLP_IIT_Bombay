# CS772 — Natural Language Processing  
### Authors: Kaustubh Shivshankar Shejole (24M2109) and Shalaka Thorat (24M0848)
---

## Assignments & Project

This repository contains solutions, code, analyses, and reports for the CS772 coursework at IIT Bombay. It includes:

- **Assignment 1:** POS Tagging  
- **Assignment 2:** Transliteration  
- **NLP Project:** English → Marathi Machine Translation with a focus on punctuation sensitivity  

Problem statements (local copies):

- [Assignment 1 — POS Tagging (PDF)](/mnt/data/CS772_2025_Assignment-1_Problem_Statement .pdf)  
- [Assignment 2 — Transliteration (PDF)](/mnt/data/CS772 Assignment 2 (1).pdf)  
- [NLP Project Task (PDF)](/mnt/data/NLP Project Task.pdf)

---

## Repository Structure

├── Assignment_1_POS_Tagger/
│ ├── data/
│ ├── notebooks/
│ ├── code/
│ │ ├── hmm_pos.py
│ │ ├── rnn_pos.py
│ │ └── llm_prompting_pos.py
│ ├── models/
│ ├── results/
│ └── README.md
│
├── Assignment_2_Transliteration/
│ ├── data/
│ ├── code/
│ │ ├── lstm_translit.py
│ │ ├── transformer_translit.py
│ │ └── llm_translit_prompt.py
│ ├── models/
│ ├── evaluation/
│ │ ├── eval_word_accuracy.py
│ │ └── eval_char_f1.py
│ └── README.md
│
├── NLP_Project_MT/
│ ├── data/
│ ├── code/
│ │ ├── prepare_punc_data.py
│ │ ├── punc_tagging_model.py
│ │ └── mt_finetune.py
│ ├── baselines/
│ ├── results/
│ └── README.md
│
├── requirements.txt
└── README.md




---

# Assignment 1 — A Comparative Study of POS Tagging Models

**Problem statement:** [Assignment 1 PDF](/mnt/data/CS772_2025_Assignment-1_Problem_Statement .pdf)

### Objective  
Build and compare three POS-tagging approaches using a common dataset and unified evaluation scheme:

1. **Hidden Markov Model (HMM)**  
2. **RNN-based Encoder–Decoder** (vanilla RNN/GRU/LSTM)  
3. **LLM-based prompting** (zero/few-shot)

### Dataset  
- **Brown corpus** (NLTK)  
- Universal POS tagset  
- Reproducible preprocessing + fixed train/val/test splits

### Implementation Checklist  
- `data/preprocess_brown.py` — tokenization, universal tagset mapping, splits  
- `code/hmm_pos.py` — transitions, emissions, smoothing, Viterbi  
- `code/rnn_pos.py` — embeddings, encoder, classifier head, training loop  
- `code/llm_prompting_pos.py` — prompting templates, evaluation harness  
- `notebooks/analysis.ipynb` — confusion matrices, tag-wise metrics

### Evaluation  
- Accuracy, Precision, Recall, F1  
- Confusion matrix (visualized)  
- Per-tag breakdown  
- Runtime + model-size comparison

### Deliverables  
- Training + evaluation scripts  
- `results/` with metrics and confusion matrices  
- Short report covering architectural choices + analysis

---

# Assignment 2 — Transliteration (Hindi Roman → Devanagari)

**Problem statement:** [Assignment 2 PDF](/mnt/data/CS772 Assignment 2 (1).pdf)

### Objective  
Train and compare transliteration systems for Romanized Hindi → Devanagari using:

1. **LSTM Seq2Seq** (≤2 layers)  
2. **Transformer Seq2Seq** (≤2 layers; include variant with local attention)  
3. **LLM prompting** (temperature / top_p sweeps)

### Dataset  
- **Aksharantar** (AI4Bharat, HuggingFace)  
- ≤ 100k training samples (documented sub-sampling)  
- Use full test set

### Implementation Checklist  
- `data/download_aksharantar.py` — download, tokenization  
- `data/subsample.py` — reproducible sampling logic  
- `code/lstm_translit.py` — encoder–decoder with attention  
- `code/transformer_translit.py` — minimal transformer, 2 layers  
- `code/llm_translit_prompt.py` — prompt templates + evaluation harness  
- `evaluation/eval_word_accuracy.py` — word accuracy  
- `evaluation/eval_char_f1.py` — character-level F1

### Evaluation & Analysis  
- Word-level exact match accuracy  
- Character-level F1  
- Greedy vs beam search (3, 5)  
- Error analysis: confused character sequences  
- Ablations: data size, depth, decoding strategy

### Deliverables  
- Model checkpoints or logs  
- Evaluation scripts + outputs  
- Short method comparison write-up  
- Optional demo script / minimal GUI

---

# NLP Project — English → Marathi MT (Punctuation Sensitivity)

**Problem statement:** [NLP Project Task PDF](/mnt/data/NLP Project Task.pdf)

### Objective  
Explore how punctuation affects machine translation quality for English → Marathi, create a punctuation-sensitive evaluation set, benchmark multiple systems, and test mitigation strategies.

### Components  

#### Dataset & Test Creation
- 50–75 English sentences where punctuation influences meaning  
- Gold Marathi references created by taking Gemini, CFILT, IndicTrans2 outputs and manually post-editing

#### Baselines
- Gemini / GPT models  
- CFILT IITB MT  
- IndicTrans2 variants (600M, 1B)

#### Modeling Strategies
- Punctuation tagging model (`_PUNC_COMMA`, etc.)  
- English → English-with-tags → MT pipeline  
- Fine-tuning IndicTrans2 on mixed punctuated/unpunctuated data  
- Optional: multi-task punctuation + translation model

#### Evaluation
- BLEU  
- CharF++  
- Indic COMET  
- BLEURT  
- COMET  
- Qualitative case studies where punctuation shifts the meaning in translation

### Workflow (Example)
- `code/prepare_punc_data.py` — training pairs  
- `baselines/collect_baseline_outputs.py` — script to gather system outputs  
- `code/punc_tagging_model.py` — token classifier  
- `code/mt_finetune.py` — fine-tune IndicTrans2  
- `results/` — metrics, human post-edits, analyses

### Deliverables
- Curated punctuation-sensitive test set (variants included)  
- Evaluation scripts + final results  
- Report with findings, mitigation strategies, and examples

---

## Requirements
All dependencies are listed in `requirements.txt`.

---

## Maintainer  
This repository contains coursework completed as part of **CS772 — Natural Language Processing**.

