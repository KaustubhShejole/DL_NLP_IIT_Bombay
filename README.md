## CS772: Deep Learning for Natural Language Processing - Course Projects

This repository contains two major project assignments completed for the **CS772: Deep Learning for NLP** course at **IIT Bombay**. The work explores various modeling techniques, from traditional statistical methods to modern deep learning architectures and Large Language Models (LLMs), for core sequence prediction problems.

---

### 🎓 Course and Authors

* **Course:** CS772 - Deep Learning for Natural Language Processing
* **Institution:** IIT Bombay
* **Authors:**
    * Kaustubh Shivshankar Shejole (24M2109)

---

### 📚 Project Summary

The assignments focus on implementing, training, and rigorously comparing different models to solve fundamental NLP tasks. Please refer to readme files and ppts in respective folders for more details. 

| Assignment | Topic | Goal | Models Implemented | Key Concepts Highlighted |
| :--- | :--- | :--- | :--- | :--- |
| **Assignment 1** | Part-of-Speech (POS) Tagging | Assign the correct grammatical tag to each word in a sequence. | HMM (Viterbi), LSTM, LLMs (Mistral/GPT) | Dynamic Programming, Contextual Modeling, Unseen Word Handling (Laplace Smoothing). |
| **Assignment 2** | Transliteration | Convert text from **Roman (Latin)** script to **Devanagari (Hindi)** script. | LSTM Encoder-Decoder, Transformer (Attention), LLM (Mistral) | Character-level Sequence-to-Sequence, Self-Attention, Phonetic-Orthographic Ambiguity. |

---

### 📂 Repository Structure

```
.
├── aksharantar_sampled/
│   ├── asm/
│   ├── ben/
│   ├── brx/
│   ├── guj/
│   ├── hin/
│   ├── kan/
│   ├── kas/
│   ├── kok/
│   ├── mai/
│   ├── mal/
│   ├── mar/
│   ├── mni/
│   ├── ori/
│   ├── pan/
│   ├── san/
│   ├── sid/
│   ├── tam/
│   ├── tel/
│   └── urd/
├── aks_dataset/
│   └── hin/
├── Assignment1/
│   ├── comparison_with_gpt5/
│   ├── final_hmm/
│   ├── final_llm/
│   ├── final_lstm/
│   └── lstm2/
└── Assignment2/
    ├── dataset/
    ├── final_llm/
    │   └── .gradio/
    └── notebooks/
        ├── models/
        └── predictions/

```
---

### ⭐ Comparative Analysis & Key Learnings

The projects provided a strong comparative view of modeling techniques:

| Comparison | Observation | Conclusion |
| :--- | :--- | :--- |
| **HMM vs. LSTM** | LSTMs (e.g., **96% accuracy** on POS) consistently outperformed HMMs (e.g., **95%**) because they capture **long-range contextual dependencies**, while HMMs are limited to the Markov assumption (local context). | **Deep Learning models** are superior for tasks requiring sophisticated context modeling. |
| **LSTM vs. Transformer** | **Transformers** excel in tasks requiring complex alignment and non-sequential dependencies (like Transliteration) due to the **Self-Attention mechanism**, which processes all tokens in parallel. | **Attention-based architectures** provide a powerful, generalized framework for complex sequence-to-sequence problems. |
| **Traditional DL vs. LLMs** | **LLMs** (like GPT-5-mini in POS Tagging) demonstrate superior zero/few-shot performance and overall generalization (e.g., **94% accuracy**). However, they sometimes fail on highly specific, rule-based character mappings (Transliteration). | **LLMs are state-of-the-art** for semantic and general NLP tasks, but dedicated **Seq2Seq models are essential** for explicit, character-level transformations. |

---

### 🔗 Common References

* **Textbook:** Pushpak Bhattacharyya and Aditya Madhav Joshi, *Natural Language Processing*.
* **Theory:** Jurafsky & Martin, *Speech and Language Processing*.
* **Corpora:** NLTK (Brown, Penn Treebank), Aksharantar Corpus.
