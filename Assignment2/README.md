# CS772 - Assignment 2: Transliteration (Roman to Devanagari)

This project implements and compares three different sequence-to-sequence models—**LSTM-based Encoder-Decoder**, **Transformer-based Encoder-Decoder (Attention)**, and **Mistral Large Language Model (LLM)**—for the task of Roman-to-Devanagari transliteration.

## 👥 Authors
* **Kaustubh Shivshankar Shejole:** 24M2109
* **Shalaka Thorat:** 24M0848
* **Date:** 18/10/2025

## 🎯 Problem Statement
**Objective:** Given a sequence of text from one input script (Roman/Latin), convert it to another script (Devanagari/Hindi).

* **Task Type:** Character-level sequence-to-sequence problem.
* **Input Script:** Roman/Latin (e.g., `Mera dost kal Mumbai jaa raha hai`)
* **Output Script:** Devanagari (e.g., `मेरा दोस्त कल मुंबई जा रहा है`)

## 💾 Dataset & Preparation

**Dataset Used:** Aksharantar Corpus

**Composition:**
* **Training Set:** 100,000 instances
* **Validation Set:** Complete Validation Set (approx. 6,000 instances)
* **Testing Set:** Complete Test Set (approx. 10,000 instances)

**Sampling Strategy:**
A stratification strategy was employed with the intuition of creating a mini-replica of the full Aksharantar dataset to ensure representativeness across the training, validation, and testing splits.

## 💻 Methodology

The transliteration task was modeled using three distinct approaches:

### 1. LSTM-Based Transliteration
A standard **Encoder-Decoder architecture** utilizing Long Short-Term Memory (LSTM) cells was implemented.
* **Encoder:** Reads the input Roman characters sequence.
* **Decoder:** Generates the output Devanagari characters sequence one step at a time.
* The model captures sequential dependencies in the input and output character streams.

### 2. Transformer-Based Transliteration
A **Transformer architecture** featuring the self-attention mechanism was implemented.
* **Encoder-Decoder Structure:** Utilizes multiple layers of multi-head attention blocks.
* **Key Advantage:** The attention mechanism allows the model to weigh the relevance of every input character when generating each output character, leading to better long-range dependency capture and alignment.

### 3. LLM-Based Transliteration (Mistral)
The pre-trained **Mistral-7B-Instruct-v0.3** model was used for comparison, typically in a few-shot or zero-shot prompting configuration to test its inherent knowledge of transliteration rules learned during pre-training.

## 📊 Results and Comparison

While specific accuracy metrics (like Character Error Rate or Word Error Rate) were not provided in the summary, a qualitative comparison of the model outputs highlights key behavioral differences:

| Input (Roman) | LSTM Output | Transformer Output | Mistral LLM Output |
| :---: | :---: | :---: | :---: |
| `sourabh` | सोरभ | **सौरभ** | **सौरभ** |
| `kaustubh` | **कौस्तुभ** | **कौस्तुभ** | **कौस्तुभ** |
| `icci` | आईसीसीआई | आईसीसीआई | ईसीसीई |
| `nishant` | **निशांत** | निशंत | **निशांत** |
| `shalaaka` | **शालाका** | शलाका | श्लाका |
| `ichha` | इचा | **इच्छा** | **इच्छा** |
| `manisha` | मनीषा | मनीशा | मानिशा |
| `main` | मंन | मैन | **मैं** |
| `shreya` | श्रेय | **श्रेया** | **श्रेया** |

### General Analysis and Comparison

| Model | Strengths | Weaknesses/Behavior |
| :--- | :--- | :--- |
| **LSTM** | Captured the essence of mapping specific phonetic Roman letter groups (e.g., 'bh', 'sh') to their corresponding Hindi letters. | Can sometimes struggle with global context, leading to phonetically correct but orthographically incorrect output. |
| **Transformer** | Learned to generalize well; computationally efficient during training due to parallel processing. | Robust and accurate for most cases, excelling at capturing the complex, non-sequential dependencies of character alignment. |
| **Mistral LLM** | Performs well for common/frequent words due to vast prior knowledge encoded during pre-training. | Does not capture the underlying **character-level mappings** as effectively as dedicated sequence-to-sequence models; fails on highly novel or less frequent proper nouns. |

## 🧠 Challenges and Learnings

### Challenges Faced
* **Understanding Attention:** A core challenge was grasping and correctly implementing the full attention mechanism within the Transformer architecture.
* **Hyperparameter Tuning:** Finding the optimal set of hyperparameters for both the LSTM and Transformer models was computationally intensive.
* **Tooling:** Learning the correct handling and usage of tools like `wandb` (Weights and Biases) for tracking model training and hyperparameter optimization.

### Key Learnings
* **Attention Mechanism:** Gained a deep understanding of the self-attention mechanism, which is central to modern NLP architectures.
* **Encoder-Decoder Structure:** The assignment clarified the practical application and importance of the Encoder-Decoder framework in sequence-to-sequence problems.
* **Error Analysis:** Understood how output errors manifest in a transliteration setting (e.g., subtle differences in *matras* or specific character choices).
* **Task Difficulty:** Realized that transliteration, despite appearing simple, is a difficult task due to ambiguities in phonetic mapping and the need to maintain orthographic conventions.



## 🔗 PPT Link
Access the presentation at this [link](https://docs.google.com/presentation/d/1Xk2MkffWfL9ED9miP-0vNSHGrmeFmNYK1MkfElNcYIc/edit?usp=sharing)


## 🌐 Demo

A live demonstration of the project, showcasing the best-performing models, has been hosted on Huggingface Spaces.

1. [LSTM Transliteration](https://huggingface.co/spaces/Kaustubh12345/CS772_Transliteration_lstm_Kaustubh)  
2. [Transformer Transliteration](https://huggingface.co/spaces/thenlpresearcher/CS772_Transliteration_Transformer_Shalaka_Kaustubh)


## 🔗 References

* **Corpus:** Aksharantar Corpus [https://ai4bharat.org/](https://ai4bharat.org/)
* **Textbook:** Pushpak Bhattacharyya and Aditya Madhav Joshi, *Natural Language Processing*, Wiley India, 2023.
* **NLP Theory:** [https://web.stanford.edu/~jurafsky/slp3/13.pdf](https://web.stanford.edu/~jurafsky/slp3/13.pdf) (Chapter 13 of Jurafsky’s Natural Language Processing book).
* **GUI Tools:** [https://www.gradio.app/](https://www.gradio.app/), [https://streamlit.io/](https://streamlit.io/)
