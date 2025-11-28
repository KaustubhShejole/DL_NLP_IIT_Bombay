# CS772 - Assignment 1: Part-of-Speech (POS) Tagging

This project implements and compares three different modeling approaches—Hidden Markov Model (HMM) with Viterbi Decoding, Long Short-Term Memory (LSTM) Networks, and Large Language Models (LLMs)—for the task of Part-of-Speech (POS) tagging on the Brown and Penn Treebank corpora.

## 👥 Authors
* **Kaustubh Shivshankar Shejole:** 24M2109
* **Shalaka Thorat:** 24M0848
* **Date:** 30/08/2025

## 🎯 Problem Statement
**Objective:** Given an input sequence of words, produce the corresponding sequence of Part-of-Speech tags.

**Example:**
* **Input:** The quick brown fox jumps over the lazy dog
* **Output:** The**DET** quick**ADJ** brown**ADJ** fox**NOUN** jumps**VERB** over**ADP** the**DET** lazy**ADJ** dog**NOUN**

**Tag Set:** Universal Tag Set (12 tags)
`[ADJ, ADP, ADV, CONJ, DET, NOUN, NUM, PRON, PRT, VERB, ., X]`

## 💾 Data & Preparation

**Datasets Used:**
1.  Complete Brown corpus
2.  Complete Penn Treebank corpus

**Source:** `nltk.download()`

**Cleaning:**
* Words were converted to lowercase specifically for the HMM model.
* Given the standard and large size of both datasets, no manual cleaning was performed.
* POS tags were used as provided in the corpora.

**Data Split:**
* **Training Data:** 80% of the Brown corpus.
* **Testing Data:**
    1.  20% of the Brown corpus.
    2.  Complete Penn Treebank corpus.

## 💻 Methodology

### 1. HMM-Based POS Tagging
The **Viterbi Algorithm** was used for decoding in the HMM.

* **Viterbi Complexity:** The algorithm has a linear time complexity of $O(|T| \cdot N)$, where $|T|$ is the number of tags and $N$ is the number of words.
* **Reason for Linear Time:** This efficiency is achieved through **Pruning** and the **Markov Assumption**. Among all paths ending with the same tag, only the most probable one is retained, as others can never lead to the globally optimal sequence.


[Image of HMM Viterbi trellis diagram]


### 2. LSTM-Based POS Tagging
A neural network approach using Long Short-Term Memory (LSTM) cells was implemented.

* **Architecture:** A standard LSTM architecture processes the word sequence, $X_1$ to $X_N$, producing the tag sequence, $Y_1$ to $Y_N$. Loss is backpropagated using **Backpropagation Through Time (BPTT)**.
* **Embeddings Used:** `sentence-transformers/all-MiniLM-L6-v2`.
* **Implementation Note:** The custom implementation achieved 96.13% accuracy on the Brown test set, comparable to the original PyTorch LSTM implementation's 96.33%.


### 3. LLM-Based POS Tagging
Large Language Models were utilized to explore their capabilities in zero-shot or few-shot POS tagging, especially for handling rare or ambiguous words.

* **Models Used:**
    * `mistralai/Mistral-7B-Instruct-v0.3` (Tested on the complete Penn Treebank corpus).
    * `GPT-5-mini` (default ChatGPT) (Tested on 100 random sentences from the Penn Treebank).

## 📊 Results and Evaluation

### Performance Comparison (Accuracy)

| Model | Test Set | Accuracy |
| :--- | :--- | :--- |
| **LSTM** | Brown (20%) | **0.96** |
| HMM | Brown (20%) | 0.95 |
| **LSTM** | Penn Treebank | **0.84** |
| **Mistral** | Penn Treebank | **0.84** |
| HMM | Penn Treebank | 0.81 |
| **ChatGPT** | Penn Treebank (100 sentences) | **0.94** |

### Detailed Metric Comparison

#### HMM vs. LSTM (Brown Corpus)
| Tag | Best Model | Observations |
| :--- | :--- | :--- |
| PUNCT, DET, X | BOTH | High accuracy across both models. |
| ADJ, CONJ, NOUN, NUM, PRON, VERB | **LSTM** | LSTM performs better on most frequent and context-sensitive tags. |
| ADP, PRT | **HMM** | HMM excels on specific functional tags. |
| **Overall** | **LSTM (0.96)** | LSTM demonstrates stronger contextual modeling capabilities. |

#### HMM vs. LSTM vs. Mistral (Penn Treebank Corpus)
| Tag | Best Model | Observations |
| :--- | :--- | :--- |
| PUNCT, VERB, X | **Mistral** | Mistral handles these effectively. |
| ADJ, CONJ, NOUN, NUM | **LSTM** | LSTM is superior on these context-sensitive tags. |
| ADP, DET, PRON | HMM, LSTM | Tie/near-tie performance. |
| PRT | **HMM** | HMM shows the strongest performance. |
| **Overall** | **LSTM & Mistral (0.84)** | Both modern approaches show significant improvement over HMM (0.81). |

#### HMM vs. LSTM vs. ChatGPT (Penn Treebank, 100 Sentences)
| Tag | Best Model | Observations |
| :--- | :--- | :--- |
| PUNCT, ADJ, ADV, NOUN, NUM, PRON, PRT, VERB, X | **ChatGPT** | GPT outperforms on most tags, demonstrating superior context handling. |
| CONJ | **LSTM** | LSTM achieved the highest score. |
| ADP, DET | HMM, LSTM | Tie/near-tie performance. |
| **Overall** | **ChatGPT (0.94)** | The LLM approach, even with minimal data, achieves the highest overall accuracy. |

### Error Analysis (Interpretation of Confusion)

Common misclassifications across models occur due to lexical ambiguity:
* **ADJ vs. NOUN:** Many words can function as both (e.g., "light", "orange").
* **ADV vs. ADJ:** Words like "hard" can be used as both an adverb and an adjective depending on the sentence structure.
* **NOUN vs. VERB:** Most nouns in English can also appear as verbs (e.g., "run", "play").

## 🧠 Challenges and Learnings

### Challenges Faced
* **Unseen Words:** Implementing robust handling techniques (like Laplace smoothing) for words encountered in testing but not in training data for the HMM.
* **Coding the Viterbi Algorithm and LSTM.**
* **Understanding Backward Pass for LSTM.**

### Key Learnings
* **Missing Data Handling:** Learned to apply techniques like **Laplace smoothing** in probabilistic models.
* **Evaluation:** Reinforced understanding of key performance metrics, including **precision, recall, and F1 score**.
* **Dynamic Programming:** Understood how the Viterbi algorithm uses dynamic programming to efficiently reduce the time complexity of finding the optimal sequence path.

This understanding of data handling, evaluation, and dynamic programming is directly applicable to other sequence modeling tasks such as Named Entity Recognition (NER).


## PPT Link
Access the presentation at this link[https://docs.google.com/presentation/d/18aFsNBDpJjkloGXCC3jJM_qrR3lheyVPl0f9gy1FnWQ/edit?usp=sharing]


## 🔗 References

* **Corpora:** [http://www.nltk.org/nltk_data](http://www.nltk.org/nltk_data)
* **Textbook:** Pushpak Bhattacharyya and Aditya Madhav Joshi, *Natural Language Processing*, Wiley India, 2023.
* **Course Material:** CS626: Speech, Natural Language Processing and the Web Course
* **GUI Tools:** [https://www.gradio.app/](https://www.gradio.app/), [https://streamlit.io/](https://streamlit.io/)
* **LLMs:** [https://mistral.ai/news/announcing-mistral-7b](https://mistral.ai/news/announcing-mistral-7b), [https://docs.mistral.ai/getting-started/quickstart/](https://docs.mistral.ai/getting-started/quickstart/)
* **LSTM:** Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing (3rd ed. draft)*. Chapter 13.
