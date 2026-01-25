# GloVe-Based Sentiment Classification with Cross-Domain Transfer Learning

This project implements **movie review sentiment classification** using **GloVe word embeddings** and **cross-domain transfer learning**.  
The goal is to evaluate how pretraining on a **non-movie text domain (AG News)** improves performance on **IMDB movie review sentiment analysis**.

Instead of relying on large pretrained language models, this project uses a **lightweight Transformer Encoder architecture**, showing that strong results can be achieved with simpler NLP models.

---

## 📌 Project Overview

The project consists of two main experiments:

### 1️⃣ Baseline IMDB Sentiment Classifier
- Trained directly on the IMDB dataset
- Uses pretrained **GloVe embeddings**
- Learns sentiment patterns only from movie reviews

### 2️⃣ Cross-Domain Transfer Learning (AG News → IMDB)
- Pretrained on the **AG News dataset** to learn:
  - Rich vocabulary representations
  - General semantic and sentiment-aware features
- Transferred pretrained weights to the IMDB task
- Fine-tuned on IMDB for sentiment classification

---

## 🧠 Motivation

Most NLP transfer learning today relies on large transformer-based models such as BERT or RoBERTa.  
This project focuses on understanding transfer learning using **classic word embeddings** and **lighter architectures**.

Key questions explored:
- How much sentiment knowledge transfers across domains
- Whether news-domain pretraining improves movie review sentiment classification
- How effective GloVe embeddings are when combined with Transformer encoders
- Performance gains without large pretrained language models

---

## 🧾 Datasets Used

### IMDB Movie Reviews
- Binary sentiment classification: **Positive / Negative**
- 50,000 labeled movie reviews

### AG News
- Multi-class news classification dataset
- Used only for **pretraining**
- Helps build a strong and diverse vocabulary

---

## 🏗️ Model Architecture

- **Embedding Layer**
  - Initialized with pretrained **GloVe vectors**
- **Transformer Encoder**
  - Lightweight Transformer encoder block
  - Captures contextual word relationships
- **Feedforward Network**
  - Two linear layers
- **Final Classification Layer**
  - Binary sentiment output

### Fine-Tuning Strategy
- Transformer encoder layers are **unfrozen**
- Final classification layers are **unfrozen**
- Embedding layer remains stable to preserve pretrained semantics

---

## ⚙️ Training Strategy

### Baseline Model
1. Initialize model with GloVe embeddings
2. Train directly on IMDB dataset
3. Evaluate test accuracy

### Transfer Learning Model
1. Pretrain model on AG News dataset
2. Transfer learned weights
3. Fine-tune on IMDB dataset
4. Evaluate and compare performance

---

## 📊 Results

| Model | Training Strategy | IMDB Test Accuracy |
|------|------------------|-------------------|
| Baseline | IMDB only | **65%** |
| Transfer Learning | AG News → IMDB fine-tuning | **86%** |

### Key Observations
- Cross-domain pretraining significantly improves accuracy
- AG News helps learn better vocabulary and semantic representations
- Lightweight Transformer + GloVe is highly effective
- Transfer learning improves generalization and reduces overfitting

---

## 🎬 Application: Movie Review Sentiment Analysis

- Classifies individual movie reviews as **positive or negative**
- Can be extended to:
  - Aggregate sentiment across reviews
  - Rank movies based on overall sentiment scores

---

## 🚀 Key Takeaways

- Transfer learning works well even across unrelated text domains
- GloVe embeddings remain powerful for NLP tasks
- Strong performance does not always require large pretrained models
- Proper pretraining and fine-tuning strategies are critical

---

## 📌 Future Work

- Add attention visualization
- Experiment with different embedding dimensions
- Extend to multi-class sentiment analysis
- Compare with transformer-based pretrained models

---

## 📎 References

- IMDB Movie Reviews Dataset
- AG News Dataset
- GloVe: Global Vectors for Word Representation
