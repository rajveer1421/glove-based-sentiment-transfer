# From Full to Efficient Fine-Tuning: Sentiment Classification with GloVe and Lightweight Transformers

This project explores **movie review sentiment classification** using **GloVe word embeddings** and a **lightweight Transformer Encoder**, with a strong emphasis on **fine-tuning strategies** and **cross-domain transfer learning**.

Instead of relying on large pretrained language models (e.g., BERT, RoBERTa), this work investigates how far we can go with:
- Classic word embeddings (GloVe)
- Smaller Transformer encoders
- Carefully designed fine-tuning techniques

A major focus of this project is experimenting with multiple fine-tuning approaches, analyzing their **accuracy, efficiency, and training cost**, and understanding the trade-offs between them.

---

## 📌 Project Overview

The project consists of two main phases:

### 1️⃣ Baseline IMDB Sentiment Classifier
- Trained directly on the IMDB dataset
- Uses pretrained **GloVe embeddings**
- Learns sentiment patterns only from movie reviews

### 2️⃣ Cross-Domain Transfer Learning (AG News → IMDB)
- Pretrained on the **AG News dataset** to learn:
  - Rich vocabulary representations
  - General semantic and contextual features
- Pretrained weights are transferred to the IMDB task
- Multiple **fine-tuning strategies** are explored and compared

---

## 🧠 Motivation

Most modern NLP pipelines depend heavily on large pretrained models.  
This project intentionally takes a **lighter and more interpretable approach** to answer the following questions:

- How much knowledge transfers across unrelated text domains?
- Can news-domain pretraining help movie review sentiment classification?
- How effective are GloVe embeddings when combined with Transformer encoders?
- Which fine-tuning strategy provides the best balance between accuracy, training time, and parameter efficiency?

---

## 🧾 Datasets Used

### IMDB Movie Reviews
- Binary sentiment classification: **Positive / Negative**
- 50,000 labeled movie reviews

### AG News
- Multi-class news classification dataset
- Used **only for pretraining**
- Helps build robust semantic representations and vocabulary coverage

---

## 🏗️ Model Architecture

- **Embedding Layer**
  - Initialized with pretrained **GloVe vectors**
- **Transformer Encoder**
  - Lightweight Transformer encoder block
  - Captures contextual word relationships
- **Feedforward Network**
  - Two linear layers inside the Transformer encoder
- **Final Classification Layer**
  - Binary sentiment output

---

## 🔧 Fine-Tuning Strategies (Core Contribution)

This project systematically explores multiple fine-tuning techniques, highlighting their advantages and trade-offs.

---

### 🔹 1. Full Fine-Tuning (Baseline)

**Approach**
- All model parameters are unfrozen:
  - Embedding layer
  - Transformer encoder
  - Final classifier

**Result**
- **IMDB Test Accuracy: 65%**

**Observation**
- Prone to overfitting
- Slower convergence
- Less effective despite higher parameter updates

---

### 🔹 2. Selective Fine-Tuning (Best Accuracy)

**Approach**
- Frozen:
  - Embedding layer
  - Most Transformer encoder parameters
- Unfrozen:
  - TransformerEncoder.linear2
  - Final classification layer

**Result**
- **IMDB Test Accuracy: 86%**

**Advantages**
- Strong generalization
- Better stability during training
- Efficient parameter updates

---

### 🔹 3. Additive Fine-Tuning (Adapter-Based)

**Approach**
- Introduced a feature adapter module:
  - 2 Linear layers + 1 ReLU activation
- Adapter placed between TransformerEncoder.linear1 and linear2
- Only the following were unfrozen:
  - Adapter layers
  - Final classifier
- Core Transformer weights remained frozen

**Result**
- **IMDB Test Accuracy: 78%**

**Observation**
- Improved parameter efficiency
- Reduced risk of catastrophic forgetting

---

### 🔹 4. Efficient Adapter Fine-Tuning (Best Efficiency)

**Approach**
- Same adapter-based structure
- Reduced dimensionality
- Minimal number of trainable parameters
- Faster convergence with fewer epochs

**Result**
- **IMDB Test Accuracy: 85%**
- Achieved in significantly less training time

---

## ⚙️ Fine-Tuning Summary

| Strategy | Trainable Parameters | Accuracy | Training Cost |
|--------|---------------------|----------|---------------|
| Full Fine-Tuning | All layers | 65% | High |
| Selective Fine-Tuning | Encoder linear2 + classifier | **86%** | Moderate |
| Additive (Adapter) | Adapter + classifier | 78% | Low |
| Efficient Adapter | Minimal adapters | **85%** | **Very Low** |

---

## 📊 Training Analysis

- Training loss and accuracy curves were plotted for each fine-tuning strategy
- Clear differences observed in convergence speed, stability, and overfitting behavior

All experiments and plots are available in the linked Kaggle notebooks.

---

## 🎬 Application: Movie Review Sentiment Analysis

- Classifies individual movie reviews as **positive or negative**
- Can be extended to aggregate sentiment analysis and movie ranking systems

---

## 🚀 Key Takeaways

- Cross-domain transfer learning works well even with classic embeddings
- Fine-tuning strategy matters more than model size
- Lightweight Transformers can achieve strong results
- Adapter-based fine-tuning offers excellent efficiency
- Large pretrained models are not always necessary

---

## 📎 Resources & Notebooks

📓 Kaggle Notebooks:
- Fine-Tuning Techniques (LoRA-inspired):  
  https://www.kaggle.com/code/rajveergup1455/fine-tuning-using-lora
- Sentiment Review Analysis:  
  https://www.kaggle.com/code/rajveergup1455/fine-tune-sentiment-review-analysis

⭐ If you find the notebooks useful, please consider upvoting them on Kaggle!
