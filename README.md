# NLP_HW4_700776912
# Programming Assignment – NLP & Transformers

**Name:** Pavani Krishna Priya Chava
**700#:** 700776912  

---

## Overview  

This repository contains solutions for three programming questions from the NLP HW4:

### Q1 – Character-Level RNN Language Model
- Goal: Train a character-level RNN to predict the next character given previous characters.
- Model: Embedding → RNN (Vanilla / GRU / LSTM) → Linear → Softmax.
- Features: Teacher forcing, cross-entropy loss, Adam optimizer.
- Outputs: Training/validation loss curves, temperature-controlled text generations, reflection on sequence length, hidden size, and sampling temperature.

### Q2 – Mini Transformer Encoder for Sentences
- Goal: Build a mini Transformer encoder to process a batch of short sentences.
- Features: Tokenization, embedding, sinusoidal positional encoding, self-attention, multi-head attention (2–4 heads), feed-forward, Add & Norm.
- Outputs: Input tokens, final contextual embeddings, attention weights/heatmap visualization.

### Q3 – Scaled Dot-Product Attention
- Goal: Implement `Attention(Q,K,V) = softmax((Q K^T)/√(d_k)) V`.
- Features: Tested with random Q, K, V inputs, prints attention weights, output vectors, and softmax stability checks.
- Notes: Demonstrates numerical stability of scaling and subtract-max softmax.

---

## Setup Instructions  

1. Install required packages:

```bash
pip install torch numpy matplotlib
