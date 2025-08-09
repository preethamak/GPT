# Build GPT from Scratch — Inspired by Andrej Karpathy

This project is a from-scratch implementation of GPT, built while following Andrej Karpathy’s tutorials and reading *Attention Is All You Need*.  
The goal was to deeply understand the inner workings of Transformers, Attention mechanisms, and how GPT models are trained and generate text.

##  What I Learned
- Fundamentals of Transformers
- The concept and math behind **Attention** (Self-Attention & Cross-Attention)
- Positional Encoding
- Multi-Head Attention
- Layer Normalization and Residual Connections
- Tokenization and embeddings
- GPT architecture and forward pass logic
- Training loop implementation from scratch

##  Training & Dataset
- Dataset: Shakespeare’s text dataset
- Model trained from scratch to generate Shakespeare-style text
- Custom implementation of attention layers and Transformer blocks

##  Key Features
- Fully implemented Transformer architecture in Python
- Self-attention and cross-attention mechanics coded manually
- Tokenizer and vocabulary building from scratch
- Ability to train on custom datasets
- Text generation from trained model

##  Example Output
After training on Shakespeare’s dataset, the model can produce text in the style of Shakespeare:


##  How to Run
```bash
git clone https://github.com/preethamak/gpt
cd gpt
python train.py
