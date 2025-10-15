#  Domain-Specific Neurology Chatbot using FLAN-T5
  
**Author:** Nicholas Eke  
  
**Date:** October 2025  



##  Project Overview

This project implements a domain-specific conversational chatbot designed to respond to neurology-related questions — particularly around *Multiple Sclerosis (MS)*, *Parkinson’s Disease (PD)*, and *Motor Neuron Disease (MND)*.  
It demonstrates the end-to-end process of fine-tuning a Transformer model (FLAN-T5) on a specialized dataset, evaluating its linguistic and semantic performance, and deploying it via a simple Gradio web interface.

The aim was to explore how transformer-based generative QA models could be adapted for specialized healthcare communication while maintaining accuracy, fluency, and contextual understanding.



##  Objectives

1. Build a domain-aligned dataset of doctor–patient style Q&A pairs in neurology.  
2. Preprocess and tokenize the dataset for Transformer input.  
3. Fine-tune a pre-trained **FLAN-T5 model** on the neurology dataset.  
4. Evaluate model performance using BLEU, ROUGE-L, and BERTScore.  
5. Deploy an interactive chatbot interface using **Gradio**.  



##  Dataset

Link: https://www.kaggle.com/datasets/jpmiller/layoutlm  
- ~1,000 domain-specific Q&A pairs curated and cleaned for training.  
- Covers symptom descriptions, medication advice, diagnostic tests, and patient communication patterns.  
- All text normalized (lowercased, stripped of punctuation, duplicates removed).  

Preprocessing Steps:
- Missing values handled by context-based replacement.  
- Tokenization via `T5Tokenizer` from Hugging Face.  
- Maximum sequence length capped at 192 tokens.  
- Dataset split: 80% train, 10% validation, 10% test.  



## Model and Training

Base Model: `google/flan-t5-small`  
  
Frameworks: Hugging Face Transformers, PyTorch, TensorFlow (for compatibility)

Training Parameters
| Parameter | Value |
|------------|--------|
| Learning Rate | 3e-5 |
| Batch Size | 4 |
| Epochs | 10 |
| Weight Decay | 0.01 |
| Beam Search | Disabled (num_beams = 1) |
| Max Tokens | 192 |
| Gradient Accumulation | Enabled |



##  Evaluation Results

| Metric | Score |
|--------|-------|
| **BLEU** | 0.05 |
| **ROUGE-L** | 0.187 |
| **BERTScore (F1)** | 0.339 |

### Interpretation
- **BLEU** reflects limited lexical overlap (expected for generative text).  
- **ROUGE-L** shows moderate recall of key phrases.  
- **BERTScore** indicates fair semantic alignment (≈0.34), suggesting the model captures meaning even when phrasing differs.

While numeric metrics are modest, qualitative testing showed that the chatbot generates coherent and medically relevant answers for most in-domain queries.



## 💬 Deployment

A simple **Gradio** web app (`src/app.py`) was built for local interaction:

```bash
python src/app.py

Once launched, visit
 http://127.0.0.1:7860
to chat with the model in real time.



Model Upload Note (Important)

Due to GitHub’s 100 MB file limit, the fine-tuned model (flan_t5_neurology_v3) — containing model.safetensors and optimizer checkpoints (~600 MB total) — could not be uploaded.

The model and tokenizer are available locally and can be shared upon request via:

Google Drive link (https://drive.google.com/file/d/1qf0q6hicjTXJ57ZrwNHQyoFD49k0CSug/view?usp=sharing)

The model is in checkpoints/flan_t5_neurology_v3

All source code, dataset, and evaluation logic are included in this repository.




How to Recreate the Model

Clone the repository:

git clone https://github.com/flammarionick/DOMAIN-SPECIFIC-CHATBOT.git
cd DOMAIN-SPECIFIC-CHATBOT


Create a virtual environment:

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


Fine-tune the base model:

python src/train_model.py


Run the chatbot:

python src/app.py





Challenges Encountered

Memory limitations: Training large transformer models on CPU-only environment caused crashes; mitigated using smaller batch sizes and disabling beam search.

Model size: Checkpoints exceeded GitHub’s file limit — attempts to clean the repository using git filter-repo were successful in removing heavy history, but large model files must remain local.

Tokenizer loading errors: Resolved missing SentencePiece dependency and added restore_tokenizer.py utility for safe reconstruction of tokenizer state.

BERTScore memory issues: Introduced batched semantic evaluation using lightweight bert-base-uncased model.




 Key Learnings

Domain-specific fine-tuning significantly improves contextual response quality over zero-shot models.

Evaluating generative QA models requires semantic metrics beyond BLEU/ROUGE.

Model deployment must consider resource and hosting constraints (e.g., using Hugging Face Hub or Streamlit Cloud for public demos).





 References

Hugging Face Transformers Library

Google FLAN-T5 Model Card

SacreBLEU & ROUGE Scoring Implementations

BERTScore: Zhang et al., 2020
