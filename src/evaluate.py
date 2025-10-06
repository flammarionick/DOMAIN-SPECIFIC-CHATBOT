# evaluate.py
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.translate.bleu_score import corpus_bleu
import math

def load_model(path="models/t5_finetuned"):
    tokenizer = T5Tokenizer.from_pretrained(path)
    model = T5ForConditionalGeneration.from_pretrained(path)
    return tokenizer, model

def compute_bleu(references, predictions):
    # references = [[ref1], [ref2], ...]
    # predictions = [pred1, pred2, ...]
    return corpus_bleu([[ref] for ref in references], predictions)

def compute_perplexity(loss):
    return math.exp(loss)

def qualitative_test(tokenizer, model, input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    output = model.generate(**inputs, max_length=50)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    tokenizer, model = load_model()
    refs = ["the cat is on the mat", "there is a cat on the mat"]
    preds = ["the cat is on the mat", "a cat sits on the mat"]

    bleu = compute_bleu(refs, preds)
    ppl = compute_perplexity(2.3)  # example loss
    print("BLEU:", bleu)
    print("Perplexity:", ppl)
    print("Qualitative:", qualitative_test(tokenizer, model, "translate English to French: Hello, how are you?"))
