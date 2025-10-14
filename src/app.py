import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

MODEL_DIR = "../models/flan_t5_neurology_v3"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

# ✅ Force to CPU if no GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=2,
            temperature=0.7,
            early_stopping=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# ✅ Gradio Interface
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Ask your neurology chatbot a question..."),
    outputs=gr.Textbox(label="Response"),
    title="🧠 Neurology Assistant Chatbot",
    description="A domain-specific chatbot fine-tuned on neurology data using FLAN-T5.",
)

iface.launch()