import gradio as gr
import tensorflow as tf
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

MODEL_DIR = "../models/t5-chatbot"

# Load fine-tuned model
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = TFT5ForConditionalGeneration.from_pretrained(MODEL_DIR)

def generate_response(user_text: str) -> str:
    """Generate chatbot response for user input."""
    input_text = "question: " + user_text
    inputs = tokenizer(input_text, return_tensors="tf", truncation=True, padding=True)

    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=3,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Build Gradio UI
iface = gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs="text",
    title="Domain-Specific Chatbot",
    description="Ask me a question within my domain (e.g., healthcare, finance, etc.)."
)

if __name__ == "__main__":
    iface.launch()