import pandas as pd
import random
import re

# --- Simple keyword-based intent tagging ---
def tag_intent(question: str) -> str:
    q = str(question).lower()
    if any(w in q for w in ["eat", "food", "diet", "drink", "nutrition"]):
        return "diet"
    elif any(w in q for w in ["symptom", "sign", "feel", "pain", "headache", "dizzy"]):
        return "symptom"
    elif any(w in q for w in ["drug", "medicine", "medication", "pill", "treatment", "insulin"]):
        return "medication"
    elif any(w in q for w in ["what is", "define", "meaning", "definition"]):
        return "definition"
    elif any(w in q for w in ["prevent", "avoid", "reduce", "lifestyle"]):
        return "prevention"
    elif any(w in q for w in ["test", "diagnose", "blood test", "check", "screening"]):
        return "diagnosis"
    else:
        return "general_health"

# Load dataset
df = pd.read_csv("medquad.csv")

# Map to chatbot schema
df_chatbot = pd.DataFrame({
    "id": range(1, len(df) + 1),
    "user": df["question"],
    "bot": df["answer"],
    "context": df["source"],  # using source as context / knowledge base snippet
    "intent": df["question"].apply(tag_intent),  # auto intent tagging
    "domain_label": [
        "in_domain" if "diabetes" in str(focus).lower() else "out_of_domain"
        for focus in df["focus area"]
    ]
})

# Filter to balance in_domain and out_of_domain
in_domain = df_chatbot[df_chatbot["domain_label"] == "in_domain"]
out_domain = df_chatbot[df_chatbot["domain_label"] == "out_of_domain"]

# Balance sizes: sample out-of-domain to match in-domain
if len(out_domain) > len(in_domain):
    out_domain = out_domain.sample(n=len(in_domain), random_state=42)

# Combine back
balanced = pd.concat([in_domain, out_domain]).reset_index(drop=True)

# Save to new CSV / JSONL
balanced.to_csv("chatbot_dataset.csv", index=False)
balanced.to_json("chatbot_dataset.jsonl", orient="records", lines=True)

print("✅ Preprocessed dataset saved: chatbot_dataset.csv & chatbot_dataset.jsonl")
print(balanced.head(10))