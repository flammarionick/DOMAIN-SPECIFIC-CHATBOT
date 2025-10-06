import os
import json
from datasets import load_dataset
from transformers import T5Tokenizer

MODEL_NAME = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

def load_and_process(data_path: str, max_input_length: int = 256, max_target_length: int = 128):
    """
    Load JSONL dataset and preprocess it for T5 fine-tuning.

    Args:
        data_path (str): Path to JSONL dataset (train/valid/test).
        max_input_length (int): Max length for input sequences.
        max_target_length (int): Max length for target sequences.

    Returns:
        datasets.DatasetDict: Tokenized dataset splits.
    """
    data_files = {
        "train": os.path.join(data_path, "train.jsonl"),
        "validation": os.path.join(data_path, "valid.jsonl"),
        "test": os.path.join(data_path, "test.jsonl"),
    }
    dataset = load_dataset("json", data_files=data_files)

    def preprocess_function(example):
        input_text = "question: " + example["user"]
        if "context" in example and example["context"]:
            input_text += " context: " + example["context"]

        target_text = example["bot"]

        model_inputs = tokenizer(
            input_text,
            max_length=max_input_length,
            truncation=True,
            padding="max_length"
        )

        labels = tokenizer(
            target_text,
            max_length=max_target_length,
            truncation=True,
            padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    tokenized = dataset.map(preprocess_function, remove_columns=dataset["train"].column_names)
    return tokenized

if __name__ == "__main__":
    dataset = load_and_process("../data")
    print(dataset)