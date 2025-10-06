import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
from preprocess import load_and_process

MODEL_NAME = "t5-base"
OUTPUT_DIR = "../models/t5-chatbot"

def to_tf_dataset(tokenized_dataset, split: str, batch_size: int = 8):
    """Convert Hugging Face Dataset to tf.data.Dataset."""
    features = {
        "input_ids": tokenized_dataset[split]["input_ids"],
        "attention_mask": tokenized_dataset[split]["attention_mask"],
    }
    labels = tokenized_dataset[split]["labels"]

    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.shuffle(1000).batch(batch_size)
    return ds

def main():
    # Load and preprocess dataset
    dataset = load_and_process("../data")
    train_tf = to_tf_dataset(dataset, "train")
    valid_tf = to_tf_dataset(dataset, "validation")

    # Load model + tokenizer
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = TFT5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn)

    # Train
    model.fit(train_tf, validation_data=valid_tf, epochs=3)

    # Save model + tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model saved at {OUTPUT_DIR}")

if __name__ == "__main__":
    main()