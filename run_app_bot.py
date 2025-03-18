import os
import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load configuration from config.json
config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# Define the path to the model (located in the "model" folder)
save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model")

# Load the model and tokenizer from the local folder
model = GPT2LMHeadModel.from_pretrained(save_path)
tokenizer = GPT2Tokenizer.from_pretrained(save_path)

# Set the end-of-sequence token as the padding token (pad_token_id)
model.config.pad_token_id = model.config.eos_token_id

# Determine which device to use (GPU with ROCm/Vulkan or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Input text for generation
input_text = "Сегодня погода"

# Encode the text into tensors and move them to the selected device
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# Create an attention mask to consider all input text tokens
attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)

# Generate text using parameters from config.json, including repetition filtering
output = model.generate(
    input_ids,
    max_length=config["max_length"],  # Maximum length of the generated text
    num_return_sequences=config["num_return_sequences"],  # Number of output sequences
    attention_mask=attention_mask,
    do_sample=config["do_sample"],  # Enable sampling for more diverse text
    temperature=config["temperature"],  # Controls randomness (0.7 = moderate randomness)
    top_k=config["top_k"],  # Limit token selection to the top-K most likely words
    top_p=config["top_p"],  # Use nucleus sampling for more natural text generation
    repetition_penalty=config["repetition_penalty"],  # Penalty for repeated words and phrases
    no_repeat_ngram_size=config["no_repeat_ngram_size"]  # Minimum phrase size that should not repeat
)

# Decode the generated output into readable text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Post-processing to remove consecutive duplicate words
words = generated_text.split()
filtered_text = []
for word in words:
    if len(filtered_text) == 0 or word != filtered_text[-1]:  # Add word only if it doesn't repeat the previous one
        filtered_text.append(word)

# Print the final result without duplicates
print("Result:", " ".join(filtered_text))
