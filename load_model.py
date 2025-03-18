import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Define the path to the "model" folder for saving the model
save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model")  # The "model" folder is located next to the script

# Load the GPT-2 model and tokenizer from Hugging Face
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Create the folder if it does not exist
os.makedirs(save_path, exist_ok=True)

# Save the model and tokenizer to the "model" folder
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("The model and tokenizer have been successfully saved in the 'model' folder!")
