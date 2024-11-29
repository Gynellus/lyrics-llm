import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_baseline_model(model_name: str = "gpt2", device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Loads the baseline GPT-2 Small model and tokenizer.

    Args:
        model_name (str): Name of the model to load from Hugging Face Hub.
        device (str): Device to load the model on ('cuda' or 'cpu').

    Returns:
        model: The loaded GPT-2 model.
        tokenizer: The loaded tokenizer.
    """
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set

    print(f"Loading baseline model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically maps to available device
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,  # Use float16 for faster inference on GPU
    )
    model.to(device)
    model.eval()  # Set model to evaluation mode
    print("Baseline model and tokenizer loaded successfully.\n")
    return model, tokenizer

def generate_baseline_lyrics(model, tokenizer, song_tags: str = "pop", max_new_tokens: int = 200,
                             temperature: float = 0.7, top_k: int = 50):
    """
    Generates song lyrics using the baseline GPT-2 Small model.

    Args:
        model: The GPT-2 model.
        tokenizer: The tokenizer associated with the model.
        song_tags (str): Genres or tags for the song (e.g., 'pop').
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.

    Returns:
        str: The generated song lyrics.
    """
    # Prepare the prompt
    prompt = f"Song with the following genres: {song_tags}. Here are the song lyrics:\n"
    print(f"Prompt:\n{prompt}")

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    # Generate text
    print("Generating lyrics using the baseline GPT-2 Small model...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=0.95,  # Cumulative probability for nucleus sampling
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,  # Ensure proper padding
        )

    # Decode the generated tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the lyrics part after the prompt
    lyrics = generated_text[len(prompt):].strip()
    return lyrics

def main():
    # Load the baseline GPT-2 Small model and tokenizer
    model_name = "gpt2"  # GPT-2 Small
    model, tokenizer = load_baseline_model(model_name)

    # Define the song tags
    song_tags = "pop"

    # Generate lyrics
    lyrics = generate_baseline_lyrics(model, tokenizer, song_tags)

    # Print the generated lyrics
    print("\n--- Generated Song Lyrics (Baseline GPT-2 Small) ---\n")
    print(lyrics)
    print("\n------------------------------------------------------\n")

    # Optional: Save the lyrics to a file
    output_file = 'generated_baseline_pop_song.txt'
    with open(output_file, 'w') as f:
        f.write(lyrics)
    print(f"Lyrics saved to {output_file}")

if __name__ == "__main__":
    main()
