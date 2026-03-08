import os
import torch
from transformers import RobertaTokenizer

def test_roberta_shape():
    print("=" * 60)
    print("TESTING ROBERTA TOKENIZATION SHAPE")
    print("=" * 60)
    
    # 1. Provide the exact path where YOUR fine-tuned model lives
    model_path = "models/roberta_fake_news"
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at '{model_path}'.")
        print("Make sure you run this script from the 'backend' folder.")
        return

    print(f"✓ Found local model directory: {model_path}")
    
    # 2. Load YOUR local tokenizer
    print("✓ Loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    
    # 3. Create a sample string
    sample_text = "This is a breaking news alert regarding the economy."
    print(f"✓ Sample Input Text: \"{sample_text}\"")
    
    # 4. Process exactly how RoBERTa requires it (truncating long text and padding to max length)
    print("✓ Tokenizing with max_length=512 and padding...")
    inputs = tokenizer(
        sample_text,
        return_tensors="pt",        # Return PyTorch tensors
        truncation=True,            # Cut off text if longer than 512
        padding="max_length",       # Pad missing words up to exactly 512
        max_length=512              # The strict length bound for RoBERTa
    )
    
    # 5. Extract the raw Tensor Shape
    # inputs['input_ids'] holds the numerical representation of the words
    shape = list(inputs['input_ids'].shape)
    
    print("\n[RESULT]")
    print(f"Output Tensor Object Type: {type(inputs['input_ids'])}")
    print(f"Actual Shape Generated:    {shape}")
    
    if shape == [1, 512]:
        print("\nSTATUS: PASS ✓ (Successfully generated a [1, 512] Tensor)")
    else:
        print("\nSTATUS: FAIL ❌ (Shape mismatch)")

if __name__ == "__main__":
    test_roberta_shape()
