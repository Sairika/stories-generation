import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st
from config import MODEL_NAME, DEVICE

def load_model():
    """Load GPT-2 model and tokenizer"""
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        return tokenizer, model
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_story(model, tokenizer, prompt, max_length=200, temperature=0.8, 
                  top_k=50, top_p=0.9, repetition_penalty=1.1):
    """Generate story using GPT-2"""
    
    try:
        # Encode the prompt
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    except Exception as e:
        st.error(f"Error generating story: {str(e)}")
        return prompt + " [Error: Could not generate continuation]"
