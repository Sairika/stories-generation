import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
import re

# Page configuration
st.set_page_config(
    page_title="GPT-2 Story Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .story-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 20px 0;
    }
    .prompt-examples {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load and cache the GPT-2 model"""
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def clean_text(text):
    """Clean and format generated text"""
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common punctuation issues
    text = re.sub(r'\s+([.!?,:;])', r'\1', text)
    
    # Fix contractions
    contractions = {
        " n't": "n't", " 's": "'s", " 're": "'re", " 've": "'ve", 
        " 'll": "'ll", " 'm": "'m", " 'd": "'d"
    }
    
    for wrong, right in contractions.items():
        text = text.replace(wrong, right)
    
    return text.strip()

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

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 AI Story Generator</h1>', unsafe_allow_html=True)
    st.markdown("*Powered by GPT-2 - Create engaging stories from simple prompts*")
    
    # Initialize session state for prompt
    if 'current_prompt' not in st.session_state:
        st.session_state.current_prompt = ""
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Generation Settings")
        
        max_length = st.slider("Max Length", 50, 500, 200)
        temperature = st.slider("Creativity (Temperature)", 0.1, 2.0, 0.8, 0.1)
        top_k = st.slider("Top-K Sampling", 1, 100, 50)
        top_p = st.slider("Top-P Sampling", 0.1, 1.0, 0.9, 0.05)
        repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.1, 0.1)
        
        st.markdown("---")
        st.markdown("**💡 Tips:**")
        st.markdown("- Higher temperature = more creative")
        st.markdown("- Lower repetition penalty = more repetitive")
        st.markdown("- Experiment with different settings!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("✍️ Enter Your Story Prompt")
        prompt = st.text_area(
            "What story would you like me to tell?",
            value=st.session_state.current_prompt,
            height=100,
            placeholder="Once upon a time in a distant galaxy...",
            help="Enter a creative prompt to start your story. The AI will continue from where you leave off!"
        )
        
        # Generate button
        if st.button("🎭 Generate Story", type="primary", use_container_width=True):
            if prompt.strip():
                generate_and_display_story(
                    prompt, max_length, temperature, top_k, top_p, repetition_penalty
                )
            else:
                st.warning("Please enter a prompt to generate a story!")
    
    with col2:
        st.subheader("💡 Example Prompts")
        st.markdown('<div class="prompt-examples">', unsafe_allow_html=True)
        
        example_prompts = [
            "In a world where dreams become reality, a young girl discovers she can control nightmares...",
            "The last person on Earth discovers they're not actually alone...",
            "A mysterious letter arrives that changes everything Sarah thought she knew about her family...",
            "Deep in the Amazon rainforest, explorers find a city that shouldn't exist...",
            "On her 18th birthday, she gained the power to see people's last thoughts..."
        ]
        
        for i, example in enumerate(example_prompts):
            if st.button(f"📝 {example[:25]}...", key=f"example_{i}"):
                st.session_state.current_prompt = example
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def generate_and_display_story(prompt, max_length, temperature, top_k, top_p, repetition_penalty):
    """Generate and display the story with loading animation"""
    
    # Loading animation
    with st.spinner("🤖 AI is crafting your story..."):
        try:
            tokenizer, model = load_model()
            
            if tokenizer is None or model is None:
                st.error("Failed to load the model. Please refresh the page and try again.")
                return
            
            # Progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Generate story
            story = generate_story(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            
            # Clean the generated text
            cleaned_story = clean_text(story)
            
            # Display results
            st.success("✨ Story generated successfully!")
            
            st.markdown('<div class="story-container">', unsafe_allow_html=True)
            st.markdown("### 📖 Your Generated Story")
            st.write(cleaned_story)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional features
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Generate Another"):
                    st.rerun()
            
            with col2:
                st.download_button(
                    "💾 Download Story",
                    data=cleaned_story,
                    file_name=f"generated_story_{int(time.time())}.txt",
                    mime="text/plain"
                )
            
            with col3:
                word_count = len(cleaned_story.split())
                st.metric("Word Count", word_count)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please try again with a different prompt or adjust the settings.")
            st.exception(e)  # For debugging

if __name__ == "__main__":
    main()
