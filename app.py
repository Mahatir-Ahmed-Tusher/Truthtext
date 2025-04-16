import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st

# Load the pre-trained model
model = tf.keras.models.load_model('detect_LLM_colab_method2')

# Prediction function
def LLM_text(text):
    prediction = model.predict([text])
    return round(prediction[0][0] * 100, 2)

# Predefined examples
ai_generated_examples = [
    "The concept of car-free cities is gaining traction as urban areas seek to reduce pollution and improve quality of life. By prioritizing pedestrian zones and public transportation, cities can create a more sustainable future.",
    "Exploring Venus has long been a goal for space agencies due to its proximity to Earth and its potential for scientific discovery. However, its harsh surface conditions pose significant challenges for exploration.",
    "Driverless cars represent a significant advancement in automotive technology, promising to reduce accidents caused by human error and improve traffic efficiency. However, ethical and regulatory challenges remain."
]

human_written_examples = [
    "Car-free cities are an interesting idea, but implementing them requires significant changes to urban infrastructure. Public transportation must be efficient and accessible to make this feasible.",
    "Venus is a fascinating planet, but its extreme temperatures and atmospheric pressure make it difficult to explore. Scientists are still figuring out how to design spacecraft that can withstand these conditions.",
    "Self-driving cars are exciting, but they raise important questions about safety and liability. How do we ensure these vehicles make ethical decisions in critical situations?"
]

# Custom CSS for premium look
def custom_css():
    st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #f0f4f8, #d9e2ec);
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        h1 {
            font-size: 48px;
            color: white;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
        }
        .hero {
            background: linear-gradient(135deg, #3498db, #8e44ad);
            padding: 40px;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            margin-bottom: 40px;
        }
        .subtitle {
            font-size: 20px;
            color: white;
            opacity: 0.9;
        }
        textarea {
            border: 2px solid #bdc3c7 !important;
            border-radius: 10px !important;
            padding: 15px !important;
            font-size: 16px !important;
            background: #fff !important;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05) !important;
        }
        button {
            background: linear-gradient(135deg, #3498db, #2980b9) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 12px 25px !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            transition: transform 0.2s, box-shadow 0.2s !important;
        }
        button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4) !important;
        }
        .example-box {
            border: 2px dashed #3498db !important;
            padding: 15px !important;
            margin-bottom: 15px !important;
            background: #fff !important;
            border-radius: 10px !important;
            cursor: pointer !important;
        }
        .example-box:hover {
            background: #ecf0f1 !important;
            border-color: #8e44ad !important;
        }
        .info-section {
            background: #fff;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
            margin-bottom: 40px;
        }
        .footer {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            text-align: center;
            border-radius: 20px;
            box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
        }
        .footer a {
            color: #3498db;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
    """, unsafe_allow_html=True)

# Apply custom CSS
custom_css()

# Hero Section with Logo
st.markdown("""
<div class="hero">
    <div style='text-align: center;'>
        <img src='https://i.postimg.cc/Vv8310Cf/Truthtext.png' alt='TextTruth Logo' style='height: 60px; margin-bottom: 20px;'>
        <h1>TextTruth: AI vs Human Text Detector</h1>
        <p class='subtitle'>Unveil the truth behind your text with cutting-edge AI technology.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Input and Output Section
st.markdown("### Enter Your Text")
user_input = st.text_area("Input Text", placeholder="Paste or type your text here...", height=150)
if st.button("Analyze Now"):
    if user_input.strip() == "":
        st.error("Please enter some text to analyze.")
    else:
        prediction = LLM_text(user_input)
        st.success(f"Your text is {prediction}% likely to be AI-generated.")

# Examples Section
st.markdown("### AI-Generated Examples")
for example in ai_generated_examples:
    st.markdown(f'<div class="example-box">{example}</div>', unsafe_allow_html=True)

st.markdown("### Human-Written Examples")
for example in human_written_examples:
    st.markdown(f'<div class="example-box">{example}</div>', unsafe_allow_html=True)

# How It Works Section
st.markdown('<div class="info-section">', unsafe_allow_html=True)
st.markdown("""
## How It Works
- **Model Architecture**: Built using transfer learning with BERT embeddings and a custom Multi-Head Attention layer.
- **Training Data**: Trained on a dataset of human and AI-generated essays from Kaggle.
- **Accuracy**: Achieved high ROC AUC scores in distinguishing AI-generated text from human-written text.
""")
st.markdown('</div>', unsafe_allow_html=True)

# Try It Out Section
st.markdown('<div class="info-section">', unsafe_allow_html=True)
st.markdown("""
## Try It Out!
Paste your own text or drag an example above into the input box, then click 'Analyze Now' to reveal the truth.
""")
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("""
© 2025 TextTruth. All rights reserved. Powered by <a href='https://huggingface.co' target='_blank'>Hugging Face</a>.
""")
st.markdown('</div>', unsafe_allow_html=True)