
import streamlit as st
import json
import os

# Professional CSS Injection
st.markdown('\n<style>\n.main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }\n.stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }\n</style>\n', unsafe_allow_html=True)

def load_registry():
    if os.path.exists('model_registry.json'):
        with open('model_registry.json', 'r') as f:
            return json.load(f)
    return {}

registry = load_registry()
model_option = st.selectbox("Select Architecture", options=list(registry.keys()) if registry else ["No models found"])

if model_option in registry:
    model_path = os.path.join('saved_models', os.path.basename(registry[model_option]['path']))
    if os.path.exists(model_path):
        st.success(f"Ready to load {model_option}")
    else:
        st.warning(f"Model file for {model_option} not found at {model_path}")
