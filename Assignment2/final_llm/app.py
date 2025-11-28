import torch
import re
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from tqdm import tqdm
import string
import os

# --- Set seeds for reproducibility ---
def set_seed_(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed_(42)
set_seed(42)

hf_token = "{hf_token}"

pos_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
pos_model = AutoModelForCausalLM.from_pretrained(pos_model_name, use_auth_token=hf_token)
pos_tokenizer = AutoTokenizer.from_pretrained(pos_model_name, use_auth_token=hf_token)

pos_tokenizer.padding_side = 'left'

# Ensure pad_token_id is set
if pos_tokenizer.pad_token_id is None:
    pos_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    pos_model.resize_token_embeddings(len(pos_tokenizer))

# --- Setup device ---
device_stereo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pos_model.to(device_stereo)

# --- Inference helper function ---
def inference(prompt, max_new_tokens=128, temperature=0.3):
    model_inputs = pos_tokenizer([prompt], return_tensors="pt").to(device_stereo)
    input_ids_cutoff = model_inputs.input_ids.size(dim=1)
    
    generated_ids = pos_model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        top_p=1.0,
        temperature=temperature,
        do_sample=True,
        pad_token_id=pos_tokenizer.eos_token_id
    )
    
    completion = pos_tokenizer.decode(
        generated_ids[0][input_ids_cutoff:],
        skip_special_tokens=True
    )
    
    return completion.strip()

# --- Transliteration function with one-shot examples ---
def get_pos_responses_with_one_shot(words, max_new_tokens=128):
    prompt_template = """
<s> [INST] You are a Transliterator.  
Your task is to transliterate each English word into Hindi (Devanagari script).  
Abbreviations like "USA" should be transliterated letter by letter (e.g., यूएसए).

[INST]
### Instructions:
- Input: a word in English.  
- Output: only the Hindi transliteration. Do NOT add explanations, extra text, or punctuation.

### Examples:
Input: janamdin
Output: जन्मदिन

Input: ram
Output: राम

Input: pushpak
Output: पुष्पक

Input: shalaka
Output: शलाका

Input: nri
Output: एनआरआई

Input: doctor
Output: डॉक्टर

### Now transliterate the following word:
Only provide the transliteration, no extra text or explanation.
Input: {word}
Output:
[/INST]
"""
    responses = []
    for word in tqdm(words):
        prompt = prompt_template.format(word=word)
        response = inference(prompt, max_new_tokens=max_new_tokens)
        responses.append(response.strip())
    return responses

import re
import string
import torch
import gradio as gr

# --- Assuming model and tokenizer are already loaded ---
# pos_model, pos_tokenizer, device_stereo
# get_pos_responses_with_one_shot function is defined as before

# Transliteration function for Gradio
def transliterate_sentence(input_str):
    # Split into words but keep punctuation separate
    words = re.findall(r'\b\w+\b|[' + re.escape(string.punctuation) + r']', input_str)

    # Transliterate only word tokens
    transliterated_words = []
    for word in words:
        if re.match(r'\w+', word):
            translit = get_pos_responses_with_one_shot([word], max_new_tokens=128)[0]
            translit = translit.strip().split('\n')[0].strip().split(' ')[0]
            transliterated_words.append(translit)
        else:
            transliterated_words.append(word)

    # Reassemble sentence and fix spacing before punctuation
    transliterated_str = ' '.join(transliterated_words)
    transliterated_str = re.sub(r'\s([?.!,](?:\s|$))', r'\1', transliterated_str)

    return transliterated_str

# --- Gradio Interface ---
iface = gr.Interface(
    fn=transliterate_sentence,
    inputs=gr.Textbox(label="Enter English sentence"),
    outputs=gr.Textbox(label="Hindi Transliteration"),
    title="Transliteration Mistral 7B by Shalaka and Kaustubh",
    description="Transliterate English text into Hindi (Devanagari) using Mistral 7B."
)

if __name__ == "__main__":
    iface.launch(share=True)