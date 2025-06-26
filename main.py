import os
import pandas as pd
import torch
import gc 
from poekey.summarize import summarize_poem
from poekey.extract_poekey import extract_emotion
from poekey.extract_poekey import extract_visuals
from poekey.extract_poekey import extract_theme
from prompting.prompt_tuner import get_sdxl_prompt_from_openai
from diffusion.generate_images import generate_image

DATA_PATH = "data/MinPo_Dataset.csv"
df = pd.read_csv(DATA_PATH)
print(f"CSV has {len(df)} rows")

OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for idx, row in df.iterrows():
    poem = row["Poem"]
    
    summary = summarize_poem(poem)

    emotion = extract_emotion(summary)
    visuals = extract_visuals(summary)
    theme = extract_theme(summary)

    prompt = get_sdxl_prompt_from_openai(emotion, visuals, theme)

    filename = os.path.join(OUTPUT_DIR, f"poem_final{idx}.png")
    generate_image(prompt, filename=filename)

    print(f"Generated image for poem {idx}: {filename}")

    gc.collect()
    torch.mps.empty_cache()