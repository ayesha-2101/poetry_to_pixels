from openai import OpenAI
import time 

client = OpenAI(api_key="api_key")

def tune_prompt(emotion, visuals, theme):
    visuals_joined = ", ".join(visuals[:3])
    
    prompt = (
        f"The central theme is '{theme}', capturing the emotion of {emotion} through visual elements like {visuals_joined}. "
        f"This scene should evoke a strong sense of {emotion}, using imagery that is both symbolic and expressive. "
        f"Imagine a detailed painting that combines the emotional tone with vivid scenery, using elements such as {visuals_joined} "
        f"to represent the essence of '{theme}'. Render this with high realism, dramatic lighting, and a cinematic atmosphere."
    )
    
    return prompt


def get_sdxl_prompt_from_openai(emotion, visuals, theme):
    base_prompt = tune_prompt(emotion, visuals, theme)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a prompt engineer skilled in crafting vivid, artistic prompts for Stable Diffusion XL."},
            {"role": "user", "content": f"Given the emotion: '{emotion}', visuals: {visuals}, and theme: '{theme}', write a visual prompt suitable for SDXL image generation. Here's a base to build on: {base_prompt}"}
        ],
        temperature=0.7,
        max_tokens=200
    )

    return response.choices[0].message.content.strip()
