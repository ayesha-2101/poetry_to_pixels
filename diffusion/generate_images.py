from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16).to("mps")

def generate_image(prompt, filename):
    image = pipe(prompt).images[0]
    image.save(f"outputs/images/{filename}.png")
