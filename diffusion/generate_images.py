from diffusers import StableDiffusionPipeline
import torch 
import gc

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",torch_dtype=torch.float32).to("mps")
pipe.enable_attention_slicing()

pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))


def generate_image(prompt, filename):
    image = pipe(prompt, height=256, width=256).images[0]
    image.save(filename)
    torch.mps.empty_cache()
    gc.collect()
