from diffusers import StableDiffusionPipeline
import torch

def generate_image(prompt, output_path="output.png"):
    # Load the pipeline
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate the image
    image = pipe(prompt).images[0]
    image.save(output_path)
    return output_path
