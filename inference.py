import torch
from controlnet_aux import OpenposeDetector
from torch import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import random
from diffusers.utils import load_image

# Change these as you want:
model_path = "./output/2500"
controlnet_path = "lllyasviel/control_v11p_sd15_openpose"
img_out_folder = "./image_output"

# Image related config -- Change if you've used a different keyword:

image = load_image("./pose.png")
processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
control_image = processor(image, hand_and_face=True)
control_image.save("./image_output/control.png")

# Try our original prompt and make sure it works okay:
# prompt = "closeup photo of ggd woman in the garden on a bright sunny day"
prompt = "a portrait photo of a sks person, winter clothe, ((sultry flirty look)), cute natural makeup, ((standing outside in snowy city street)), ultra realistic, elegant, highly detailed, intricate, sharp focus, depth of field, f/1. 8, 85mm, (centered image composition), (professionally color graded), ((bright soft diffused light)), volumetric fog, trending on instagram, trending on tumblr, hdr 4k, 8k"
negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
num_samples = 10
guidance_scale = 7
controlnet_conditioning_scale=0.3
num_inference_steps = 25
height = 512
width = 512

controlnet = ControlNetModel.from_pretrained(controlnet_path)


# Setup the scheduler and pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(model_path, controlnet=controlnet)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()


# Generate the images:
images = pipe(
        prompt,
        image=image,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale
    ).images

# Loop on the images and save them:
for img in images:
    i = random.randint(0, 200)
    img.save(f"{img_out_folder}/v2_{i}.png")
