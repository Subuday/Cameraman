import torch
from controlnet_aux import OpenposeDetector
from torch import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
import random
from diffusers.utils import load_image

# Change these as you want:
model_path = "output/c1-s6"
controlnet_path = "lllyasviel/control_v11p_sd15_openpose"
img_out_folder = "./image_output"

# Image related config -- Change if you've used a different keyword:

image = load_image("https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/images/input.png")
processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
control_image = processor(image, hand_and_face=True)
control_image.save("./image_output/control.png")

# Try our original prompt and make sure it works okay:
# prompt = "closeup photo of ggd woman in the garden on a bright sunny day"
prompt = "hyperrealistic portrait of sks with long wavy hair, in modern cafe, urban photography, photo in nice cozy cafe in the city center, sultry flirty look, staring at camera, beautiful beautiful symmetrical face, cute natural makeup, soft natural light, intimate portrait composition, professional photo, highly detailed, portrait photography, photo-realism, professionally color graded, trending on instagram, shot on iphone 14 pro max, high resolution, 50.0mm, f/3.2, 8k, unsplash"
negative_prompt = "deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4, text, huge eyes, strabismus, asymmetrical pupils, plump face, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
num_samples = 20
guidance_scale = 7
controlnet_conditioning_scale=1
num_inference_steps = 150
height = 512
width = 512

controlnet = ControlNetModel.from_pretrained(controlnet_path)


# Setup the scheduler and pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_path) #StableDiffusionControlNetPipeline.from_pretrained(model_path, controlnet=controlnet)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()


# Generate the images:
device = "cpu"

seed1 = 21
generator1 = torch.Generator(device).manual_seed(seed1)

seed2 = 144
generator2 = torch.Generator(device).manual_seed(seed2)

images1 = pipe(
        prompt,
        # image=image,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator1
        # controlnet_conditioning_scale=controlnet_conditioning_scale
    ).images
    
images2 = pipe(
        prompt,
        # image=image,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator2
        # controlnet_conditioning_scale=controlnet_conditioning_scale
    ).images
    
# Loop on the images and save them:
for img in images1:
    i = random.randint(0, 200)
    img.save(f"{img_out_folder}/v2_{i}.png")
    
for img in images2:
    i = random.randint(0, 200)
    img.save(f"{img_out_folder}/v2_{i}.png")
