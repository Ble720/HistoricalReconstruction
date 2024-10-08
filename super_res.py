from PIL import Image
from diffusers import DiffusionPipeline
import torch
import os

src = './chap4_src'
save = './save_200/'

device = "cuda" if torch.cuda.is_available() else "cpu"

# load model and scheduler


pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages")
pipeline = pipeline.to(device)

# let's download an  image
for r, d, f in os.walk(src):
    for file in f:
        if file.endswith(".jpg") or file.endswith(".jpeg"):
            subfolder = r[r.find('\\')+1:]
            if not os.path.exists(save + subfolder):
                os.makedirs(save + subfolder)

            f_path = os.path.join(r, file)
            s_path = save + subfolder + '/' + file[:file.rfind('.')+1] + 'jpg'

            low_res_img = Image.open(f_path).convert("RGB")
            low_res_img = low_res_img.resize((128, 128))

# run pipeline in inference (sample random noise and denoise)
            upscaled_image = pipeline(low_res_img, num_inference_steps=200, eta=1).images[0]
# save image
            upscaled_image.save(s_path)