from PIL import Image
from diffusers import DiffusionPipeline
import torch
import os
import argparse


def resize(img: Image.Image):
    img.thumbnail((128, 128), Image.BILINEAR)

    # Create a new blank image with padding
    new_img = Image.new("RGB", (128, 128))

    # Paste the original image onto the top-left corner
    new_img.paste(img, (0, 0))

    return new_img

def crop_super_res_output(img: Image.Image, orig_size: tuple) -> Image.Image:
    orig_w, orig_h = orig_size
    crop_w, crop_h = orig_w * 4, orig_h * 4

    return img.crop((0, 0, crop_w, crop_h))

def run(src, save, device='cuda'):
    pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages")
    pipeline = pipeline.to(device)

    # let's download an  image
    for r, d, f in os.walk(src):
        for file in f:
            if file.endswith(".jpg") or file.endswith(".jpeg"):
                
                f_path = os.path.join(r, file)
                s_path = save + '/' + file[:file.rfind('.')+1] + 'jpg'

                low_res_img = Image.open(f_path).convert("RGB")
                width, height = low_res_img.size

                low_res_img = resize(low_res_img)
                #low_res_img.save(s_path)

    # run pipeline in inference (sample random noise and denoise)
                print('before: ', low_res_img.size)
                upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
                print('after: ', upscaled_image.size)
                upscaled_image = crop_super_res_output(upscaled_image, (width, height))
                upscaled_image.save(s_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='path to source images')
    parser.add_argument('-s', '--save', help='save path')
    parser.add_argument('-d', '--device', help='device', default='cpu')
    args = parser.parse_args()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    run(args.path, args.save, args.device)