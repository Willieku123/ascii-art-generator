# coding:utf-8
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cupy as np
import numpy as real_np
import string
from scipy import signal
import time
import argparse


def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height


def text_to_image(input_text, image_size, dark, language):
    image = Image.new("RGB", image_size, "black" if dark else "white")
    draw = ImageDraw.Draw(image)
    
    if language == "mandarin":
        font = ImageFont.truetype("mingliu.ttc", 9)
    elif language == "english":
        font = ImageFont.truetype("consola.ttf", 11)
    text_width, text_height = textsize("哈", font)
    
    x = (image_size[0] - text_width) // 2
    y = (image_size[1] - text_height) // 2

    text_color = (255, 255, 255) if dark else (0, 0, 0)

    draw.text((x, y), input_text, font=font, fill=text_color)
    
    return np.array(image).astype(float)  # H, W, C


def show_image(img_np, bias=False):
    a = Image.fromarray(real_np.array(img_np.get()))
    a.show(a)


def remap_by_occurrence(arr):
    arr = real_np.asarray(arr.get())
    # Count occurrences of each integer in the array
    unique_values, counts = real_np.unique(arr, return_counts=True)

    # Sort based on occurrences
    sorted_indices = real_np.argsort(counts)[::-1]
    sorted_values = unique_values[sorted_indices]

    # Create a mapping dictionary based on sorted order
    remap_dict = {value: i for i, value in enumerate(sorted_values)}

    # Use the mapping to remap the array
    remapped_arr = real_np.vectorize(remap_dict.get)(arr)

    return np.asarray(remapped_arr)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path', type=str, default="wiki.png")  #, required=True
    parser.add_argument('--char', type=str, default=string.printable[:-5])

    parser.add_argument('--language', type=str, default="english", choices=["english", "mandarin"])
    parser.add_argument('--dark', action="store_true")
    parser.add_argument('--size_factor', type=int, default=8)
    args = parser.parse_args()


    #args.char = 
    #args.char = " ASCIRT"
    #args.char = "星期一清大沒有停電，交又贏"
    #args.path = 'image3.jpg'
    #args.path = 'image4.png'
    #args.path = 'image6.jpg'
    #args.path = 'ml.jpg'
    #args.path = 'ascii_art.png'



    # get char list
    char_list = [i for i in args.char]

    if args.language == "mandarin":
        PATCH_H = 9
        PATCH_W = 9
    elif args.language == "english":
        PATCH_H = 12
        PATCH_W = 6
    else:
        raise NotImplementedError(f"'{args.language}' language not implemented")


    # get normalized char image patches
    char_img_list = [text_to_image(char,(PATCH_W,PATCH_H), args.dark, args.language)[..., 0] for char in char_list]
    b = np.tile(np.concatenate(char_img_list, axis=1), (1, 2, 1))
    char_max = max([i.mean() for i in char_img_list])
    char_min = min([i.mean() for i in char_img_list])

    # get normalized target image
    img = Image.open(args.path).convert('L')
    img = img.resize((PATCH_H*PATCH_W*args.size_factor, PATCH_H*(PATCH_W*args.size_factor * img.size[1] // img.size[0]) ))#.filter(ImageFilter.FIND_EDGES)
    img.show(img)
    target_img = np.array(img).astype(float)


    # get closest char patches for each target image patches
    H, W = target_img.shape
    H = H // PATCH_H
    W = W // PATCH_W

    output_img = np.zeros_like(target_img)

    #'''
    start_time = time.time()
    losses = []
    for char_patch in char_img_list:
        target_img_norm = target_img / 255
        char_patch_norm = (char_patch-char_min)/(char_max-char_min)
        unit_mean = np.ones_like(char_patch_norm) * char_patch_norm.mean()
        
        # per patch mean
        per_patch_mean = real_np.abs( signal.correlate2d(target_img_norm.get(), np.ones_like(char_patch_norm).get(), mode='same', boundary='symm') - char_patch_norm.sum().get())
        
        # per pixel MSE
        per_pixel_MSE =  - signal.correlate2d((1-target_img_norm).get(), (1-char_patch_norm).get(), mode='same', boundary='symm') - signal.correlate2d(target_img_norm.get(), char_patch_norm.get(), mode='same', boundary='symm')
        
        loss = per_patch_mean + per_pixel_MSE * 2

        losses.append(np.asarray(loss))
    loss_map = np.stack(losses, axis=0)
    decision_map = np.argmin(loss_map, axis=0)


    for i in range(H):
        for j in range(W):
            H_start = i*PATCH_H + PATCH_H //3
            H_end = (i+1)*PATCH_H - PATCH_H //3
            W_start = j*PATCH_W + PATCH_W //3
            W_end = (j+1)*PATCH_W - PATCH_W //3

            decision = decision_map[H_start : H_end, W_start : W_end]
            occurence = remap_by_occurrence(decision).flatten()
            decision = decision.flatten()[np.argmax(occurence)]

            output_img[i*PATCH_H : (i+1)*PATCH_H, j*PATCH_W : (j+1)*PATCH_W] = char_img_list[int(decision)]
    print(f"elapsed time {time.time() - start_time} seconds.")
    #'''


    show_image(output_img)