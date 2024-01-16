# coding:utf-8
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import string
from scipy import signal
import time
import argparse
import os


def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height


def text_to_image(input_text, image_size, dark, language, font_size=9):
    image = Image.new("RGB", image_size, "black" if dark else "white")
    draw = ImageDraw.Draw(image)
    
    if language == "mandarin":
        font = ImageFont.truetype("mingliu.ttc", font_size)
    elif language == "english":
        font = ImageFont.truetype("consola.ttf", font_size)
    text_width, text_height = textsize("哈", font)
    
    x = (image_size[0] - text_width) // 2
    y = (image_size[1] - text_height) // 2

    text_color = (255, 255, 255) if dark else (0, 0, 0)

    draw.text((x, y), input_text, font=font, fill=text_color)
    
    return np.array(image).astype(float)  # H, W, C


def show_image(img_np, bias=False):
    a = Image.fromarray(np.array(img_np))
    a.show(a)


def remap_by_occurrence(arr):
    # Count occurrences of each integer in the array
    unique_values, counts = np.unique(arr, return_counts=True)

    # Sort based on occurrences
    sorted_indices = np.argsort(counts)[::-1]
    sorted_values = unique_values[sorted_indices]

    # Create a mapping dictionary based on sorted order
    remap_dict = {value: i for i, value in enumerate(sorted_values)}

    # Use the mapping to remap the array
    remapped_arr = np.vectorize(remap_dict.get)(arr)

    return remapped_arr


def get_loss_map(target_img, char_img_list):
    char_max = max([i.mean() for i in char_img_list])
    char_min = min([i.mean() for i in char_img_list])

    losses = []

    for char_patch in char_img_list:
        target_img_norm = target_img / 255
        char_patch_norm = (char_patch-char_min)/(char_max-char_min)
        
        # per patch mean
        per_patch_mean = np.abs( signal.correlate2d(target_img_norm, np.ones_like(char_patch_norm), mode='same', boundary='symm') - char_patch_norm.sum())
        
        # per pixel MSE
        per_pixel_MSE =  - signal.correlate2d((1-target_img_norm), (1-char_patch_norm), mode='same', boundary='symm') - signal.correlate2d(target_img_norm, char_patch_norm, mode='same', boundary='symm')
        
        loss = per_patch_mean + per_pixel_MSE * 2

        losses.append(loss)

    return np.stack(losses, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path', type=str, default="input/cats.jpg")  #, required=True

    parser.add_argument('--output_path', type=str, default="output/output.png")
    parser.add_argument('--output_txt', type=str, default="output/output.txt")

    parser.add_argument('--language', type=str, default="english", choices=["english", "mandarin"])
    parser.add_argument('--char', type=str, default=string.printable[:-5])
    parser.add_argument('--dark', action="store_true")
    parser.add_argument('--size_factor', type=int, default=8)
    parser.add_argument('--output_size_factor', type=int, default=2)
    args = parser.parse_args()


    start_time = time.time()


    # get list of char image patches
    if args.language == "mandarin":
        PATCH_H = 9
        PATCH_W = 9
    elif args.language == "english":
        PATCH_H = 12
        PATCH_W = 6
    else:
        raise NotImplementedError(f"'{args.language}' language not implemented")

    char_list = [i for i in args.char]
    char_img_list = [text_to_image(char,(PATCH_W,PATCH_H), args.dark, args.language)[..., 0] for char in char_list]
    char_img_list_big = [text_to_image(char, (PATCH_W * args.output_size_factor, PATCH_H * args.output_size_factor), args.dark, args.language, 9 * args.output_size_factor)[..., 0] for char in char_list]
    print(f"Char list: {args.char}")    


    # get resized target image
    img = Image.open(args.path).convert('L')
    img = img.resize((PATCH_H*PATCH_W*args.size_factor, PATCH_H*(PATCH_W*args.size_factor * img.size[1] // img.size[0]) ))#.filter(ImageFilter.FIND_EDGES)
    target_img = np.array(img).astype(float)
    
    output_img = np.zeros_like(np.tile(target_img, (args.output_size_factor, args.output_size_factor)))



    loss_map = get_loss_map(target_img, char_img_list)
    decision_map = np.argmin(loss_map, axis=0)

    if not os.path.exists(os.path.dirname(args.output_txt)):
        os.makedirs(os.path.dirname(args.output_txt))
    f = open(args.output_txt, 'w')

    for i in range(target_img.shape[0] // PATCH_H):
        for j in range(target_img.shape[1] // PATCH_W):
            H_start = i*PATCH_H + PATCH_H //3
            H_end = (i+1)*PATCH_H - PATCH_H //3
            W_start = j*PATCH_W + PATCH_W //3
            W_end = (j+1)*PATCH_W - PATCH_W //3

            decision = decision_map[H_start : H_end, W_start : W_end]
            occurence = remap_by_occurrence(decision).flatten()
            decision = int(decision.flatten()[np.argmax(occurence)])

            f.write(args.char[decision])
            output_img[i*PATCH_H*args.output_size_factor : (i+1)*PATCH_H*args.output_size_factor, j*PATCH_W*args.output_size_factor : (j+1)*PATCH_W*args.output_size_factor] = char_img_list_big[decision]
        f.write('\n')
    f.close()

    out = Image.fromarray(output_img.astype(np.uint8))
    out.save(args.output_path)
    #show_image(out)

    print(f"elapsed time {time.time() - start_time} seconds.")