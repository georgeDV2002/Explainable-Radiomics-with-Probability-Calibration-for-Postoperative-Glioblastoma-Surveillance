#!/usr/bin/env python3
"""
Concatenate two images (e.g., ROC + CM) side-by-side with optional scaling.
Defaults match lightGBM__5b.py outputs.

"""

from PIL import Image

def concat_images_side_by_side(img1_path="roc_plot_test__7.png",
                               img2_path="confusion_matrix__7.png",
                               output_path="side_by_side__7.png",
                               scale1=1.0, scale2=1.0, bg=(255, 255, 255)):
    # Open
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # Scale
    w1, h1 = img1.size
    w2, h2 = img2.size
    img1 = img1.resize((int(w1 * scale1), int(h1 * scale1)), Image.LANCZOS)
    img2 = img2.resize((int(w2 * scale2), int(h2 * scale2)), Image.LANCZOS)

    # Pad heights to match
    w1, h1 = img1.size
    w2, h2 = img2.size
    max_h = max(h1, h2)

    def pad_to_h(img, target_h):
        w, h = img.size
        if h == target_h:
            return img
        top = (target_h - h) // 2
        bottom = target_h - h - top
        canvas = Image.new("RGB", (w, target_h), bg)
        canvas.paste(img, (0, top))
        return canvas

    img1 = pad_to_h(img1, max_h)
    img2 = pad_to_h(img2, max_h)

    # Concatenate horizontally
    total_w = img1.width + img2.width
    combined = Image.new("RGB", (total_w, max_h), bg)
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width, 0))

    combined.save(output_path)
    print(f"[saved] {output_path}")

if __name__ == "__main__":
    # Defaults: put ROC left, CM right
    concat_images_side_by_side()

