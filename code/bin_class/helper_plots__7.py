#!/usr/bin/env python3
from PIL import Image

def concat_images_side_by_side(img1_path, img2_path, output_path="combined.png",
                               scale1=1.0, scale2=1.0):
    # Open images
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # Resize by scale factors
    w1, h1 = img1.size
    w2, h2 = img2.size
    img1 = img1.resize((int(w1 * scale1), int(h1 * scale1)), Image.LANCZOS)
    img2 = img2.resize((int(w2 * scale2), int(h2 * scale2)), Image.LANCZOS)

    # Update sizes after resize
    w1, h1 = img1.size
    w2, h2 = img2.size

    # Match heights by symmetric padding
    max_h = max(h1, h2)

    def pad_image(img, target_h):
        w, h = img.size
        if h == target_h:
            return img
        top_pad = (target_h - h) // 2
        bottom_pad = target_h - h - top_pad
        new_img = Image.new("RGB", (w, target_h), (255, 255, 255))  # white background
        new_img.paste(img, (0, top_pad))
        return new_img

    img1 = pad_image(img1, max_h)
    img2 = pad_image(img2, max_h)

    # Concatenate horizontally
    total_w = img1.width + img2.width
    combined = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width, 0))

    # Save
    combined.save(output_path)
    print(f"Saved combined image as {output_path}")

# Example usage:
# make first image half-size, second image 1.2x bigger
concat_images_side_by_side("roc_plot_test__5.png", "confusion_matrix.png",
                           "side_by_side.png", scale1=0.5, scale2=1.2)

