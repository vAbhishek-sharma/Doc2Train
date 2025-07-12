# utils/image_save.py
import os

def save_image_data(img_data: bytes, output_dir: str, base_name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)

    # Check if base_name already has a valid extension
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}
    root, ext = os.path.splitext(base_name)
    if ext.lower() in valid_exts:
        filename = base_name
    else:
        filename = base_name + ".png"

    img_path = os.path.join(output_dir, filename)
    with open(img_path, "wb") as f:
        f.write(img_data)
    return img_path
