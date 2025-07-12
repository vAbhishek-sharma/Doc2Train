# In doc2train/utils/common.py
import os

def save_image_data(img_data: bytes, output_dir: str, base_name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, f"{base_name}.png")
    with open(img_path, "wb") as f:
        f.write(img_data)
    return img_path
