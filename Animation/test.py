import numpy as np
from PIL import Image, ImageDraw
import os
import random

def create_synthetic_dataset(output_dir="test_datasets", num_images_per_stage=50, image_size=64):
    """Create synthetic test datasets with three distinct stages"""
    
    # Create output directories
    stage_dirs = [
        os.path.join(output_dir, "stage_0_circles"),
        os.path.join(output_dir, "stage_1_squares"), 
        os.path.join(output_dir, "stage_2_triangles")
    ]
    
    for stage_dir in stage_dirs:
        os.makedirs(stage_dir, exist_ok=True)
    
    # Generate stage 0: Circles
    print("Generating circles (stage 0)...")
    for i in range(num_images_per_stage):
        img = Image.new('RGB', (image_size, image_size), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw random circles
        for _ in range(random.randint(3, 8)):
            x = random.randint(10, image_size-10)
            y = random.randint(10, image_size-10)
            radius = random.randint(5, 15)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
        
        img.save(os.path.join(stage_dirs[0], f"circle_{i:03d}.png"))
    
    # Generate stage 1: Squares
    print("Generating squares (stage 1)...")
    for i in range(num_images_per_stage):
        img = Image.new('RGB', (image_size, image_size), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw random squares
        for _ in range(random.randint(3, 8)):
            x = random.randint(5, image_size-20)
            y = random.randint(5, image_size-20)
            size = random.randint(8, 18)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.rectangle([x, y, x+size, y+size], fill=color)
        
        img.save(os.path.join(stage_dirs[1], f"square_{i:03d}.png"))
    
    # Generate stage 2: Triangles
    print("Generating triangles (stage 2)...")
    for i in range(num_images_per_stage):
        img = Image.new('RGB', (image_size, image_size), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw random triangles
        for _ in range(random.randint(3, 8)):
            x = random.randint(10, image_size-10)
            y = random.randint(10, image_size-10)
            size = random.randint(8, 15)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            points = [
                (x, y-size),
                (x-size, y+size),
                (x+size, y+size)
            ]
            draw.polygon(points, fill=color)
        
        img.save(os.path.join(stage_dirs[2], f"triangle_{i:03d}.png"))
    
    print(f"Dataset created in '{output_dir}' with {num_images_per_stage} images per stage")
    return stage_dirs

# Create the synthetic dataset
test_dirs = create_synthetic_dataset()