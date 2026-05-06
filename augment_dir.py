import os
import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import albumentations as A
import cv2

def create_directories():
    paths = [
        'augmentation_data/train/images',
        'augmentation_data/train/labels',
        'augmentation_data/val/images',
        'augmentation_data/val/labels'
    ]
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def copy_train_files():
    source_img_dir = Path('./total_dataset_per_video/train/images')
    source_label_dir = Path('./total_dataset_per_video/train/labels')
    
    image_files = list(source_img_dir.glob('*.jpg'))
    print(f"Found {len(image_files)} train image files")

    copied_files = []
    for img_path in image_files:
        dest_img = Path('augmentation_data/train/images') / img_path.name
        shutil.copy2(img_path, dest_img)
        
        label_path = source_label_dir / f'{img_path.stem}.txt'
        dest_label = Path('augmentation_data/train/labels') / label_path.name
        
        if label_path.exists():
            shutil.copy2(label_path, dest_label)
            copied_files.append((str(dest_img), str(dest_label)))

    return copied_files

def copy_val_files():
    source_img_dir = Path('./total_dataset_per_video/val/images')
    source_label_dir = Path('./total_dataset_per_video/val/labels')
    
    image_files = list(source_img_dir.glob('*.jpg'))
    print(f"Found {len(image_files)} val image files")

    for img_path in image_files:
        dest_img = Path('augmentation_data/val/images') / img_path.name
        shutil.copy2(img_path, dest_img)
        
        label_path = source_label_dir / f'{img_path.stem}.txt'
        dest_label = Path('augmentation_data/val/labels') / label_path.name
        
        if label_path.exists():
            shutil.copy2(label_path, dest_label)

def copy_yaml():
    source_yaml = Path('./total_dataset_per_video/data.yaml')
    dest_yaml = Path('augmentation_data/data.yaml')
    
    if source_yaml.exists():
        with open(source_yaml, 'r') as f:
            lines = f.readlines()
        
        updated_lines = []
        for line in lines:
            if line.startswith('train:'):
                updated_lines.append('train: /content/augmentation_data/train/images\n')
            elif line.startswith('val:'):
                updated_lines.append('val: /content/augmentation_data/val/images\n')
            else:
                updated_lines.append(line)
        
        with open(dest_yaml, 'w') as f:
            f.writelines(updated_lines)

def adjust_brightness(image_path, label_path, bright=True):
    img = Image.open(image_path)
    enhancer = ImageEnhance.Brightness(img)
    factor = random.uniform(1.2, 2.0) if bright else random.uniform(0.2, 0.8)
    img_enhanced = enhancer.enhance(factor)
    
    suffix = 'bright' if bright else 'dark'
    new_img_path = str(Path(image_path).parent / f'{Path(image_path).stem}_{suffix}{Path(image_path).suffix}')
    new_label_path = str(Path(label_path).parent / f'{Path(label_path).stem}_{suffix}{Path(label_path).suffix}')
    
    img_enhanced.save(new_img_path)
    shutil.copy2(label_path, new_label_path)
    return new_img_path, new_label_path

def adjust_color(image_path, label_path):
    img = Image.open(image_path)
    contrast_factor = random.uniform(5.0, 20.0)
    color_factor = random.uniform(5.0, 20.0)
    
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    img = ImageEnhance.Color(img).enhance(color_factor)
    
    new_img_path = str(Path(image_path).parent / f'{Path(image_path).stem}_color{Path(image_path).suffix}')
    new_label_path = str(Path(label_path).parent / f'{Path(label_path).stem}_color{Path(label_path).suffix}')
    
    img.save(new_img_path)
    shutil.copy2(label_path, new_label_path)
    return new_img_path, new_label_path

def add_noise(image_path, label_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    
    mean = random.uniform(-10, 10)
    std_dev = random.uniform(10, 75)
    noise = np.random.normal(mean, std_dev, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    new_img_path = str(Path(image_path).parent / f'{Path(image_path).stem}_noise{Path(image_path).suffix}')
    new_label_path = str(Path(label_path).parent / f'{Path(label_path).stem}_noise{Path(label_path).suffix}')
    
    Image.fromarray(noisy_img).save(new_img_path)
    shutil.copy2(label_path, new_label_path)
    return new_img_path, new_label_path

def apply_blur(image_path, label_path):
    img = Image.open(image_path)
    blur_radius = random.uniform(1, 4)
    img_blur = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    new_img_path = str(Path(image_path).parent / f'{Path(image_path).stem}_blur{Path(image_path).suffix}')
    new_label_path = str(Path(label_path).parent / f'{Path(label_path).stem}_blur{Path(label_path).suffix}')
    
    img_blur.save(new_img_path)
    shutil.copy2(label_path, new_label_path)
    return new_img_path, new_label_path

def add_weather(image_path, label_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    transform = A.Compose([A.RandomFog(p=1.0)])
    augmented = transform(image=img_array)
    
    new_img_path = str(Path(image_path).parent / f'{Path(image_path).stem}_weather{Path(image_path).suffix}')
    new_label_path = str(Path(label_path).parent / f'{Path(label_path).stem}_weather{Path(label_path).suffix}')
    
    Image.fromarray(augmented['image']).save(new_img_path)
    shutil.copy2(label_path, new_label_path)
    return new_img_path, new_label_path

def add_compression(image_path, label_path):
    img = Image.open(image_path)
    quality = random.randint(10, 30)
    
    new_img_path = str(Path(image_path).parent / f'{Path(image_path).stem}_compress{Path(image_path).suffix}')
    new_label_path = str(Path(label_path).parent / f'{Path(label_path).stem}_compress{Path(label_path).suffix}')
    
    img.save(new_img_path, 'JPEG', quality=quality)
    shutil.copy2(label_path, new_label_path)
    return new_img_path, new_label_path

def add_mosaic(image_path, label_path):
    img = Image.open(image_path)
    width, height = img.size
    
    block_size = random.randint(1, 6)
    small_img = img.resize((width // block_size, height // block_size), Image.NEAREST)
    mosaic_img = small_img.resize((width, height), Image.NEAREST)
    
    new_img_path = str(Path(image_path).parent / f'{Path(image_path).stem}_mosaic{Path(image_path).suffix}')
    new_label_path = str(Path(label_path).parent / f'{Path(label_path).stem}_mosaic{Path(label_path).suffix}')
    
    mosaic_img.save(new_img_path)
    shutil.copy2(label_path, new_label_path)
    return new_img_path, new_label_path

def horizontal_flip(image_path, label_path):
    img = Image.open(image_path)
    img_flipped = ImageOps.mirror(img)
    
    with open(label_path, 'r') as f:
        labels = [line.strip().split() for line in f]
    
    adjusted_labels = []
    for label in labels:
        class_id = label[0]
        x_center = 1 - float(label[1])
        y_center = label[2]
        width = label[3]
        height = label[4]
        adjusted_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    new_img_path = str(Path(image_path).parent / f'{Path(image_path).stem}_hflip{Path(image_path).suffix}')
    new_label_path = str(Path(label_path).parent / f'{Path(label_path).stem}_hflip{Path(label_path).suffix}')
    
    img_flipped.save(new_img_path)
    with open(new_label_path, 'w') as f:
        f.write('\n'.join(adjusted_labels))
    
    return new_img_path, new_label_path

def vertical_flip(image_path, label_path):
    img = Image.open(image_path)
    img_flipped = ImageOps.flip(img)
    
    with open(label_path, 'r') as f:
        labels = [line.strip().split() for line in f]
    
    adjusted_labels = []
    for label in labels:
        class_id = label[0]
        x_center = label[1]
        y_center = 1 - float(label[2])
        width = label[3]
        height = label[4]
        adjusted_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    new_img_path = str(Path(image_path).parent / f'{Path(image_path).stem}_vflip{Path(image_path).suffix}')
    new_label_path = str(Path(label_path).parent / f'{Path(label_path).stem}_vflip{Path(label_path).suffix}')
    
    img_flipped.save(new_img_path)
    with open(new_label_path, 'w') as f:
        f.write('\n'.join(adjusted_labels))
    
    return new_img_path, new_label_path

def augment_image_and_label(image_path, label_path, augmentation_type):
    augmentation_functions = {
        'bright': lambda i, l: adjust_brightness(i, l, bright=True),
        'dark': lambda i, l: adjust_brightness(i, l, bright=False),
        'color': adjust_color,
        'noise': add_noise,
        'blur': apply_blur,
        'weather': add_weather,
        'compress': add_compression,
        'mosaic': add_mosaic,
        'hflip': horizontal_flip,
        'vflip': vertical_flip
    }
    
    if augmentation_type not in augmentation_functions:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")
    
    return augmentation_functions[augmentation_type](image_path, label_path)

def main():
    try:
        print("Starting augmentation process...")
        
        create_directories()
        
        copied_files = copy_train_files()
        copy_val_files()
        copy_yaml()
        
        print(f"Copied {len(copied_files)} original train files")
        
        augmentation_types = [
            'bright', 'dark', 'color', 'noise', 'blur',
            'weather', 'compress', 'mosaic', 'hflip', 'vflip'
        ]
        
        total_augmented = 0
        
        print(f"Applying {len(augmentation_types)} augmentations to each image...")
        for i, (img_path, label_path) in enumerate(copied_files):
            for aug_type in augmentation_types:
                try:
                    augment_image_and_label(img_path, label_path, aug_type)
                    total_augmented += 1
                    
                    if (total_augmented % 1000 == 0):
                        print(f"Progress: {total_augmented}/{len(copied_files)*len(augmentation_types)} augmentations")
                except Exception as e:
                    print(f"Error on {Path(img_path).stem} with {aug_type}: {str(e)}")
        
        print(f"Augmentation complete. Total augmented images: {total_augmented}")
        print(f"Total dataset size: {len(copied_files) + total_augmented} images")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()