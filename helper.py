import os
import shutil
from pathlib import Path
from PIL import Image

def copy_images_evenly_sampled():
    """
    Copy images from FF_faces to FF_faces_simplified with:
    - 5 evenly sampled images per person
    - Flattened structure (no subfolders)
    - Global sequential naming (1.png, 2.png, 3.png, ...)
    - Resized from 256x256 to 128x128
    """
    
    source_root = Path("data/FF_faces")
    dest_root = Path("data/FF_faces_simplified")
    
    # Define the 6 classes
    classes = {
        'real': 'Real',
        'fake/Deepfakes': 'Deepfakes',
        'fake/Face2Face': 'Face2Face',
        'fake/FaceShifter': 'FaceShifter',
        'fake/FaceSwap': 'FaceSwap',
        'fake/NeuralTextures': 'NeuralTextures'
    }
    
    # Process each class
    for source_path, dest_name in classes.items():
        print(f"\nProcessing {dest_name}...")
        source_dir = source_root / source_path
        dest_dir = dest_root / dest_name
        
        # Create destination directory if it doesn't exist
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Counter for naming (local to each class - starts at 1)
        image_counter = 1
        
        # Get all person folders
        person_folders = sorted([f for f in source_dir.iterdir() if f.is_dir()])
        
        for person_folder in person_folders:
            # Get all PNG image files (sorted by name)
            images = sorted([
                f for f in person_folder.iterdir() 
                if f.suffix.lower() == '.png'
            ])
            
            if len(images) == 0:
                continue
            
            # Sample 5 images evenly spread
            num_images = len(images)
            if num_images <= 5:
                # If 5 or fewer images, take all
                sampled_images = images
            else:
                # Evenly sample 5 images (indices: 0, 7, 14, 21, 29 for 30 images)
                indices = [int(i * (num_images - 1) / 4) for i in range(5)]
                sampled_images = [images[i] for i in indices]
            
            # Resize and save images
            for img in sampled_images:
                new_name = f"{image_counter}.png"
                dest_path = dest_dir / new_name
                
                # Open image, resize from 256x256 to 128x128 using LANCZOS (highest quality for downsizing)
                with Image.open(img) as im:
                    resized_im = im.resize((128, 128), Image.Resampling.LANCZOS)
                    resized_im.save(dest_path)
                
                image_counter += 1
            
            # Print progress every 250 images
            if image_counter % 250 == 0:
                print(f"  Processed {image_counter} images...")
        
        total_images = len(list(dest_dir.iterdir()))
        print(f"  ✓ Completed {dest_name}: {total_images} images copied and resized to 128x128")
    
    print("\n" + "="*60)
    print("All classes processed successfully!")
    print("="*60)
    
    # Print summary
    print("\nSummary:")
    for dest_name in classes.values():
        count = len(list((dest_root / dest_name).iterdir()))
        print(f"  {dest_name}: {count} images")

if __name__ == "__main__":
    copy_images_evenly_sampled()
