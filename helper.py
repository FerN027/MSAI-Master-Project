import os

def deleteHalf():
    root = 'data/ASVspoof_LA/test_2'
    # Inside two folders: real and fake
    
    subfolders = ['real', 'fake']
    
    for subfolder in subfolders:
        folder_path = os.path.join(root, subfolder)
        
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist, skipping...")
            continue
        
        # Get all audio files in the subfolder
        files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        
        print(f"\nProcessing {subfolder} folder...")
        print(f"Total files before deletion: {len(files)}")
        
        deleted_count = 0
        # Delete every other file (indices 1, 3, 5, 7, ...)
        for i in range(1, len(files), 2):
            file_path = os.path.join(folder_path, files[i])
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        
        remaining_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        print(f"Files deleted: {deleted_count}")
        print(f"Files remaining: {remaining_files}")

if __name__ == "__main__":
    response = input("This will delete half of the audio files in data/ASVspoof_LA/test_2/real and fake folders. Continue? (yes/no): ")
    if response.lower() == 'yes':
        deleteHalf()
        print("\nDone!")
    else:
        print("Operation cancelled.")