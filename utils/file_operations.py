# file_operations.py
import os
import shutil

def delete_files_with_classes(folder_path: str) -> None:

    for filename in os.listdir(folder_path):
        if 'classes' in filename:
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
            print(f"Deleted: {file_path}                    ", end='\r', flush=True)


def moving_images(source_directory: str, destination_directory: str) -> None:
    
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    for filename in os.listdir(source_directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            source_path = os.path.join(source_directory, filename)
            destination_path = os.path.join(destination_directory, filename)
            shutil.move(source_path, destination_path)
            print(f"Moved: {source_path} => {destination_path}                  ", end='\r', flush=True)

    if os.path.exists(source_directory):
        shutil.rmtree(source_directory)
