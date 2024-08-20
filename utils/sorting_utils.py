import os

def sort_files_by_creation_time(folder_path):
    
    files = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            files.append((file_path, os.path.getctime(file_path)))  # Store file path and creation time

    sorted_files = sorted(files, key=lambda x: x[1])  # Sort files based on creation time
    sorted_file_paths = [file_path for file_path, _ in sorted_files]  # Extract file paths
    return sorted_file_paths

def sort_files_by_name(folder_path):

    files = os.listdir(folder_path)
    sorted_files = sorted(files, key=lambda x: int(x.split('.')[0]))
    file_paths = [os.path.join(folder_path, file) for file in sorted_files]
    return file_paths

def sort_files_by_name_byte_code(folder_path):
    files = os.listdir(folder_path)
    sorted_files = sorted(files, key=lambda x: int(x.split(b'.')[0]))
    file_paths = [os.path.join(folder_path, file.decode('utf-8')) for file in sorted_files]
    return file_paths


