import os

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, 'memory', 'something.txt')

try:
    with open(file_path, 'r') as f:
        content = f.read()
        print("File content: ")
        print("---")
        print(content)
        print("---")

except FileNotFoundError:
    print(f"Oh no! I couldn't find the file at {file_path}")