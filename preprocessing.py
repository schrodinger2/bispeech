import os

DATA_DIR = "data/AR" 

# Step 1: delete all non-FLAC files
for filename in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, filename)

    if os.path.isfile(path) and not filename.lower().endswith(".flac"):
        os.remove(path)

# Step 2: rename FLAC files to 1.flac, 2.flac, ...
flac_files = sorted(
    f for f in os.listdir(DATA_DIR)
    if f.lower().endswith(".flac")
)

for idx, filename in enumerate(flac_files, start=1):
    old_path = os.path.join(DATA_DIR, filename)
    new_name = f"{idx+100}.flac" # i will be adding 100 files of egyptian arabic from another set later
    new_path = os.path.join(DATA_DIR, new_name)

    os.rename(old_path, new_path)

