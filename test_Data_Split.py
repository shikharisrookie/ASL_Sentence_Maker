import os, shutil
from sklearn.model_selection import train_test_split

BASE_DIR = r"Z:\Shikhar\ASL\Sign-Language-Recognition\asl_alphabet_train\asl_alphabet_train"
TEST_DIR = r"Z:\Shikhar\ASL\Sign-Language-Recognition\asl_alphabet_test\asl_alphabet_test"

if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)

    for class_name in os.listdir(BASE_DIR):
        class_path = os.path.join(BASE_DIR, class_name)
        if not os.path.isdir(class_path): 
            continue

        files = os.listdir(class_path)
        # Take 10% of images for test
        _, test_files = train_test_split(files, test_size=0.1, random_state=42)

        # Create subfolder for this class
        split_path = os.path.join(TEST_DIR, class_name)
        os.makedirs(split_path, exist_ok=True)

        for f in test_files:
            shutil.copy(os.path.join(class_path, f), os.path.join(split_path, f))

