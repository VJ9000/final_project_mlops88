import os
import shutil
from data import prepare_data_dirs, TRAIN_DIR, VAL_DIR

def test_prepare_data_dirs_creates_directories():
    # Call the function to test
    prepare_data_dirs()

    # Check if the training and validation directories are created
    assert os.path.exists(TRAIN_DIR)
    assert os.path.exists(VAL_DIR)

    # Clean up the created directories after the test is done
    shutil.rmtree(TRAIN_DIR, ignore_errors=True)
    shutil.rmtree(VAL_DIR, ignore_errors=True)