import os

try:
    import dvc
except ImportError:
    print("dvc is not installed")

# dvc needs to be in the path
os.system("dvc pull")