import os

try:
    import dvc
except ImportError:
    print("dvc is not installed")

# dvc needs to be in the path
os.system("dvc pull")


if __name__ == '__main__':
    # Get the data and process it
    pass