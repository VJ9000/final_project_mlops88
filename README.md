# Project Description

#### Overall Goal of the Project:

The primary goal of this project is to develop a robust image recognition system capable of accurately identifying various fruits and vegetables. Leveraging deep learning techniques, the project aims to build a model that can categorize images into specific food categories, distinguishing between a wide array of fruits and vegetables.

#### Framework and Integration:

PyTorch is the main deep learning framework used in the course, will serve as the backbone for this project. PyTorch's extensive capabilities in handling image data and constructing neural networks make it an ideal choice. The integration of PyTorch will involve leveraging its pre-trained image models, allowing for efficient and effective model training and deployment aswell as enabling us to use the framework [pytorch-image-models](https://github.com/huggingface/pytorch-image-models).

#### Data for Analysis:

The project will utilize the "Fruits and Vegetables Image Recognition Dataset" sourced from Kaggle. This dataset contains images of numerous fruits and vegetables, spanning categories such as bananas, apples, carrots, tomatoes, and many more. The dataset is organized into three primary folders - train, test, and validation - each containing images of 30 different food items. These images, totaling 1200 across all categories, will serve as the foundational dataset for training and validating the image recognition model. The dataset also furfills the requirement of size less than 10GB. Can be found [here](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition/data)

#### Expected Models:

In utilizing the PyTorch framework, we will be looking into convolutional neural network (CNN) architectures. Specifically, models like ResNet, DenseNet, or EfficientNet, among others available within PyTorch's model zoo, are anticipated for their robustness in image classification tasks. These models will undergo fine-tuning or transfer learning approaches to adapt to the specifics of the fruits and vegetables dataset, facilitating accurate recognition and classification. We have named our project mlops88_ezCNNs for (group 88 in mlops easy CNNs)

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── mlops88_ezCNNs  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
