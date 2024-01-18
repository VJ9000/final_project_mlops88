import torch
from models.model import MyTimmNet
from data.make_dataset import get_data_loaders
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import hydra
import wandb
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(log_model="all")


def predict_single_image(model, image_path):
    """Run prediction for a single image using the given model.

    Args:
        model (MyTimmNet): The trained model to use for prediction.
        image_path (str): The file path to the image.

    Returns:
        Tuple[Tensor, str, Image]: A tuple containing the model predictions,
        the predicted class name, and the input image with added prediction text.
    """
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    image = image.resize((256, 256))

    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_class = torch.max(output, 1)
    predicted_class_name = model.class_names[predicted_class.item()]

    image_with_text = add_text_to_image(image, predicted_class_name)
    return output, predicted_class_name, image_with_text


def add_text_to_image(image, text):
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, size=12)
    draw.text((0, 0), f"Predicted: {text}", fill="blue", font=font)
    return image


@hydra.main(version_base=None, config_path="../conf", config_name="config_test")
def main(cfg):
    wandb.init(project="Predicting with RESNET18")
    model = MyTimmNet.load_from_checkpoint(cfg.paths.checkpoint_path)
    prediction, predicted_class, image_with_prediction = predict_single_image(
        model, cfg.paths.testing_image
    )
    wandb.log({"image": [wandb.Image(image_with_prediction, caption="Testing Image")]})
    wandb.finish()


if __name__ == "__main__":
    main()
