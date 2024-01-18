from data.make_dataset import get_data_loaders
from models.model import MyTimmNet
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra

@hydra.main(version_base=None, config_path="../conf", config_name="config_1")
def main(cfg):
    train_loader,test_loader, class_names = get_data_loaders(cfg.paths.all_images ,cfg.params.batch_size)
    model = MyTimmNet(train_loader, test_loader, class_names, model_name=cfg.models.name, learning_rate=cfg.params.lr)
    trainer = pl.Trainer(max_epochs=cfg.params.epoch_count)  
    trainer.fit(model, train_loader, test_loader)
    trainer.test(model, test_loader)

if __name__ == "__main__":
    main()