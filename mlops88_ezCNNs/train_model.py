from data.make_dataset import get_data_loaders
from models.model import MyTimmNet
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
import wandb
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(log_model="all")

@hydra.main(version_base=None, config_path="../conf", config_name="config_1")
def main(cfg):
    
    wandb.init(
    project="Testing with RESNET18",
    )
    
    
    
    train_loader,test_loader, class_names = get_data_loaders(cfg.paths.all_images ,cfg.params.batch_size)
    model = MyTimmNet(train_loader, test_loader, class_names, model_name=cfg.models.name, learning_rate=cfg.params.lr)
    trainer = pl.Trainer(logger=wandb_logger,max_epochs=cfg.params.epoch_count)  
    trainer.fit(model, train_loader, test_loader)
    trainer.test(model, test_loader)
    
    wandb.finish()

if __name__ == "__main__":
    main()