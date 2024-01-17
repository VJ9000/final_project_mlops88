from data.make_dataset import load_data
from models.model import MyTimmNet
import pytorch_lightning as pl

if __name__ == "__main__":
    # Assuming you have train_loader and test_loader already defined
    
    train_loader,test_loader, class_names = load_data()
    model = MyTimmNet(train_loader, test_loader, num_classes=len(class_names), model_name='resnet18')

    trainer = pl.Trainer(max_epochs=5)  
    trainer.fit(model, train_loader, test_loader)
    trainer.test(model, test_loader)