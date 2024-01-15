import torchvision
import torch
import os

from data.py import val_transforms, model
from sklearn.metrics import accuracy_score

batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TEST_DIR = os.path.join(DATA_DIR, 'test')

test_dataset = torchvision.datasets.ImageFolder(TEST_DIR, transform=val_transforms)
test_batch_gen = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


state = torch.load(checkpoint_path)
model.load_state_dict(state['model_state_dict'])

model.eval()

y_pred = []
y_true = []
with torch.no_grad():
    for X_batch, y_batch in test_batch_gen:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
                
        logits = model(X_batch)
        y_pred += logits.max(1)[1].detach().cpu().numpy().tolist()
        y_true += y_batch.cpu().numpy().tolist()
        
final_accuracy = accuracy_score(y_pred, y_true)
print('Final test accuracy: {:.2f} %'.format(final_accuracy * 100))