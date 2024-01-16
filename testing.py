import torch
from sklearn.metrics import accuracy_score

device = 'cpu'

# Testing functions
def test_model(model, test_batch_gen):
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