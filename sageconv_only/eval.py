from sklearn.metrics import roc_auc_score
import torch
import numpy as np


def auc(model, load):
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for data in load:
            data = data.to('cpu')
            pred = model(data).detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)

    predictions = np.hstack(predictions)
    labels = np.hstack(labels)

    return roc_auc_score(labels, predictions)
