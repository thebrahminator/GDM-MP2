import numpy as np
import pandas as pd
import pickle
import csv
import os
import torch
from sklearn.preprocessing import LabelEncoder
from dataset_binary import YCBDataset
from torch_geometric.data import DataLoader
from network import RecNet, train, Rec2
from eval import auc
np.random.seed(42)


if __name__ == '__main__':

    clicks_dataset = pd.read_csv('../dataset/yoochoose-clicks.dat',names=['session_id', 'timestamp', 'item_id', 'caetgory'])
    buy_dataset = pd.read_csv('../dataset/yoochoose-buys.dat', names=['session_id', 'timestamp', 'item_id', 'price',
                                                                      'quantity'])

    print('Finished Reading Dataset...')
    clicks_dataset['valid_session'] = clicks_dataset.session_id.map(clicks_dataset.
                                                                    groupby('session_id')['item_id'].size() > 2)
    clicks_dataset = clicks_dataset.loc[clicks_dataset.valid_session].drop('valid_session', axis=1)

    sampled_session_id = np.random.choice(clicks_dataset.session_id.unique(), 200000, replace=False)
    clicks_dataset = clicks_dataset.loc[clicks_dataset.session_id.isin(sampled_session_id)]

    print('Picked 200000 random samples...')
    clicks_dataset.groupby('session_id')['item_id'].size().mean()
    enconder = LabelEncoder()
    clicks_dataset['item_id'] = enconder.fit_transform(clicks_dataset.item_id)
    clicks_dataset['label'] = clicks_dataset.session_id.isin(buy_dataset.session_id)

    mean_ = clicks_dataset.drop_duplicates('session_id')['label'].mean()
    print(mean_)

    clicks_dataset.to_csv('temp.csv', index=False)
    dataset = YCBDataset(root='../', dataset=clicks_dataset)
    dataset = dataset.shuffle()
    x = 160000
    y = 180000
    train_dataset = dataset[:x]
    val_dataset = dataset[x:y]
    test_dataset = dataset[y:]
    batch_size = 256
    train_data = DataLoader(train_dataset, batch_size=batch_size)
    test_data = DataLoader(test_dataset, batch_size=batch_size)
    val_data = DataLoader(val_dataset, batch_size=batch_size)

    print("Finished Splitting Dataset")
    model = Rec2().to('cpu')

    print("Starting Run of the Network")
    for epoch in range(10):
        loss = train(model, train_data)
        train_accuracy = auc(model, train_data)
        eval_accurancy = auc(model, val_data)
        test_accuracy = auc(model, test_data)
        print(f"Epoch: {epoch + 1}; Loss: {loss}; Train Acc: {train_accuracy}, Val Acc: {eval_accurancy}, "
              f"Test Acc: {test_accuracy}")