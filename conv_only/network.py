from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import torch
import pandas as pd


class Rec2(torch.nn.Module):
    def __init__(self):
        super(Rec2, self).__init__()

        dataset = pd.read_csv('temp.csv')
        self.conv1 = GraphConv(256, 256)  # 128
        self.pool1 = TopKPooling(256, ratio=0.8)  # 128
        self.conv2 = GraphConv(256, 256)  # 128
        self.pool2 = TopKPooling(256, ratio=0.8)  # 128
        self.conv3 = GraphConv(256, 256)  # 128
        self.pool3 = TopKPooling(256, ratio=0.8)  # 128
        self.conv4 = GraphConv(256, 256)  # 128
        self.pool4 = TopKPooling(256, ratio=0.8)  # 128
        self.item_embedding = torch.nn.Embedding(num_embeddings=dataset.item_id.max() + 1, embedding_dim=256)
        self.lin1 = torch.nn.Linear(512, 256)  # 256, 128
        self.lin2 = torch.nn.Linear(256, 64)  # 128
        self.lin3 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.item_embedding(x)
        x = x.squeeze(1)

        x = F.relu(self.conv1(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)

        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv4(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 + x4

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)

        return x


class RecNet(torch.nn.Module):
    def __init__(self):
        super(RecNet, self).__init__()

        dataset = pd.read_csv('temp.csv')
        self.conv1 = GraphConv(128, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.conv4 = GraphConv(128, 128)  # 128
        self.pool4 = TopKPooling(128, ratio=0.8)  # 128
        self.item_embedding = torch.nn.Embedding(num_embeddings=dataset.item_id.max() + 1, embedding_dim=128)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.item_embedding(x)
        x = x.squeeze(1)

        x = F.relu(self.conv1(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x, edge_index, _, batch, _, _ = self.pool4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 + x4

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)

        return x


def train(model, trained):
    model.train()

    loss_all = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    crit = torch.nn.BCELoss()
    for data in trained:
        data = data.to('cpu')
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to('cpu')
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(trained)
