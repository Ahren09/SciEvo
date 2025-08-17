"""
Updated: 2024.8.25
"""

import os
import os.path as osp
import pickle
import sys
import warnings
from typing import Union, Any

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
from torch_geometric.loader.link_neighbor_loader import LinkNeighborLoader
from torch_geometric.loader.utils import filter_data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
from tqdm import tqdm, trange

sys.path.append(osp.join(os.getcwd(), "src"))
sys.path.append(os.getcwd())

from utility.utils_misc import project_setup
from arguments import parse_args
import const

warnings.simplefilter(action='ignore', category=UserWarning)


class Net(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels, out_channels, embed_dim: int = 100, device: str = "cuda:0"):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim


        self.embedding = torch.nn.Embedding(
            num_nodes,
            embedding_dim=embed_dim, sparse=True,
            dtype=torch.float32).to(device)
        nn.init.uniform_(self.embedding.weight, a=-0.1, b=0.1)
        self.conv1 = GCNConv(embed_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv2(x, edge_index, edge_weight)

        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


class ArXivLinkNeighborLoader(LinkNeighborLoader):

    def filter_fn(self, out: Any) -> Union[Data, HeteroData]:
        """
        We need this function to get the real node idss
        """

        if isinstance(self.data, Data):
            if isinstance(out, tuple):
                (node, row, col, edge, edge_label_index, edge_label) = out

            else:
                assert isinstance(out.metadata, tuple) and len(out.metadata) == 4
                node = out.node
                row = out.row
                col = out.col
                edge = out.edge
                edge_label_index = out.metadata[1]
                edge_label = out.metadata[2]

            if hasattr(self, "link_sampler"):
                data = filter_data(self.data, node, row, col, edge,
                                   self.link_sampler.perm)
            elif hasattr(self, "neighbor_sampler"):

                data = filter_data(self.data, node, row, col, edge,
                                   self.neighbor_sampler.perm)
            else:
                raise NotImplementedError

            data.edge_label_index = edge_label_index
            data.edge_label = edge_label

            # Added 2022.12.19
            data.node = node
            data.edge = edge

            return data if self.transform is None else self.transform(data)

        else:
            return super(LinkNeighborLoader, self).filter_fn(out)


def train(model, loader, optimizer, optimizer_sparse, criterion, epoch: int, device):
    model.train()
    total_loss = 0

    for i, data in enumerate(loader):
        optimizer.zero_grad()
        optimizer_sparse.zero_grad()

        x = model.embedding(data.node.to(device))
        # Do not use edge weights for now
        z = model(x, data.edge_index.to(device)
                  # data.edge_attr.to(device)
                  )

        # neg_edge_index = negative_sampling(
        #     edge_index=data.edge_index.to(device),
        #     num_nodes=data.num_nodes,
        #     num_neg_samples=data.edge_index.size(1),
        #     method='sparse').to(device)

        neg_edge_index = torch.stack(
            [data.edge_index[0], torch.randint(0, data.num_nodes, (data.edge_index.shape[1],))])

        edge_label = torch.cat([
            torch.ones(data.edge_index.shape[1], device=device),
            torch.zeros(data.edge_index.shape[1], device=device)
        ], dim=0)

        out = model.decode(z, torch.concat([data.edge_index.to(device), neg_edge_index.to(device)], axis=1))

        loss = criterion(out, edge_label)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer_sparse.step()

    return total_loss


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []

    node_embeddings = torch.zeros((model.num_nodes, model.embed_dim), dtype=torch.float32).to(device)

    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating"):
            x = model.embedding(data.node.to(device))
            z = model(x, data.edge_index.to(device))

            node_embeddings[data.node] = x.clone()

            # neg_edge_index = negative_sampling(
            #     edge_index=data.edge_index.to(device),
            #     num_nodes=data.num_nodes,
            #     num_neg_samples=data.edge_index.size(1),
            #     method='sparse').to(device)

            neg_edge_index = torch.stack(
                [data.edge_index[0], torch.randint(0, data.num_nodes, (data.edge_index.shape[1],))])

            out = model.decode(z, torch.concat([data.edge_index.to(device), neg_edge_index.to(device)], axis=1))

            edge_label = torch.cat([
                torch.ones(data.edge_index.shape[1], device=device),
                torch.zeros(data.edge_index.shape[1], device=device)
            ], dim=0)

            loss = criterion(out, edge_label)
            total_loss += loss.item()

            y_true.extend(edge_label.cpu().numpy())
            y_pred.extend(out.sigmoid().cpu().numpy())

    roc_auc = roc_auc_score(y_true, y_pred)
    return total_loss, roc_auc, node_embeddings


if __name__ == "__main__":
    project_setup()
    args = parse_args()

    path_graph = osp.join(args.output_dir, f'{args.feature_name}_edges.parquet')

    
    edge_df = pd.read_parquet(path_graph)

    year = args.start_year
    while year <= args.end_year:

        if year <= 1994:
            edges = edge_df.query("published_year >= 1985 and published_year < 1995")
            year = 1994

        else:
            edges = edge_df.query(f"published_year == {year}")

        edge_counts = edges.groupby([const.SOURCE, const.DESTINATION]).size().reset_index()
        edge_counts.rename(columns={0: 'weight'}, inplace=True)

        # Step 1: Create a unique node list
        unique_nodes = pd.concat([edges[const.SOURCE], edges[const.DESTINATION]]).unique()

        # Step 2: Create a mapping from node names to integers
        node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}

        # Step 3: Apply the mapping to the source and destination columns
        # edge_counts[const.SOURCE] = edge_counts[const.SOURCE].map(node_mapping)
        # edge_counts[const.DESTINATION] = edge_counts[const.DESTINATION].map(node_mapping)

        # Convert to PyTorch Geometric graph
        src = torch.tensor(edges[const.SOURCE].map(node_mapping).values, dtype=torch.long)
        dst = torch.tensor(edges[const.DESTINATION].map(node_mapping).values, dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
        # weights = torch.tensor(edge_counts['weight'].values, dtype=torch.float)

        full_data = Data(x=None, edge_index=to_undirected(edge_index)  # , edge_attr=weights
                         )

        # Set up model
        model = Net(len(node_mapping), 128, args.embed_dim, device=args.device)
        model.to(args.device)

        optimizer = torch.optim.Adam(
            params=[p for n, p in model.named_parameters() if
                    "embedding" not in n],
            lr=args.lr)
        optimizer_sparse = torch.optim.SparseAdam(
            params=[p for n, p in model.named_parameters() if "embedding" in n],
            lr=args.lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        train_loader = ArXivLinkNeighborLoader(full_data, num_neighbors=[10, 10], batch_size=args.batch_size,
                                               shuffle=True)
        eval_loader = ArXivLinkNeighborLoader(full_data, num_neighbors=[10, 10], batch_size=args.batch_size,
                                              shuffle=False)

        for epoch in trange(args.epochs, desc=f"Training {year}", position=0, leave=True):
            loss = train(model, train_loader, optimizer, optimizer_sparse, criterion, epoch, args.device)

            if epoch == 0 or (epoch + 1) % args.save_every == 0:
                loss, roc_auc, node_embeddings = evaluate(model, eval_loader, criterion, args.device)

                embed_path = osp.join(args.checkpoint_dir, f"{args.feature_name}_{args.tokenization_mode}", const.GCN,
                                      f"{const.GCN}_embeds_{year}.pkl")

                os.makedirs(osp.dirname(embed_path), exist_ok=True)

                # Save the embeddings
                if epoch + 1 >= (args.epochs):
                    with open(embed_path, "wb") as f:
                        pickle.dump({"embed": node_embeddings.cpu(),
                                     "node_mapping": node_mapping,
                                     }, f)

        year += 1

        # Retrieve the node embeddings
