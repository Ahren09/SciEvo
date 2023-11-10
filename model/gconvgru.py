import datetime
import logging
import os

import numpy as np
import os.path as osp
import pytz
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch_geometric_temporal import GConvGRU, temporal_signal_split
from scipy import sparse as sp
from scipy.sparse import find
from tqdm import trange

from arguments import parse_args
from dataset.dynamic_graph_dataset import CustomizedDynamicGraphStaticSignal
from utility.utils_logging import configure_default_logging


# TODO: Add more models
MODEL2CLASS = {
    "GConvGRU": GConvGRU,
}



class RecurrentGCN(nn.Module):
    def __init__(self, model, node_features: int, hidden_dim: int = 128,
                 transform_input: bool = True, device: str = "cuda:0",
                 do_node_classification: bool = False, do_node_regression: bool = False,
                 do_edge_classification: bool = False, do_edge_regression: bool = False,
                 do_link_prediction: bool = False,
                 num_classes_nodes: int = 0, num_classes_edges: int = 0):
        r"""
        Initialize Discrete-Time Dynamic Graph (DTDG) model.

        Args:
            model (str): Model type to be used for the GCN.
            node_features (int): Dimension of the input node features.
            hidden_dim (int, optional): Size of the hidden dimension. Defaults to 128.
            transform_input (bool, optional): If True, maps the input features to the same
                                              dimensionality as hidden_dim. Defaults to True.
            device (str, optional): Compute device for the model. Defaults to "cuda:0".
            num_classes_nodes (int, optional): Number of classes for node classification.
                                               Defaults to 0.
            num_classes_edges (int, optional): Number of classes for edge classification.
                                               Defaults to 0.

        """

        super(RecurrentGCN, self).__init__()

        self.transform_input = transform_input
        self.device = device
        self.num_classes_nodes = num_classes_nodes
        self.hidden_dim = hidden_dim
        self.num_classes_edges = num_classes_edges

        self.do_node_classification = do_node_classification
        self.do_node_regression = do_node_regression
        self.do_edge_classification = do_edge_classification
        self.do_edge_regression = do_edge_regression
        self.do_link_prediction = do_link_prediction

        if (do_node_classification and num_classes_nodes == 2) or do_node_regression:
            # For binary classification / regression on nodes, we use a single output neuron.
            self.lin_node = nn.Linear(hidden_dim, 1)

        elif do_node_classification and num_classes_nodes >= 3:
            # For multi-label classification on nodes, we use an MLP with SoftMax.
            self.lin_node = nn.Linear(hidden_dim, num_classes_nodes)

        if (do_edge_classification and self.num_classes_edges == 2) or do_edge_regression:

            self.lin_edge = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1)
            )


        elif do_edge_classification and self.num_classes_edges >= 3:
            self.lin_edge = nn.Linear(hidden_dim * 2,
                                      self.num_classes_edges)

        if self.transform_input:
            self.mlp_input = nn.Sequential(
                nn.Linear(node_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
            ).to(device)

        self.recurrent = MODEL2CLASS[model](hidden_dim, hidden_dim, 1).to(device)

        # TODO
        # if args.dataset_name == "Science2013Ant":
        #     # For the Ants dataset, encode the node type as a categorical feature
        #     self.embed_type = nn.Embedding(5, hidden_dim)
        #     self.linear_emb = nn.Linear(hidden_dim * 2, hidden_dim)

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)

        self.bceloss = nn.BCEWithLogitsLoss()

    def forward(self, h_0):

        h_1 = F.relu(h_0)
        out = self.node_regressor(h_1)
        return out

    def get_embedding(self, x, edge_index, edge_weight, y):
        if self.transform_input:
            x = self.mlp_input(x)

        h_0 = self.recurrent(x, edge_index, edge_weight)

        # h = torch.cat([h_0, self.embed_type(y.int())], dim=1)
        # h = F.relu(self.linear_emb(h))
        # h = F.dropout(h, p=0.1, training=self.training)
        # h = self.linear1(h)
        #
        # return h

        return h_0

    def link_prediction(self, h, src, dst):
        source_node_emb = h[src]
        target_node_emb = h[dst]

        score = torch.sum(source_node_emb * target_node_emb, dim=1)

        return score

    def transform_output(self, h_0):
        out = F.relu(h_0)
        out = F.dropout(out, p=0.1, training=self.training)
        h_1 = self.linear1(out)
        out = F.relu(h_1)
        out = F.dropout(out, p=0.1, training=self.training)

        return out, h_1

    def node_output_layer(self, out):
        score = self.lin_node(out)

        if self.do_node_classification:
            if self.num_classes_nodes == 2:
                score = nn.Sigmoid()(score)
                score = score.squeeze()

            elif self.num_classes_nodes >= 3:
                score = nn.Softmax(dim=1)(score)

            else:
                raise ValueError(
                    "Invalid number of classes for node classification")

        return score

    def edge_output_layer(self, edge_features):
        # src = out[edge_index[0]]
        # dst = out[edge_index[1]]

        # score = self.lin_edge(torch.concat([src, dst], dim=1))
        score = self.lin_edge(edge_features)
        score = nn.LogSoftmax(dim=1)(score)

        return score


def main():


    print("Constructing Dynamic Graph Model ...")

    idx_snapshot = 0

    format_string = "%Y-%m-%d"

    edge_li, edge_weights_li = [], []

    # Number of nodes
    N = None


    for start_year in range(1991, 2024):

        for start_month in range(1, 13):

            # TODO
            if idx_snapshot >= 4:
                break

            # Treat all papers before 1990 as one single snapshot
            if start_year < 1992:
                start = datetime.datetime(1970, 1, 1, 0, 0, 0, tzinfo=pytz.utc)

                end = datetime.datetime(1992, 1, 1, 0, 0, 0, tzinfo=pytz.utc)

            elif start_year == 2023 and start_month == 11:
                break

            else:
                if start_month == 12:
                    # Turn to January next year
                    end = datetime.datetime(start_year + 1, 1, 1, 0, 0, 0, tzinfo=pytz.utc)

                else:

                    # Turn to the next month in the same year
                    end = datetime.datetime(start_year, start_month + 1, 1, 0, 0, 0, tzinfo=pytz.utc)

                start = datetime.datetime(start_year, start_month, 1, 0, 0, 0, tzinfo=pytz.utc)

            idx_snapshot += 1
            filename = (f"outputs/snapshots/graph_{idx_snapshot}_{start.strftime(format_string)}"
                        f"_{end.strftime(format_string)}.npz")

            graph = sp.load_npz(filename)
            N = graph.shape[0]
            row_indices, col_indices, edge_weights = find(graph)

            assert edge_weights.max() == graph.max()

            edge_weights = np.log2(edge_weights + 1)

            edges = np.array(list(zip(row_indices, col_indices))).T

            edge_li += [edges]
            edge_weights_li += [edge_weights]


            # TODO
            if idx_snapshot >= 4:
                break

            if start_year < 1992:
                break



    D = args.embedding_dim
    # Calculate the variance for Xavier initialization
    variance = 2.0 / (N + D)

    # Initialize the embedding matrix with Xavier initialization
    features = np.random.randn(N, D) * np.sqrt(variance)

    full_dataset = CustomizedDynamicGraphStaticSignal(edge_index=edge_li, edge_weights=edge_weights_li,
                                                      features=features)

    train_dataset, test_dataset = train_dataset, test_dataset = temporal_signal_split(full_dataset,
                                                        train_ratio=1.)

    training_args = {
        "do_link_prediction": True,
        "do_node_regression": False,
        "do_node_classification": False,
        "do_edge_regression": True,
        "do_edge_classification": False,
    }

    model = RecurrentGCN(model=args.model_name,
                         node_features=args.embedding_dim,
                         hidden_dim=args.embedding_dim,
                         transform_input=False, device=args.device,
                         **training_args)

    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.9)

    def train_one_epoch(epoch: int, embeds_li: list):
        model.train()
        scheduler.step()

        if epoch % args.step_size == 0:
            logger.info(f"Epoch: {epoch}, LR: {scheduler.get_lr()}")

        cost = 0
        node_mask = None

        for idx_snapshot, snapshot in enumerate(train_dataset):
            log_str = f"\tEp.{epoch}\tIter{idx_snapshot}"
            snapshot = snapshot.to(args.device)

            node_mask = torch.ones(snapshot.num_nodes,
                                   dtype=torch.bool) if not hasattr(snapshot,
                                                                    "node_mask") else snapshot.node_mask

            h_0 = model.get_embedding(snapshot.x, snapshot.edge_index,
                                      snapshot.edge_attr, snapshot.y)

            if training_args["do_link_prediction"]:
                # Only pick negative nodes from the nodes that appear in the current graph
                nodes = snapshot.edge_index.unique().cpu()
                nodes = nodes.numpy().reshape(-1)
                neg_target = np.random.choice(nodes,
                                              size=snapshot.edge_index.shape[1])
                neg_target = torch.tensor(neg_target, dtype=torch.long,
                                          device=args.device)

                pos_score = model.link_prediction(h_0, snapshot.edge_index[0],
                                                  snapshot.edge_index[1])
                neg_score = -model.link_prediction(h_0, snapshot.edge_index[0],
                                                   neg_target)

                loss_link_pred = model.bceloss(pos_score,
                                               torch.ones_like(
                                                   pos_score)) + model.bceloss(
                    neg_score, torch.ones_like(neg_score))

                log_str += f"\tLink: {loss_link_pred.item():.3f}"

                cost += loss_link_pred

            # Transform the embedding from the model's output
            if training_args["do_node_regression"] or training_args[
                "do_edge_regression"] or training_args[
                "do_node_classification"] or \
                    training_args["do_edge_classification"]:
                out, h_1 = model.transform_output(h_0)
                emb = h_1

            if training_args["do_node_regression"]:
                pred_node = model.node_output_layer(out)

                loss_node_regression = torch.mean(
                    (pred_node[node_mask] - snapshot.y[
                        node_mask]) ** 2)

                cost += loss_node_regression * 0.1

                log_str += f"\tNode Reg: {loss_node_regression.item():.3f}"

            if training_args["do_edge_regression"]:
                edge_features = torch.cat(
                    [out[snapshot.edge_index[0]], out[snapshot.edge_index[1]]],
                    dim=1)
                pred_edge = model.edge_output_layer(edge_features)

                loss_edge_regression = torch.mean(
                    (pred_edge.squeeze(1) - snapshot.edge_attr) ** 2)

                cost += loss_edge_regression * 0.1

                log_str += f"\tEdge Reg: {loss_edge_regression.item():.3f}"

            # logger.info(log_str)
            if (epoch + 1) % args.save_every == 0:
                embeds_li += [emb.detach().cpu().numpy()]

        cost = cost / (idx_snapshot + 1)
        total_loss = cost.item()
        cost.backward()

        optimizer.step()
        optimizer.zero_grad()

        return total_loss

    cache_dir = osp.join("data", "")
    os.makedirs(cache_dir, exist_ok=True)

    pbar = trange(1, args.epochs + 1, desc='Train', leave=True)

    for epoch in pbar:

        # Store the embeddings at each epoch
        embeds_li = []
        loss = train_one_epoch(epoch, embeds_li)
        pbar.set_postfix({
            "epoch": epoch,
            "loss": "{:.3f}".format(loss)

        })

        if (epoch + 1) % args.save_every == 0:
            embeds: np.ndarray = np.stack([emb for emb in embeds_li])
            del embeds_li
            logger.info(
                f"[Embeds] Saving embeddings for Ep. {epoch + 1} with shape {embeds.shape} ...")
            np.save(osp.join(cache_dir,
                             f"{args.model_name}_embeds_Ep{epoch + 1}_Emb{args.embedding_dim}.npy"),
                    embeds)










if __name__ == "__main__":

    args = parse_args()
    configure_default_logging()
    logger = logging.getLogger(__name__)

    main()