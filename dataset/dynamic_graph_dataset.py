import torch
from torch_geometric.data import Data

import numpy as np

from typing import List, Union

from torch_geometric_temporal import DynamicGraphStaticSignal

Edge_Index = Union[np.ndarray, None]
Edge_Weight = Union[np.ndarray, None]
Node_Features = List[Union[np.ndarray, None]]
Targets = List[Union[np.ndarray, None]]
Additional_Features = List[np.ndarray]



class CustomizedDynamicGraphStaticSignal(DynamicGraphStaticSignal):
    def __init__(
            self,
            edge_index: List[Edge_Index],
            edge_weight: List[Edge_Weight] = None,
            features: Node_Features = None,
            targets: Targets = None,
            # node_masks: List[np.ndarray] = None,
            embedding_dim: int = 32,
            **kwargs: Additional_Features
    ):
        r"""
        Dataset for static graph (constant edges / graph connectivities) and static signal (constant edge features).

        Args:
            edge_index (List[Edge_Index]):
            edge_weight (List[Edge_Weight]:
            features (Node_Features)::
            targets (Targets):
            **kwargs (Additional_Features):
        """
        self.edge_index = edge_index

        if edge_weight is None:
            edge_weight = [np.ones(e_index.shape[1]) for e_index in edge_index]

        self.edge_weight = edge_weight

        self.features = features

        if targets is None:
            targets = [np.array([]) for _ in range(len(edge_index))]

        self.targets = targets
        self.additional_feature_keys = []

        for key, value in kwargs.items():

            setattr(self, key, value)
            self.additional_feature_keys.append(key)

        self.snapshot_count = len(self.edge_index)

    def __len__(self):
        return self.snapshot_count

    def __getitem__(self, time_index: Union[int, slice]):
        if isinstance(time_index, slice):
            snapshot = CustomizedDynamicGraphStaticSignal(
                self.edge_index, # The edge_index for StaticGraphStaticSignal remains constant over time, which is different from DynamicGraphStaticSignal. So we only keep one copy of it.
                self.edge_weight[time_index],
                features=self.features,
                targets=self.targets[time_index],
                **{key: getattr(self, key)[time_index] for key in
                   self.additional_feature_keys}
            )


        else:
            x = torch.tensor(self.features, dtype=torch.float)
            edge_index = torch.tensor(self.edge_index[time_index],
                                      dtype=torch.long)
            edge_weight = torch.tensor(self.edge_weight[time_index],
                                       dtype=torch.float)

            additional_features = self._get_additional_features(time_index)
            if isinstance(self.targets, (np.ndarray, list)) and len(self.targets[time_index]) > 0:
                y = torch.tensor(self.targets[time_index], dtype=torch.float)
                node_mask = torch.tensor(self.node_masks[time_index],
                                         dtype=torch.bool)
                snapshot = Data(x=x, edge_index=edge_index,
                                edge_attr=edge_weight,
                                y=y, node_mask=node_mask, **additional_features)

            else:
                snapshot = Data(x=x, edge_index=edge_index,
                                edge_attr=edge_weight,
                                **additional_features)

        return snapshot

    def __next__(self):
        if self.t < len(self.edge_index):
            snapshot = self[self.t]
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration