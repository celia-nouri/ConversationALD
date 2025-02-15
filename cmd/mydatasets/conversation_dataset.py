import os
import copy
import torch
from typing import Optional, Callable
import numpy as np
from torch_geometric.data import Dataset
from .pyg_datasets.pyg_dataset import GraphConversationDataset

# This code was strongly inspired by: https://github.com/liamhebert/MultiModalDiscussionTransformer

class ConversationDataset(Dataset):
    def __init__(
        self,
        size='medium',
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.num_labels = 18673 if size =='large' or size == "cad" else 7210 if size =='medium' else 500 if size == 'small' else 1002 if size == "cad-small" else 6200
        self.k = 0
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        path = os.path.expandvars("$SLURM_TMPDIR/")
        return [path + "raw_graphs.json"]

    @property
    def processed_file_names(self):
        path = os.path.expandvars("$SLURM_TMPDIR/balanced_cad/processed")
        # TODO: this will be the total number of labels in the dataset, will have to update manually
        return [path + f"/graph-{i}.pt" for i in range(self.num_labels)]
    def len(self):
        return len(self.processed_file_names)

    # algorithm, go depth first, then do a second pass
    def get_relative_depth(self, node, depths={}) -> dict:
        distances = copy.deepcopy(depths)
        for key in distances.keys():
            distances[key][0] += 1
        distances[node["id"]] = [0, 0]

        for x in node["tree"]:
            val = self.get_relative_depth(x, distances)
            for key, value in val.items():
                if key not in distances:
                    value[1] = value[1] + 1
                    distances[key] = value
        node["distances"] = distances
        return copy.deepcopy(distances)

    def spread_downwards(self, node, depths={}):
        dists = copy.deepcopy(depths)
        for key, value in dists.items():
            if key not in node["distances"]:
                value[0] += 1
                node["distances"][key] = value
        for x in node["tree"]:
            self.spread_downwards(x, node["distances"])

    def collapse_tree(self, comment, data, root_images):
        if "id" not in comment["data"]:
            comment["data"]["id"] = comment["id"]
        comment["data"]["id"] = comment["id"]

        id = comment["data"]["id"]
        i = 0
        if comment["data"]["id"] in data:
            if comment["data"]["body"] != data[comment["data"]["id"]][0]["body"]:
                if data[comment["data"]["id"]][0]["body"] == "[deleted]":
                    if len(comment["images"]) == 0:
                        comment["images"] = root_images
                    data[comment["data"]["id"]] = (
                        comment["data"],
                        comment["images"],
                        comment["distances"],
                        comment["data"]["label"],
                    )
                    print("updated!")
        else:
            if len(comment["images"]) == 0:
                comment["images"] = root_images
            data[comment["data"]["id"]] = (
                comment["data"],
                comment["images"],
                comment["distances"],
                comment["data"]["label"],
            )
        for child in comment["tree"]:
            self.collapse_tree(child, data, root_images)

    def get(self, idx):
        path = self.processed_file_names[idx]
        a = torch.load(path)
        return a
    

def create_ald_conversation_dataset(size='medium', validation=True):
    assert size in ["small", "small-1000", "medium", "large", "cad", "cad-small", "print"] 
    # Set SLURM_TMPDIR
    # Switch to the right directory, depending on local or server configuration
    # Path removed for anonymization
    #os.environ['SLURM_TMPDIR'] = "path_to_repo/data"
    os.environ['SLURM_TMPDIR'] = "path_to_repo/data"
    dataset = ConversationDataset(size, root="balanced_cad")

    path = os.path.expandvars("$SLURM_TMPDIR")
    train_idx = []
    train_filename = path + "/train-idx.txt" if size == "medium" else path + "/train-idx-many.txt" if size == "large" else path + "/train-idx-small.txt"
    test_filename = path + "/test-idx-only.txt" if size == "medium" else path + "/test-idx-only-many.txt" if size == "large" else path + "/test-idx-only-small.txt"
    if size == "small-1000":
        train_filename = path + "/train-idx-small-1000.txt"  
        test_filename = path + "/test-idx-only-small-1000.txt"
    if size == "cad":
        train_filename = path + "/cad-train-idx-many.txt"
        test_filename = path + "/cad-test-idx-many.txt"
    if size == "cad-small":
        val_filename = path + "/cad-val-idx-small.txt"
        test_filename = path + "/cad-test-idx-small.txt"
    if size == "print":
        val_filename = path + "/print-index.txt"
        train_filename = path + "/print-index.txt"
        test_filename = path + "/print-index.txt"
    with open(train_filename, "r") as file:
        for line in file:
            train_idx.append(int(line[:-1]))

    val_idx = []
    if validation:
        val_filename = path + "/val-idx.txt" if size == "medium" else path + "/val-idx-many.txt" if size == "large" else path + "/val-idx-small.txt"
        test_filename = path + "/test-idx.txt" if size == "medium" else path + "/test-idx-many.txt" if size == "large" else path + "/test-idx-small.txt"
        if size == "small-1000":
            val_filename = path + "/val-idx-small-1000.txt"  
            test_filename = path + "/test-idx-small-1000.txt"
        if size == "cad":
            val_filename = path + "/cad-val-idx-many.txt"
            test_filename = path + "/cad-test-idx-many.txt"
        if size == "cad-small":
            val_filename = path + "/cad-val-idx-small.txt"
            test_filename = path + "/cad-test-idx-small.txt"
        if size == "print":
            val_filename = path + "/print-index.txt"
            train_filename = path + "/print-index.txt"
            test_filename = path + "/print-index.txt"
        with open(val_filename, "r") as file:
            for line in file:
                val_idx.append(int(line[:-1]))

    test_idx = []
    with open(test_filename, "r") as file:
        for line in file:
            test_idx.append(int(line[:-1]))

    return {
        "dataset": dataset,
        "train_idx": np.array(train_idx),
        "valid_idx": np.array(val_idx),
        "test_idx": np.array(test_idx),
        "source": "pyg",
    }

def create_graph_conversation_dataset(size='medium', validation=True, seed=0):
    conversation_data = create_ald_conversation_dataset(size, validation)
    graph_dataset = GraphConversationDataset(
        conversation_data["dataset"], 
        seed=seed, 
        train_idx=conversation_data["train_idx"],
        valid_idx=conversation_data["valid_idx"],
        test_idx=conversation_data["test_idx"],
    )
    return graph_dataset

class ConversationGraphDatasetLoader:
    def __init__(self, size, validation=True, seed=0):
        self.graphdataset = create_graph_conversation_dataset(size, validation, seed)
        self.dataset = self.graphdataset.dataset
        self.train_idx = self.graphdataset.train_idx
        self.test_idx = self.graphdataset.test_idx
        self.val_idx = self.graphdataset.valid_idx

    def get_train_data(self):
        return [self.graphdataset.get(idx) for idx in self.train_idx]

    def get_test_data(self):
        return [self.graphdataset.get(idx) for idx in self.test_idx]
    
    def get_val_data(self):
        return [self.graphdataset.get(idx) for idx in self.val_idx]



class ConversationDatasetLoader:
    def __init__(self, size, validation=True):
        conversation_data = create_ald_conversation_dataset(size, validation)
        self.dataset = conversation_data["dataset"]
        self.train_idx = conversation_data["train_idx"]
        self.test_idx = conversation_data["test_idx"]
        self.val_idx = conversation_data["valid_idx"]

    def get_train_data(self):
        return [self.dataset.get(idx) for idx in self.train_idx]

    def get_test_data(self):
        return [self.dataset.get(idx) for idx in self.test_idx]
    
    def get_val_data(self):
        return [self.dataset.get(idx) for idx in self.val_idx]

