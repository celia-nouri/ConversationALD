import json 
import torch
from torch.utils.data import Dataset
import random
from transformers import AutoTokenizer, DistilBertTokenizer, DistilBertModel, RobertaTokenizer, RobertaModel
from .preprocess import get_bert_feature_representation
from .hateful_discussions import HatefulDiscussionsDatasetLoader, HatefulDiscPygDatasetLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
          


def get_data_loaders(size='medium', validation=True):
    print(f"Building hateful discussions dataloaders: size = {size}, validation = {validation}")

    ds = HatefulDiscussionsDatasetLoader(size, validation)
    train_loader, valid_loader, test_loader = ds.get_train_data(), ds.get_val_data(), ds.get_test_data()
    return train_loader, valid_loader, test_loader

def get_pyg_data_loaders(size='medium', validation=True, seed=0):
    print(f"Building hateful discussions dataloaders: size = {size}, validation = {validation}")

    ds = HatefulDiscPygDatasetLoader(size, validation, seed)
    train_loader, valid_loader, test_loader = ds.get_train_data(), ds.get_val_data(), ds.get_test_data()
    return train_loader, valid_loader, test_loader
