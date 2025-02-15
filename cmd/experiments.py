import json
import wandb
from models.model import all_model_names, all_base_pretrained_models, get_device
from transformers import AdamW
import torch.nn.functional as F
from mydatasets.mydataloaders import get_graph_dataloaders
from models.model import get_model

from utils.train_eval_utils import train
import argparse

from transformers import set_seed
import torch
import numpy as np
import random

def set_all_seeds(seed):
    set_seed(seed)  # Hugging Face transformers
    torch.manual_seed(seed)  # PyTorch
    np.random.seed(seed)  # NumPy
    random.seed(seed)  # Python random
    torch.cuda.manual_seed_all(seed)  # CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_experiments(args):
    model_name = args.model
    size = args.size
    validation = args.validation
    n_epochs = args.epochs
    undirected = args.undirected
    temp_edges = args.temp_edges
    num_layers = args.num_layers
    seed = args.seed

    learning_rate = args.lr
    weight_decay = args.wd    
    assert validation in [True, False], "Invalid validation setting: {}".format(validation)
    assert model_name in all_model_names, "Invalid model name: {}".format(model_name)
    assert size in ["small", "small-1000", "medium", "large", "cad", "cad-small"], "Invalid size setting: {}".format(size)

    device = get_device()
    print(f"Training {model_name} with seed {seed} on {size} ALD Conversation dataset with validation={validation}, for {n_epochs} epochs, a learning rate of {learning_rate} and weight decay of {weight_decay}...")
    print(f"Model hyperparams are num layers {num_layers}, undirected {undirected}, temporal edges {temp_edges}")

    # Log hyperparameters
    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": n_epochs,
        "hidden_channels": 64,
        "num_heads": 1,
        "model": model_name,
        "dataset": 'ald_conversation',
        "undirected": undirected,
        "temp_edges": temp_edges,
        "size": size,
        "validation": validation,
        "seed": seed,
    }    
    set_all_seeds(seed)

    train_loader, valid_loader, test_loader = get_graph_dataloaders(size, validation, 0) #get_data_loaders(size, validation)
    
    model = get_model(args, model_name, hidden_channels=64, num_heads=1)

    criterion = F.binary_cross_entropy_with_logits
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print("Training set size: ", len(train_loader))
    print("Validation set size: ", len(valid_loader))
    print("Test set size: ", len(test_loader))

    train(args, model, train_loader, valid_loader, test_loader, criterion, optimizer, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    models_string = json.dumps(all_model_names)
    pretrained_model_string = json.dumps(all_base_pretrained_models)
    parser.add_argument('--model', type=str, default="gat-test", help='the model to use, can take one of the following values: ' + models_string)
    # Options are "bert-base-uncased", "bert-base-cased", "roberta-base", "xlm-roberta-base", "allenai/longformer-base-4096", "answerdotai/ModernBERT-base", "answerdotai/ModernBERT-large" 
    parser.add_argument('--pretrained-model-name', type=str, default="bert-base-uncased", help='name for pretrained text model to use to generate text embeddings, can take one of the following values: ' + pretrained_model_string)
    parser.add_argument('--undirected', type=bool, default=False, help='define the graph model as an undirected graph')
    parser.add_argument('--temp-edges', type=bool, default=False, help='add temporal edges to the graph')
    parser.add_argument('--num-layers', type=int, default=5, help='the number of GAT layers in graph models')
    parser.add_argument('--trim', type=str, default="recent", help='graph construction trimming streatgy, should be either affordance, recent, or left empty for no trimming.')
    parser.add_argument('--new-trim', type=bool, default=False, help='rather or not to use the new trimming strategy (edge from post to target node only, instead of edges from post to all other nodes)')

    parser.add_argument('--size', type=str, default='cad', help='the size of the dataset, can take one of the following values: ["small", "medium", "large", "small-1000", "cad"]')
    parser.add_argument('--validation', type=bool, default=True, help='rather or not to use a validation set for model tuning')
    parser.add_argument('--epochs', type=int, default=20, metavar='E', help='number of epochs')
    parser.add_argument('--lr', type=float, default=3e-6, metavar='E', help='learning rate')
    parser.add_argument('--wd', type=float, default=0.1, metavar='E', help='weight decay')
    # seeds = [42, 7, 123, 2025, 99, 39, 1801, 762, 4504, 3]
    parser.add_argument('--seed', type=int, default=42, help='seed for training reproduciability')


    args = parser.parse_args() 
    
    run_experiments(args)
