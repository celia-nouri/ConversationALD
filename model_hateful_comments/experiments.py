import json
import wandb
from models.model import all_model_names, get_device
from transformers import AdamW
import torch.nn.functional as F
from mydatasets.mydataloaders import get_pyg_data_loaders
from models.model import get_model
from fairseq.utils import get_available_activation_fns

from utils.train_eval_utils import train
import argparse


def run_experiments(args):
    model_name = args.model
    size = args.size
    validation = args.validation
    n_epochs = args.epochs
    undirected = args.undirected
    temp_edges = args.temp_edges
    num_layers = args.num_layers

    learning_rate = args.lr
    weight_decay = args.wd    
    assert validation in [True, False], "Invalid validation setting: {}".format(validation)
    assert model_name in all_model_names, "Invalid model name: {}".format(model_name)
    assert size in ["small", "small-1000", "medium", "large", "cad", "cad-small"], "Invalid size setting: {}".format(size)
    if model_name == "multimodal-transformer":
        args.with_graph = True
    else:
        args.with_graph = False

    device = get_device()
    print(f"Training {model_name} on {size} Hateful Discussions dataset with validation={validation}, for {n_epochs} epochs, a learning rate of {learning_rate} and weight decay of {weight_decay}...")
    print(f"Model hyperparams are num layers {num_layers}, undirected {undirected}, temporal edges {temp_edges}")

    # Log hyperparameters
    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": n_epochs,
        "hidden_channels": 64,
        "num_heads": 1,
        "model": model_name,
        "dataset": 'hateful_discussions',
        "undirected": undirected,
        "temp_edges": temp_edges,
        "size": size,
        "validation": validation,
        "with_graph": args.with_graph,
    }    

    train_loader, valid_loader, test_loader = get_pyg_data_loaders(size, validation, 0) #get_data_loaders(size, validation)
    
    # Instantiate your model
    model = get_model(args, model_name, hidden_channels=64, num_heads=1)

    # Define optimizer and loss function
    criterion = F.binary_cross_entropy_with_logits
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #criterion = loss_fn

    print("Training set size: ", len(train_loader))
    print("Validation set size: ", len(valid_loader))
    print("Test set size: ", len(test_loader))

    train(args, model, train_loader, valid_loader, test_loader, criterion, optimizer, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    models_string = json.dumps(all_model_names)
    parser.add_argument('--model', type=str, default='gat-test', help='the model to use, can take one of the following values: ' + models_string)
    parser.add_argument('--undirected', type=bool, default=False, help='define the graph model as an undirected graph')
    parser.add_argument('--temp-edges', type=bool, default=False, help='add temporal edges to the graph')
    parser.add_argument('--num-layers', type=int, default=3, help='the number of GAT layers in graph models')

    parser.add_argument('--with_graph', type=bool, default=False, help='rather or not to use a graphormer in the model to represent discussion dynamics')
    parser.add_argument('--size', type=str, default='cad', help='the size of the dataset, can take one of the following values: ["small", "medium", "large", "small-1000", "cad"]')
    parser.add_argument('--validation', type=bool, default=True, help='rather or not to use a validation set for model tuning')
    parser.add_argument('--epochs', type=int, default=20, metavar='E', help='number of epochs')
    parser.add_argument('--lr', type=float, default=3e-6, metavar='E', help='learning rate')
    parser.add_argument('--wd', type=float, default=0.1, metavar='E', help='weight decay')

    parser.add_argument('--enable-images', type=bool, default=True, metavar='E', help='rather or not to use the post images for training, defaults to True')


    parser.add_argument('--encoder-embed-dim', type=int, default=768, help='the dimension of the encoder embeddings')
    parser.add_argument('--encoder-layers', type=int, default=1, help='number of encoder layers')
    parser.add_argument('--num_bottleneck_tokens', type=int, default=2, help='number of bottle neck tokens')
    parser.add_argument('--num_fusion_layers', type=int, default=1,  help='number of fusion layers')
    parser.add_argument('--num_fusion_stack', type=int, default=5,  help='number of fusion stacks')
    parser.add_argument('--num_graph_stack', type=int, default=4,  help='number of graph stacks')

    parser.add_argument('--freeze_initial_encoders', type=bool, default=True, help='rather or not to freeze the initial encoder layers')

    # graph args
    parser.add_argument('--num_atoms', type=int, default=512 * 9, help='number of atoms')
    parser.add_argument('--num_edges', type=int, default=512 * 3, help='number of edges')
    parser.add_argument('--num_in_degree', type=int, default=512, help='number of in degrees')
    parser.add_argument('--num_out_degree', type=int, default=512,  help='number of out degrees')
    parser.add_argument('--num_edge_dis', type=int, default=128,  help='number of edge dis types in the graph')
    parser.add_argument('--multi_hop_max_dist', type=int, default=5,  help='max distance of multi-hop edges')
    parser.add_argument('--num_spatial', type=int, default=512,  help='number of spatial types in the graphs')
    parser.add_argument('--edge_type', type=str, default="multi_hop", help="edge type in the graph")
    parser.add_argument('--spatial-pos-max', type=int, default=1024,  help='max spatial position')
    parser.add_argument('--max_nodes', type=int, default=1000,  help='number of atoms')

    """Add model-specific arguments to the parser."""
    # Arguments related to dropout
    parser.add_argument("--dropout", type=float, metavar="D", default=0.4, help="dropout probability")
    parser.add_argument("--attention-dropout", type=float, metavar="D", default=0.3, help="dropout probability for attention weights")
    parser.add_argument("--act-dropout", type=float, metavar="D", default=0.3, help="dropout probability after activation in FFN")
    # Arguments related to hidden states and self-attention
    parser.add_argument("--encoder-ffn-embed-dim", type=int, default=768, metavar="N", help="encoder embedding dimension for FFN")
    parser.add_argument("--encoder-attention-heads", type=int, default=12, metavar="N",help="num encoder attention heads")

    # Arguments related to input and output embeddings
    parser.add_argument("--split", type=int, metavar="N", help="dataset split to use (not used in code)")
    parser.add_argument("--share-encoder-input-output-embed", action="store_true", help="share encoder input and output embeddings")
    parser.add_argument("--encoder-learned-pos", action="store_true", help="use learned positional embeddings in the encoder")
    parser.add_argument("--no-token-positional-embeddings", action="store_true", help="if set, disables positional embeddings (outside self attention)")
    parser.add_argument("--max-positions", type=int, help="number of positional embeddings to learn")
    parser.add_argument("--num-classes", type=int, default=1, help="number of classes for classification head")
    # Arguments related to parameter initialization
    parser.add_argument("--apply-graphormer-init", action="store_true", help="use custom param initialization for Graphormer")

    # misc params
    parser.add_argument("--activation-fn", choices=get_available_activation_fns(), default='relu', help="activation function to use")
    parser.add_argument("--encoder-normalize-before", action="store_true", help="apply layernorm before each encoder block")
    parser.add_argument("--pre-layernorm", action="store_true", help="apply layernorm before self-attention and ffn. Without this, post layernorm will used")

    args = parser.parse_args() 
    
    run_experiments(args)
