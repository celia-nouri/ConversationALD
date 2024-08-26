import torch 
from tqdm import tqdm 
import wandb
import argparse
from models.model import get_model, get_device, all_model_names
#from datasets.dataloader import get_data_loaders
from mydatasets.mydataloaders import get_pyg_data_loaders

from transformers import AdamW
import torch.nn.functional as F
from utils.train_eval_utils import train
#from mydatasets.hateful_discussions import get_data_loaders
from fairseq.utils import get_available_activation_fns

import json

wandb.init(project="hatespeech-class")


# Define custom binary cross-entropy loss function with mask for ignored labels
def masked_bce_loss(predictions, labels):
    # Create a mask to ignore labels with value -1
    valid_mask = (labels != -1).float()
    labels = labels.float()
    while labels.dim() > 1:
        labels = labels.squeeze(0)
    
    # Calculate binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(predictions, labels, reduction='none')
    
    # Apply the valid mask to filter out losses corresponding to ignored labels
    masked_loss = loss * valid_mask
    
    # Calculate the mean loss
    mean_loss = torch.mean(masked_loss)
    
    return mean_loss

def populate_keys(in_dict, out_dict, keys, suffix=""):
    for k in keys:
        out_dict[suffix + k] = in_dict[k] if k in in_dict.keys() else ""
    return out_dict


def collect_data_from_conv(a):
    # a is dict with keys ['x', 'x_text', 'y_mask', 'edge_index', 'distance_matrix', 'x_images', 'y', 'x_image_index']
    output_data = {}
    output_data["true_binary_label"] = a['y'].item()
    true_index = [i for i in range(len(a.y_mask)) if a.y_mask[i]]
    assert len(true_index) == 1
    true_index = true_index[0]
    output_data["index_in_conv"] = true_index
    output_data["length_conv"] = len(a.y_mask)

    comment = a['x_text'][true_index]
    x, a2, a3, label = comment
    # x is a dict with keys: 'author', 'author_flair_css_class', 'author_flair_text', 'body', 'can_gild', 'collapsed', 'collapsed_reason', 'controversiality', 'created_utc', 'distinguished', 'edited', 'gilded', 'id', 'is_submitter', 'link_id', 'parent_id', 'retrieved_on', 'score', 'stickied', 'subreddit', 'subreddit_id', 'label'])

    #print(x.keys())
    output_data = populate_keys(x, output_data, ["author", "body", "controversiality", "subreddit", "label", "score", "parent_id", "id", "created_utc", "edited"], suffix="")

    x_post, _, _, label_post = a['x_text'][0]
    # label_post is NA

    # x_post is a dict with keys: []'archived', 'author', 'author_flair_css_class', 'author_flair_text', 'brand_safe', 'contest_mode', 'created_utc', 'distinguished', 'domain', 'edited', 'gilded', 'hidden', 'hide_score', 'id', 'is_crosspostable', 'is_reddit_media_domain', 'is_self', 'is_video', 'link_flair_css_class', 'link_flair_text', 'locked', 'media', 'media_embed', 'num_comments', 'num_crossposts', 'over_18', 'parent_whitelist_status', 'permalink', 'pinned', 'post_hint', 'preview', 'retrieved_on', 'score', 'secure_media', 'secure_media_embed', 'selftext', 'spoiler', 'stickied', 'subreddit', 'subreddit_id', 'suggested_sort', 'thumbnail', 'thumbnail_height', 'thumbnail_width', 'title', 'url', 'whitelist_status', 'label', 'body'])

    output_data = populate_keys(x_post, output_data, ["created_utc", "id", "score", "parent_id", "id", "num_comments", "score", "title", "url", "body", "selftext"], suffix="post_")
    return output_data


# Helper function to print shape or length of an array or tensor
def print_shape_or_length(obj, name):
    if isinstance(obj, torch.Tensor):
        print("Shape of tensor ", name, ": ", obj.shape)
    elif isinstance(obj, (list, tuple)):
        print("Length of array", name, ": ", len(obj))
    else:
        print("Unknown object type", name)

def data_dump_for_analysis(loader, output_file):
    progress_bar = tqdm(loader, desc=f"Data dump to {output_file}")
    for data in progress_bar:
        # debugging and data analysis purposes
        output_dic = collect_data_from_conv(data)
        with open(output_file, 'a', encoding='utf-8') as file:
            json_str = json.dumps(output_dic)
            file.write(json_str + '\n')


# Instead of functional version of BCE with logits, you could use the object oriented definition 
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)



def main_train():
    print("Running model training...")
    parser = argparse.ArgumentParser()

    models_string = json.dumps(all_model_names)
    parser.add_argument('--model', type=str, default='bert-class', help='the model to use, can take one of the following values: ' + models_string)
    parser.add_argument('--undirected', type=bool, default=True, help='define the graph model as an undirected graph')
    parser.add_argument('--temp-edges', type=bool, default=False, help='add temporal edges to the graph')
    parser.add_argument('--with_graph', type=bool, default=False, help='rather or not to use a graphormer in the model to represent discussion dynamics')
    parser.add_argument('--size', type=str, default='cad', help='the size of the dataset, can take one of the following values: ["small", "medium", "large", "small-1000", "cad"]')
    parser.add_argument('--validation', type=bool, default=True, help='rather or not to use a validation set for model tuning')
    parser.add_argument('--epochs', type=int, default=20, metavar='E', help='number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, metavar='E', help='learning rate')
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

    device = get_device()
    n_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    hidden_channels = 64
    num_heads = 1
    dataset_name = args.dataset_name
    model_name = args.model
    print("Dataset name: ", dataset_name, ". Model name: ", model_name)

    # Log hyperparameters
    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": n_epochs,
        "batch_size": batch_size,
        "hidden_channels": hidden_channels,
        "num_heads": num_heads,
        "model": model_name,
        "dataset": dataset_name,
    }    

    train_loader, valid_loader, test_loader = get_pyg_data_loaders(args.size, args.validation, 0) 

    # Instantiate the model
    model = get_model(args, model_name, hidden_channels=64, num_heads=1)

    # Define optimizer and loss function
    criterion = masked_bce_loss
    if model_name == "roberta-class":
        criterion = F.binary_cross_entropy_with_logits
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    #criterion = loss_fn

    train(args, model, train_loader, valid_loader, test_loader, criterion, optimizer, device=device)

def main_data_dump(output_file=""):
    print("Running data dump...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str, default='large', help='the size of the dataset, can take one of the following values: ["small", "medium", "large", "small-1000", "cad"]')
    parser.add_argument('--dataset_name', default='hateful_discussions', type=str, help='dataset name, can take one of these values: ["Palestine_convs_roberta", "hateful_discussions"]')
    parser.add_argument('--validation', type=bool, default=True, help='rather or not to use a validation set for model tuning')
    parser.add_argument('--split', type=str, default='train', help='train, validation or test splits')

    args = parser.parse_args()

    train_loader, valid_loader, test_loader = get_pyg_data_loaders(args.size, args.validation, 0) 
    if output_file == "":
        output_file = "/Users/celianouri/Stage24/HatefulDiscussionsModeling/data/" + args.size + "_" + args.split + "_dump.jsonl"


    #train_loader, _, _ = get_data_loaders(dataset_name=args.dataset_name, batch_size=1, num_workers=0, args=args)
    if args.split == 'test':
        data_dump_for_analysis(test_loader, output_file)
    elif args.split == 'validation':
        data_dump_for_analysis(valid_loader, output_file)
    else:
        data_dump_for_analysis(train_loader, output_file)


if __name__ == "__main__":
    #main_train()
    main_data_dump()
