import torch 
from tqdm import tqdm 
import wandb
import argparse
from models.model import get_model, get_device
#from datasets.dataloader import get_data_loaders
from transformers import AdamW
import torch.nn.functional as F
from utils.train_eval_utils import train
from mydatasets.hateful_discussions import get_data_loaders
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
    true_index = [i for i in range(len(a.y_mask)) if a.y_mask[i] == True]
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

    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of gnn layers.')

    parser.add_argument('--emb_dim', type=int, default=1024, help='Feature size.')
    parser.add_argument('--dataset_name', default='hateful_discussions', type=str, help='dataset name, can take one of these values: ["Palestine_convs_roberta", "hateful_discussions"]')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--epochs', type=int, default=20, metavar='E', help='number of epochs')

    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR', help='learning rate')  
    parser.add_argument('--dropout', type=float, default=0.4, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=16, metavar='BS', help='batch size') 

    parser.add_argument('--seed', type=int, default=100, help='random seed') 
    
    parser.add_argument('--model', type=str, default='fb-roberta-hate', help='can take one of the following values: ["text-class", "distil-class", "simple-graph", "roberta-class]') 
    parser.add_argument('--validation', type=bool, default=True, help='rather or not to evaluate on validation set for model tuning') 
    parser.add_argument('--size', type=str, default='medium', help='the size of the dataset, can take one of these values: ["small", "medium", "large"]') 

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

    train_loader, valid_loader, test_loader = get_data_loaders()

    # Instantiate the model
    model = get_model(args, model_name, hidden_channels=64, num_heads=1)

    # Define optimizer and loss function
    criterion = masked_bce_loss
    if model_name == "roberta-class":
        criterion = F.binary_cross_entropy_with_logits
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    #criterion = loss_fn

    train(args, model, train_loader, valid_loader, test_loader, criterion, optimizer, device=device)

def main_data_dump(output_file):
    print("Running data dump...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='hateful_discussions', type=str, help='dataset name, can take one of these values: ["Palestine_convs_roberta", "hateful_discussions"]')
    args = parser.parse_args()

    train_loader, _, _ = get_data_loaders(dataset_name=args.dataset_name, batch_size=1, num_workers=0, args=args)
    data_dump_for_analysis(train_loader, output_file)


if __name__ == "__main__":
    main_train()
    #main_data_dump("data/hatefuldiscussion_data/traindata_dump.jsonl")
