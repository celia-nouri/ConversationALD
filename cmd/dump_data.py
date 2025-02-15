import torch 
from tqdm import tqdm 
import wandb
import argparse
from mydatasets.mydataloaders import get_graph_dataloaders

import torch.nn.functional as F

import json

wandb.init(project="ald-conversation")

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


def main_data_dump(output_file=""):
    print("Running data dump...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str, default='large', help='the size of the dataset, can take one of the following values: ["small", "medium", "large", "small-1000", "cad"]')
    parser.add_argument('--validation', type=bool, default=True, help='rather or not to use a validation set for model tuning')
    parser.add_argument('--split', type=str, default='train', help='train, validation or test splits')

    args = parser.parse_args()

    train_loader, valid_loader, test_loader = get_graph_dataloaders(args.size, args.validation, 0) 
    if output_file == "":
        output_file = "path_to_repo/data/" + args.size + "_" + args.split + "_dump.jsonl"


    if args.split == 'test':
        data_dump_for_analysis(test_loader, output_file)
    elif args.split == 'validation':
        data_dump_for_analysis(valid_loader, output_file)
    else:
        data_dump_for_analysis(train_loader, output_file)


if __name__ == "__main__":
    main_data_dump()
