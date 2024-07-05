import json
import random 
import torch
from transformers import AutoTokenizer, DistilBertTokenizer, DistilBertModel, RobertaTokenizer, RobertaModel

# Preprocessing of comments and posts 

### DATA ###

## r_Palestine_comments
# 161,741 comments
# Fields in JSON object: ['_meta', 'all_awardings', 'approved_at_utc', 'approved_by', 'archived', 'associated_award', 'author', 'author_flair_background_color', 'author_flair_css_class', 'author_flair_richtext', 'author_flair_template_id', 'author_flair_text', 'author_flair_text_color', 'author_flair_type', 'author_fullname', 'author_is_blocked', 'author_patreon_flair', 'author_premium', 'awarders', 'banned_at_utc', 'banned_by', 'body', 'can_gild', 'can_mod_post', 'collapsed', 'collapsed_because_crowd_control', 'collapsed_reason', 'collapsed_reason_code', 'comment_type', 'controversiality', 'created', 'created_utc', 'distinguished', 'downs', 'edited', 'gilded', 'gildings', 'id', 'is_submitter', 'likes', 'link_id', 'locked', 'mod_note', 'mod_reason_by', 'mod_reason_title', 'mod_reports', 'name', 'no_follow', 'num_reports', 'parent_id', 'permalink', 'removal_reason', 'replies', 'report_reasons', 'retrieved_on', 'saved', 'score', 'score_hidden', 'send_replies', 'stickied', 'subreddit', 'subreddit_id', 'subreddit_name_pre

## r_Palestine_posts
# 19,833 posts
# Fields in JSON object: ['_meta', 'all_awardings', 'allow_live_comments', 'approved_at_utc', 'approved_by', 'archived', 'author', 'author_flair_background_color', 'author_flair_css_class', 'author_flair_richtext', 'author_flair_template_id', 'author_flair_text', 'author_flair_text_color', 'author_flair_type', 'author_fullname', 'author_is_blocked', 'author_patreon_flair', 'author_premium', 'awarders', 'banned_at_utc', 'banned_by', 'can_gild', 'can_mod_post', 'category', 'clicked', 'content_categories', 'contest_mode', 'created', 'created_utc', 'crosspost_parent', 'crosspost_parent_list', 'discussion_type', 'distinguished', 'domain', 'downs', 'edited', 'gilded', 'gildings', 'hidden', 'hide_score', 'id', 'is_created_from_ads_ui', 'is_crosspostable', 'is_meta', 'is_original_content', 'is_reddit_media_domain', 'is_robot_indexable', 'is_self', 'is_video', 'likes', 'link_flair_background_color', 'link_flair_css_class', 'link_flair_richtext', 'link_flair_template_id', 'link_flair_text', 'link_flair_text_color', 'link_flair_type', 'locked', 'media', 'media_embed', 'media_only', 'mod_note', 'mod_reason_by', 'mod_reason_title', 'mod_reports', 'name', 'no_follow', 'num_comments', 'num_crossposts', 'num_reports', 'over_18', 'parent_whitelist_status', 'permalink', 'pinned', 'post_hint', 'preview', 'pwls', 'quarantine', 'removal_reason', 'removed_by', 'removed_by_category', 'report_reasons', 'retrieved_on', 'saved', 'score', 'secure_media', 'secure_media_embed', 'selftext', 'send_replies', 'spoiler', 'stickied', 'subreddit', 'subreddit_id', 'subreddit_name_prefixed', 'subreddit_subscribers', 'subreddit_type', 'suggested_sort', 'thumbnail', 'thumbnail_height', 'thumbnail_width', 'title', 'top_awarded_type', 'total_awards_received', 'treatment_tags', 'ups', 'upvote_ratio', 'url', 'url_overridden_by_dest', 'user_reports', 'view_count', 'visited', 'whitelist_status', 'wls']

def preprocess(posts_path, comments_path):
    
    nodes = {}
    relations = {}
    posts_ids = []
    
    with open(posts_path, 'r') as file:
        # Preprocess posts file, add a node for each post. Also create a list of post ids to keep track of the graph roots.
        for _, line in enumerate(file):
            json_obj = json.loads(line.strip())
            key = json_obj["name"]
            # contruct the node value
            value = {"type": "post", "id": key, "name": get_value(json_obj, "name"), "author": get_value(json_obj, "author"), "author_fullname": get_value(json_obj, "author_fullname"), 
                     "title": get_value(json_obj, "title"), "num_comments": get_value(json_obj, "num_comments"), "score": get_value(json_obj, "score"), "subreddit": get_value(json_obj, "subreddit"), "subreddit_id": get_value(json_obj, "subreddit_id"), 
                     "ups": get_value(json_obj, "ups"), "upvote_ratio": get_value(json_obj, "upvote_ratio"), "url": get_value(json_obj, "url")}
            nodes[key] = value
            posts_ids.append(key)

    with open(comments_path, 'r') as file:
        # Preprocess comments file. Add a node for each comment and a relation between parent and child to the relation table.
        for _, line in enumerate(file):
            json_obj = json.loads(line.strip())
            key = json_obj["name"]
            value = {"type": "comment", "id": key, "author": get_value(json_obj, "author"), "author_fullname": get_value(json_obj, "author_fullname"), 
                     "name": get_value(json_obj, "name"), "score": get_value(json_obj, "score"), "subreddit": get_value(json_obj, "subreddit"), "subreddit_id": get_value(json_obj, "subreddit_id"), 
                     "ups": get_value(json_obj, "ups"), "controversiality": get_value(json_obj, "controversiality"), "body": get_value(json_obj, "body"), "parent_id": get_value(json_obj, "parent_id")}
            nodes[key] = value
            parent_id = get_value(json_obj, "parent_id")
            if parent_id in relations.keys():
                relations[parent_id].append(key)
            else:
                relations[parent_id] = [key]
    return posts_ids, nodes, relations

def get_value(obj, key):
    if key in obj.keys():
        return obj[key]
    else:
        return ""

def create_edges(parents, relations):
    edges = []
    if not parents:
        return edges
    
    parent = parents[0]
    leftover_parents = parents[1:]

    if parent in relations:
        children = relations[parent]
        for child in children:
            edges.append((parent, child))
        # Extend children with leftover_parents and pass it to create_edges
        edges += create_edges(children + leftover_parents, relations)
    return edges

def create_graphs(roots, nodes, relations):
    all_vertices, all_edges = {}, {}
    for root in roots:
        edges = create_edges([root], relations)
        vertices = []
        if root in nodes.keys():
            vertices.append(nodes[root])
        for edge in edges:
            _, child = edge
            if child in nodes.keys():
                vertices.append(nodes[child])
        all_vertices[root] = vertices
        all_edges[root] = edges
    return all_vertices, all_edges

def write_to_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def append_line_to_file(data, file_path):
    with open(file_path, 'a') as file:
        file.write(json.dumps(data) + '\n')

def is_controv_conv(conv): 
    for vertex in conv:
        # returns true if conversation contrains at least one controversial comment
        if vertex["type"] == "comment" and "controversiality" in vertex.keys() and vertex["controversiality"] == 1:
            return True 
    return False


def train_test_val_datasets(vertices_path, tokenizer, bert_model, run_name, run_bert=False):
    total_controv, total_noncontrov = 0, 0
    final_vertex_num = 0
    id_map = {}

    with open(vertices_path, 'r') as file:
        vertices = json.load(file)
        for root_id in vertices.keys():

            conv = vertices[root_id]
    
            # only consider conversations that have at least 3 comments (1 post + 3 comments)
            if len(conv) > 4:
                conversation = []

                controv_conv = is_controv_conv(conv)
                if controv_conv:
                    total_controv += 1
                elif total_controv > total_noncontrov:
                    total_noncontrov += 1
                else:
                    continue

                for vertex in conv:
                    vertex_id = vertex["name"]
                    if vertex_id in id_map.keys():
                        vertex_num = id_map[vertex_id]
                    else:
                        final_vertex_num = final_vertex_num + 1
                        vertex_num = final_vertex_num 
                        id_map[vertex_id] = vertex_num

                    if "parent_id" not in vertex.keys():
                        parent_vertex_num = 0
                    elif vertex["parent_id"] in id_map.keys(): 
                        parent_vertex_num = id_map[vertex["parent_id"]]
                    else:
                        final_vertex_num = final_vertex_num + 1
                        parent_vertex_num = final_vertex_num
                        id_map[vertex["parent_id"]] = parent_vertex_num

                    text = ""
                    if "body" in vertex.keys():
                        text = vertex["body"]
                    elif "title" in vertex.keys():
                        text = vertex["title"]
                    text = " ".join(text.split())
                    new_vertex =  {
                        "is_comment": 1 if vertex["type"] == "comment" else 0,
                        "score": 3,
                        "ups": 3,
                        "vertex_num": vertex_num,
                        "text": text,
                        "parent_id": parent_vertex_num,
                        "controversiality": 0 if "controversiality" not in vertex.keys() else vertex["controversiality"],
                    }

                    if run_bert:
                        new_vertex["text_bert_emb"] = get_bert_feature_representation(new_vertex["text"], tokenizer, bert_model)

                    conversation.append(new_vertex)
                # write conversation to train/test/validation file
                rand_float = random.random()
                if rand_float < 0.7:
                    append_line_to_file({root_id: conversation}, 'data/' + run_name + '/' + 'train_data.jsonl')
                elif rand_float < 0.85:
                    append_line_to_file({root_id: conversation}, 'data/' + run_name + '/' +'val_data.jsonl')
                else:
                    append_line_to_file({root_id: conversation}, 'data/' + run_name + '/' + 'test_data.jsonl')

    if len(id_map) > 0:
        write_to_file(id_map, 'data/id_map.json')


def get_bert_feature_representation(text, tokenizer, bert_model, max_length=100):
    # Tokenize the input text
    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=max_length,
        pad_to_max_length=True,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    # Get the last hidden state from the BERT-like model
    with torch.no_grad():
        outputs = bert_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        last_hidden_state = outputs.last_hidden_state

        # Extract the feature representation for the [CLS] token
        cls_token_representation = last_hidden_state[:, 0, :].squeeze().tolist()

    return cls_token_representation
   
"""
def main():
    #roots, nodes, relations = preprocess('data/r_Palestine_posts.jsonl', 'data/r_Palestine_comments.jsonl')
    #vertices_dic, edges_dic = create_graphs(roots, nodes, relations)
    #write_to_file(vertices_dic, 'data/Palestine_vertices.json')
    #write_to_file(edges_dic, 'data/Palestine_edges.json')
    
    # DistilBERT
    #tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    #bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    # RoBERTa
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    bert_model = RobertaModel.from_pretrained('roberta-base')

    train_test_val_datasets('data/Palestine_vertices.json', tokenizer, bert_model, 'Palestine_convs_roberta', run_bert=True)
    

if __name__ == "__main__":
    main()
    """