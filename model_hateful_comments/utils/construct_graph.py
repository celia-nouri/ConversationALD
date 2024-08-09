"""
Data(
  x={
    input_ids=[38, 100],
    token_type_ids=[38, 100],
    attention_mask=[38, 100],
  },
  edge_index=[2, 74],
  y=[1],
  x_images=[1, 3, 224, 224],
  x_text=[38],
  distance_matrix=[38],
  x_image_index=[38],
  y_mask=[38],
  idx=84,
  attn_bias=[39, 39],
  spatial_pos=[38, 38],
  distance=[38, 38],
  in_degree=[38],
  out_degree=[38]
)
comment, image_path, edge_dic, label =  x_text[0]     # length is 4

comment:  dict_keys(['archived', 'author', 'author_flair_css_class', 'author_flair_text', 'brand_safe', 'contest_mode', 'created_utc', 'distinguished', 'domain', 'edited', 'gilded', 'hidden', 'hide_score', 'id', 'is_crosspostable', 'is_reddit_media_domain', 'is_self', 'is_video', 'link_flair_css_class', 'link_flair_text', 'locked', 'media', 'media_embed', 'num_comments', 'num_crossposts', 'over_18', 'parent_whitelist_status', 'permalink', 'pinned', 'post_hint', 'preview', 'retrieved_on', 'score', 'secure_media', 'secure_media_embed', 'selftext', 'spoiler', 'stickied', 'subreddit', 'subreddit_id', 'suggested_sort', 'thumbnail', 'thumbnail_height', 'thumbnail_width', 'title', 'url', 'whitelist_status', 'label', 'body'])
image_path:  ['images/6zn7nf/6zn7nf-0.png']
edge_dic:  {'6zn7nf': [0, 0], 'dmwizv1': [0, 1], 'dmwwfsb': [0, 2], 'dmwzee2': [0, 2], 'dmwjhu3': [0, 1], 'dmwkbt3': [0, 2], 'dmwkhn5': [0, 3], 'dmwkzg3': [0, 4], 'dmwmqx3': [0, 5], 'dmwmwca': [0, 6], 'dmwttt3': [0, 7], 'dmwng13': [0, 6], 'dmwlybc': [0, 5], 'dmwnsvk': [0, 6], 'dmwo9fm': [0, 7], 'dmxocf0': [0, 6], 'dmws5ly': [0, 4], 'dmwxdjs': [0, 5], 'dmwxypw': [0, 6], 'dmxc4fs': [0, 7], 'dmxl31d': [0, 7], 'dmwswoj': [0, 5], 'dmxirbi': [0, 3], 'dmxltob': [0, 4], 'dmxlu96': [0, 5], 'dmxlvq8': [0, 6], 'dmxt922': [0, 5], 'dmxtdqq': [0, 6], 'dmwobso': [0, 2], 'dmwoe9e': [0, 3], 'dmwohq7': [0, 4], 'dmwoxlz': [0, 5], 'dmwp01p': [0, 6], 'dmwp83c': [0, 7], 'dmwp9k7': [0, 6], 'dmwwc8m': [0, 5], 'dmwwdbq': [0, 6], 'dmxj7vm': [0, 4]}
label:  NA

"""

def get_sentiment_features(x):
  return x


def temporal_feature(x):
  return x

def get_graph(x):
  posts_ids, nodes, relations, depths, edge_list, vid2num, vnum2id, temporal_info  = preprocess(x)
  vertices_dic, edges_dic = create_graphs(posts_ids, nodes, relations)
  print('edge dic ', edges_dic)
  print('relations ', relations)
  print('edge list ', edge_list)
  print('depth list ', depths)
  print('vid2num ', vid2num)
  print('vnum2id ', vnum2id)
  return x


def preprocess(conversation):
  nodes = {}
  relations = {}
  posts_ids = []
  depths = {}
  edge_list = []
  next_vertex = 0
  vertex_id_to_num = {}
  vertex_num_to_id = []
  temporal_info = []

  for i, comment_obj in enumerate(conversation):
    x, _, _, _ = comment_obj
    key = x['id']
    if key in vertex_id_to_num:
      print("Error, the vertex id ", key, " was already present in the id to num dictionnary")
    else: 
      vertex_id_to_num[key] = next_vertex
      vertex_num_to_id.append(key)
      temporal_info.append(x['created_utc'])
      next_vertex += 1
    
    if i == 0: #initial post 
    
      nodes[key] = x
      posts_ids.append(key)
      depths[key] = 0
      '''
      The keys of post object:
        ['archived', 'author', 'author_flair_css_class', 'author_flair_text', 'brand_safe', 
        'contest_mode', 'created_utc', 'distinguished', 'domain', 'edited', 'gilded', 'hidden', 'hide_score', 
        'id', 'is_crosspostable', 'is_reddit_media_domain', 'is_self', 'is_video', 'link_flair_css_class', 
        'link_flair_text', 'locked', 'media', 'media_embed', 'num_comments', 'num_crossposts', 'over_18', 
        'parent_whitelist_status', 'permalink', 'pinned', 'retrieved_on', 'score', 'secure_media', 'secure_media_embed', 
        'selftext', 'spoiler', 'stickied', 'subreddit', 'subreddit_id', 'suggested_sort', 'thumbnail', 'thumbnail_height', 
        'thumbnail_width', 'title', 'url', 'whitelist_status', 'label', 'body'])
      '''
    else: # comments
      '''
      The keys of comment objects:  
        ['author', 'author_flair_css_class', 'author_flair_text', 'body', 'can_gild', 'collapsed', 
        'collapsed_reason', 'controversiality', 'created_utc', 'distinguished', 'edited', 'gilded', 
        'id', 'is_submitter', 'link_id', 'parent_id', 'retrieved_on', 'score', 'stickied', 'subreddit', 
        'subreddit_id', 'label'])
      '''
    # same depth, one after the other 

      nodes[key] = x
      parent_id = get_value(x, "parent_id")
      if '_' in parent_id:
        parent_id = parent_id.split('_')[1]
      if parent_id in relations.keys():
        relations[parent_id].append(key)
      else:
        relations[parent_id] = [key]
      if parent_id in depths.keys():
        parent_depth = depths[parent_id]
        if key in depths.keys():
          print('error the child key depth was already populated')
        depths[key] = parent_depth + 1
      else:
        print('error the parent depth was not filled in the dict...')
        edge_list.append((vertex_id_to_num(parent_id), vertex_id_to_num(key)))

  return posts_ids, nodes, relations, depths, edge_list, vertex_id_to_num, vertex_num_to_id, temporal_info


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


