def get_sentiment_features(x):
  return x


def temporal_edges(temporal_info, depths, vid2num, undirected=False):
  temporal_edges = {}
  reversed_depths = {}
  for vertex_id, depth_info in depths.items():
    depth, parent_id, root_id = depth_info
    vertex_num = vid2num[vertex_id]
    new_key = str(depth) + '+' + parent_id + '+' + root_id
    reversed_depths.setdefault(new_key, []).append(vertex_num)
  
  for depth_key in reversed_depths.keys():
    depth, parent_id, root_id = depth_key.split('+')
    vertex_nums = reversed_depths[depth_key]
    if len(vertex_nums) > 1:
      sorted_nums = [index for index, _ in sorted(zip(vertex_nums, temporal_info), key=lambda x: int(x[1]))]
      edge_set = set()
      for i in range(len(sorted_nums) - 1):
        edge_set.add((sorted_nums[i], sorted_nums[i+1]))
        if undirected:
          edge_set.add((sorted_nums[i+1], sorted_nums[i]))
      temporal_edges.setdefault(root_id, []).extend(list(edge_set))

  return temporal_edges

def get_graph(x, mask, with_temporal_edges=False, undirected=False, trim="affordance", new_trim=False):
  masked_index = mask.nonzero(as_tuple=True)[0]
  x_node, _, _, label = x[masked_index]
  my_id = x_node['id']

  assert label != 'NA', 'NA label should not be assigned to a node we train on'
  
  posts_ids, nodes, relations, depths, edge_list, vid2num, vnum2id, temporal_info, num_nodes, conv_indices_to_keep  = preprocess(x, masked_index, undirected, trim, new_trim)
  my_new_mask_idx = vid2num[my_id]
  if len(posts_ids) ==  1:
    edges_dic_num = {posts_ids[0]: edge_list}
  else:
    edges_dic_num = {my_id: edge_list}
  # dictionaries with key = root, value = vertex/edge list

  if with_temporal_edges:
    tempo_edges = temporal_edges(temporal_info, depths, vid2num, undirected)
    edges_dic_num = merge_dictionaries(edges_dic_num, tempo_edges)
    #edge_list = list(set(edge_list + tempo_edges))
  return nodes, edges_dic_num, conv_indices_to_keep, my_new_mask_idx

def get_hetero_graph(x, mask, with_temporal_edges=False, trim="affordance", new_trim=False):
  masked_index = mask.nonzero(as_tuple=True)[0]
  x_node, _, _, label = x[masked_index]
  my_id = x_node['id']
  assert label != 'NA', 'NA label should not be assigned to a node we train on'
  
  posts_ids, comments_nodes, relations, depths, comments_edge_list, vid2num, vnum2id, temporal_info, num_comment_nodes, conv_indices_to_keep  = preprocess(x, masked_index, False, trim, new_trim)
  my_new_mask_idx = vid2num[my_id]
  if len(posts_ids) ==  1:
    comments_edges_dic_num = {posts_ids[0]: comments_edge_list}
  else:
    comments_edges_dic_num = {my_id: comments_edge_list}
  # dictionaries with key = root, value = vertex/edge list

  if with_temporal_edges:
    tempo_edges = temporal_edges(temporal_info, depths, vid2num)
    comments_edges_dic_num = merge_dictionaries(comments_edges_dic_num, tempo_edges)
    #edge_list = list(set(edge_list + tempo_edges))
  x = [y for i, y in enumerate(x) if i in conv_indices_to_keep]
  user_to_comments_edges, num_users, user_id2num = get_user_graphs(x, vid2num)
  return num_comment_nodes, comments_edges_dic_num, num_users, user_to_comments_edges, conv_indices_to_keep, my_new_mask_idx

def get_user_graphs(conversation, comments_id2num):
  edge_list = []
  next_vertex = 0
  user_id2num = {}

  for i, comment_obj in enumerate(conversation):
    x, _, _, _ = comment_obj
    comment_id = x['id']
    comment_num = -1
    if comment_id in comments_id2num.keys():
      comment_num = comments_id2num[comment_id]
    else:
      raise ValueError('error: comment id ', comment_id, ' is not in comment_id2num dic. This should not be happening...')
    user_id = x['author']
    user_num = -1
    if user_id in user_id2num.keys():
      user_num = user_id2num[user_id]
    else:
      user_id2num[user_id] = next_vertex
      user_num = user_id2num[user_id]
      next_vertex += 1
    if comment_num != -1 and user_num != -1:
      edge_list.append((user_num, comment_num))
  edge_list = sorted(edge_list, key=lambda x: (x[0], x[1]))
  return edge_list, next_vertex, user_id2num


def convert_to_num(edges_dic, vid2num):
  edges_dic_num = {}
  for root_id, edge_list in edges_dic.items():
    edges_num = []
    for edge in edge_list:
      edge1, edge2 = edge
      edges_num.append((vid2num[edge1], vid2num[edge2]))
    edges_dic_num[root_id] = edges_num
  return edges_dic_num

def merge_dictionaries(dict1, dict2):
    keys = set(dict1) | set(dict2)
    merged_dict = {
        key: list(set(dict1.get(key, []) + dict2.get(key, [])))
        for key in keys
    }
    return merged_dict

def preprocess(conversation, mask_index, undirected=False, trim="affordance", new_trim=False):
  edge_set = set()
  nodes = {}
  relations = {}
  posts_ids = []
  depths = {}
  edge_list = []
  next_vertex = 0
  vertex_id_to_num = {}
  vertex_num_to_id = []
  temporal_info = []
  ids_to_keep = set()
  post_id = ""
  conv_indices_to_keep = []

  assert trim in ["affordance", "recent", ""]

  if trim == "affordance":
    ids_to_keep = trim_tree(conversation, mask_index)
  elif trim == "recent":
    ids_to_keep = trim_tree_recent(conversation, mask_index)

  target_node = conversation[mask_index]
  target_id = target_node[0]['id']

  for i, comment_obj in enumerate(conversation):
    x, _, _, _ = comment_obj
    key = x['id']
    if key in ids_to_keep:
      conv_indices_to_keep.append(i)
    else:
      continue
    if key in vertex_id_to_num:
      raise RuntimeError("Error, the vertex id ", key, " was already present in the id to num dictionnary")
    else: 
      vertex_id_to_num[key] = next_vertex
      vertex_num_to_id.append(key)
      temporal_info.append(x['created_utc'])
      next_vertex += 1
    
    if i == 0: #initial post 
      nodes[key] = x
      posts_ids.append(key)
      post_id = key
      # set depths dict to (depth, parent_id, root_id)
      depths[key] = (0, "", key)
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
        parent_depth, _, my_root = depths[parent_id]
        if key in depths.keys():
          raise RuntimeError('error the child key depth was already populated')
        depths[key] = (parent_depth + 1, parent_id, my_root)
      else:
        RuntimeError('error the parent depth was not filled in the dict...')
      # add an edge from parent post to child
      if parent_id in vertex_id_to_num.keys() and key in vertex_id_to_num.keys():
        edge_set.add((vertex_id_to_num[parent_id], vertex_id_to_num[key]))
    
      # Another tree trimming strategy: only add an edge from initial post to the target comment
      # if new_trim is False, we instead add an edge from the initial post to all nodes in the trimmed tree
      if post_id in vertex_id_to_num.keys():
        if new_trim:
          if key == target_id:
            edge_set.add((vertex_id_to_num[post_id], vertex_id_to_num[key]))
        else:
          edge_set.add((vertex_id_to_num[post_id], vertex_id_to_num[key]))

      if undirected and parent_id in vertex_id_to_num.keys(): # add revert all edges
        edge_set.add((vertex_id_to_num[key], vertex_id_to_num[parent_id]))

  edge_list = list(edge_set)
  return posts_ids, nodes, relations, depths, edge_list, vertex_id_to_num, vertex_num_to_id, temporal_info, next_vertex, conv_indices_to_keep


def trim_tree(conversation, node_index):
    target_node = conversation[node_index]
    target_timestamp = target_node[0]['created_utc']
    target_id = target_node[0]['id']

    children = {}
    node_by_id = {}
    for i, node in enumerate(conversation):
        node_id = node[0]['id']
        parent_id = node[0]['parent_id']

        node_by_id[node_id] = node[0]
        if parent_id not in children.keys():
            children[parent_id] = []
        children[parent_id].append(node[0])

    trimmed_ids = set()
    root_node = conversation[0]
    root_id = root_node[0]['id']
    trimmed_ids.add(root_id)

    depth_1_nodes = children.get(root_id, [])

    depth_1_nodes_sorted = sorted(depth_1_nodes, key=lambda x: x['score'], reverse=True)
    if len(depth_1_nodes_sorted) > 5:
        depth_1_nodes_sorted = depth_1_nodes_sorted[:5]  # Trim down to top 5 nodes

    for node in depth_1_nodes_sorted:
        if node['created_utc'] <= target_timestamp:
            trimmed_ids.add(node['id'])

            depth_2_nodes = children.get(node['id'], [])
            if depth_2_nodes:
                top_depth_2_node = max(depth_2_nodes, key=lambda x: x['score'])
                if top_depth_2_node['created_utc'] <= target_timestamp:
                    trimmed_ids.add(top_depth_2_node['id'])
    
    current_node = target_node[0]
    while current_node['parent_id'] != '':
        parent_id = current_node['parent_id']
        if parent_id != "" and parent_id in node_by_id:
            parent_node = node_by_id[parent_id]
            if parent_node['created_utc'] <= target_timestamp:
                trimmed_ids.add(parent_id)

            else:
              raise RuntimeError("error: parent node more recent than the child node: parent created at ",  parent_node['created_utc'], ', should be smaller than child created at ', target_timestamp)
            current_node = parent_node
        else:
            break

    trimmed_ids.add(target_id)

    return list(trimmed_ids)

def trim_tree_recent(conversation, node_index):
    target_node = conversation[node_index]
    target_timestamp = target_node[0]['created_utc']
    target_id = target_node[0]['id']

    filtered_nodes = [node[0] for node in conversation if node[0]['created_utc'] <= target_timestamp]
    filtered_nodes_sorted = sorted(filtered_nodes, key=lambda x: x['created_utc'], reverse=True)
    trimmed_nodes = filtered_nodes_sorted[:25]
    trimmed_ids = {node['id'] for node in trimmed_nodes}
    trimmed_ids.add(target_id)

    return list(trimmed_ids)

def get_value(obj, key):
    if key in obj.keys():
        return obj[key]
    else:
        return ""
