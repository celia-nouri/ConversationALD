import torch
import torch_geometric.transforms as T
from torch_geometric.nn import RGCNConv, GraphConv, GATConv, to_hetero
from torch_geometric.data import Data, HeteroData
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, DistilBertModel, RobertaModel, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from utils.construct_graph import get_graph, get_hetero_graph
from torchsummary import summary

from models.multimodal_transformer import GraphormerModel, GraphormerEncoder



all_model_names = ["simple-graph", "distil-class", "text-class", "roberta-class", "bert-class", "bert-concat", "fb-roberta-hate", "img-text-transformer", "text-graph-transformer", "multimodal-transformer", "gat-model", "gat-test", "hetero-graph"]
#var tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True);

# DistilBERT Classifier model 
class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    
class RoBERTaHateClassifier(nn.Module):
    def __init__(self, roberta_model_name="roberta-base", hidden_dim=256, output_dim=1, dropout_prob=0.1):
        super(RoBERTaHateClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.fc1 = nn.Linear(self.roberta.config.hidden_size, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token is at index 0
        x = torch.relu(self.fc1(cls_output))
        x = self.dropout(x)
        # Note: we use F.binary_cross_entropy_with_logits as the criterion for classification 
        # hence, we should NOT apply a softmax or sigmoid after the output layer
        return self.fc2(x)

# Simple Text Classifier model 
class SimpleTextClassifier(torch.nn.Module):

    def __init__(self, input_dim=768, hidden_dim=256, output_dim=1, dropout=0.1):
        super(SimpleTextClassifier, self).__init__()
        
        # Linear layer to map input embeddings to hidden dimension
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(0.1)        
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x= self.dropout(x)        
        x = self.fc2(x)        
        x = torch.sigmoid(x)
        return x


class BERTConcat(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_classes=2, hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3):
        super(BERTConcat, self).__init__()
        device = get_device()
        
        # Define a custom configuration for BERT with dropout
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,  # Number of classes for classification
            hidden_dropout_prob=hidden_dropout_prob,  # Dropout for hidden layers
            attention_probs_dropout_prob=attention_probs_dropout_prob  # Dropout for attention layers
        )
        
        # Load pre-trained BERT model for sequence classification with the custom config
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            config=self.config
        ).to(device)
        
        # Tokenizer for encoding text inputs
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    
    def forward(self, context_texts, target_text, labels):
        device = get_device()
        # Combine context texts and target text using the [SEP] token
        input_text = self.prepare_input(context_texts, target_text)
        
        # Tokenize and encode the concatenated input text
        encoding = self.tokenizer(
            input_text,
            padding='max_length',  # Pads up to the max length
            truncation=True,       # Truncate context if needed
            max_length=512,        # BERT's max input size
            return_tensors='pt',   # Return PyTorch tensors
        )
        encoding = encoding.to(device)
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        # Pass the inputs through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        return outputs
    
    def prepare_input(self, context_texts, target_text):
        # Prepare the concatenated input text with [SEP] tokens
        # Assuming context_texts is a list of strings
        combined_context = " [SEP] ".join(context_texts)

        # estimate of number of tokens to keep.
        max_ctx_len = 512 - int(len(" [SEP] " + target_text) / 4)
        if max_ctx_len > 0:
            combined_context = combined_context[:max_ctx_len]
        else: 
            combined_context = ""
        
        # Concatenate context and target text
        input_text = combined_context + " [SEP] " + target_text
        
        return input_text

class SimpleGraphModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads=1):
        super(SimpleGraphModel, self).__init__()
        #self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
        self.fc = torch.nn.Linear(hidden_channels, 1)  # Output one value for binary classification
    
    
    def forward(self, x, edge_indices):   
        # Save the original batch size
        #batch_size = x.size(0)
        device = x.device
        edge_indices = edge_indices.to(device)
    
        x = x.squeeze(0)
        edge_indices = edge_indices.squeeze(0)
        # SHAPE OF X: [#vertices, 768=input_dim] --> [#vertices, 64=hidden_dim] --> [#vertices, 1] --> [#vertices]
        
        edge_indices = edge_indices.t().contiguous()
  
        x = self.gat1(x, edge_indices)
        x = F.relu(x)
 
        # GAT layer 2
        x = self.gat2(x, edge_indices)
        x = F.relu(x)

        # Classification layer (binary classification)
        out = self.fc(x) 
        out = out.squeeze(-1)        
        return out

# Credits, code from DialogueGCN https://github.com/declare-lab/conv-emotion/blob/master/DialogueGCN/model.py
class GraphNetwork(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_relations, max_seq_len, hidden_size=64, dropout=0.5, no_cuda=False):
        """
        The Speaker-level context encoder in the form of a 2 layer GCN.
        """
        super(GraphNetwork, self).__init__()
        
        self.conv1 = RGCNConv(num_features, hidden_size, num_relations, num_bases=30)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        #self.matchatt = MatchingAttention(num_features+hidden_size, num_features+hidden_size, att_type='general2')
        self.linear   = nn.Linear(num_features+hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.smax_fc  = nn.Linear(hidden_size, num_classes)
        self.no_cuda = no_cuda 

    def forward(self, x, edge_index, edge_norm, edge_type, seq_lengths, umask, nodal_attn, avec):
        
        out = self.conv1(x, edge_index, edge_type, edge_norm)
        out = self.conv2(out, edge_index)
        emotions = torch.cat([x, out], dim=-1)
        log_prob = classify_node_features(emotions, seq_lengths, umask, self.matchatt, self.linear, self.dropout, self.smax_fc, nodal_attn, avec, self.no_cuda)
        return log_prob

def classify_node_features(emotions, seq_lengths, umask, matchatt_layer, linear_layer, dropout_layer, smax_fc_layer, nodal_attn, avec, no_cuda):
    """
    Function for the final classification, as in Equation 7, 8, 9. in the paper.
    """

    if nodal_attn:

        #emotions = attentive_node_features(emotions, seq_lengths, umask, matchatt_layer, no_cuda)
        hidden = F.relu(linear_layer(emotions))
        hidden = dropout_layer(hidden)
        hidden = smax_fc_layer(hidden)

        if avec:
            return torch.cat([hidden[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])

        log_prob = F.log_softmax(hidden, 2)
        log_prob = torch.cat([log_prob[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
        return log_prob

    else:

        hidden = F.relu(linear_layer(emotions))
        hidden = dropout_layer(hidden)
        hidden = smax_fc_layer(hidden)

        if avec:
            return hidden

        log_prob = F.log_softmax(hidden, 1)
        return log_prob

# New graph model
class GATModel(torch.nn.Module):
    def __init__(self, in_channels=768, hidden_channels=768, num_heads_1=8, num_heads_2=1, dropout_1=0.6, dropout_2=0.6):
        super(GATModel, self).__init__()
        #self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads_1, dropout=dropout_1)
        self.gat2 = GATConv(hidden_channels * num_heads_1, hidden_channels, heads=num_heads_2)
        self.fc = torch.nn.Linear(1536, 768)  # Output one value for binary classification
        bert = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2,
        )
        self.text_model = bert.bert
        self.text_pooler = self.text_model.pooler
        self.node_classifier = bert.classifier
        self.bert_dropout = bert.dropout
    

    def forward(self, data):   
        # Save the original batch size
        #batch_size = x.size(0)
        x = data.x
        device = get_device()
        edge_indices = data.edge_index.permute(1, 0)
        mask = data["y_mask"]

        # x["token_type_ids"] is [#comments, 100]
        x_token_type_ids = x["token_type_ids"] # [#comments, 100]
        x_attention_mask = x["attention_mask"] # [#comments, 100]
        x_input_ids = x["input_ids"] # [#comments, 100]
        bert_output = self.text_model(
            token_type_ids=x_token_type_ids,
            attention_mask=x_attention_mask,
            input_ids=x_input_ids,
        ).last_hidden_state # [#comments, 100, 768]
        
        g_data = Data(x=bert_output, edge_index=edge_indices.t().contiguous())
        x, edge_indices = g_data.x, g_data.edge_index
        x = x.to(device)
        edge_indices = edge_indices.to(device)
        cls_embeddings = bert_output[:, 0, :] 
         # GAT layer 1
        x = self.gat1(cls_embeddings, edge_indices)
        x = F.elu(x)
         # GAT layer 2
        x = self.gat2(x, edge_indices)
        x_gemb = x[mask, :]
        x_emb = cls_embeddings[mask, :]

        concat_out = torch.cat([x_gemb, x_emb], dim=1)
        # SHAPE OF X: [#vertices, 768=input_dim] --> [#vertices, 64=hidden_dim] --> [#vertices, 1] --> [#vertices]
        

        # Classification layer (binary classification)
        out = self.fc(concat_out) 
        #out = self.text_pooler(out)
        #out = self.bert_dropout(out)
        out = self.node_classifier(out)
        return out

# New graph model
class GATTest(torch.nn.Module):
    def __init__(self, in_channels=768, hidden_channels=768, num_heads_1=8, num_heads_2=1, dropout_1=0.4, dropout_2=0.4, undirected=True, temp_edges=False, num_layers=2):
        super(GATTest, self).__init__()
        #self.fc1 = nn.Linear(in_channels, hidden_channels)
        if num_layers == 1:
            self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads_2, dropout=dropout_1)
        else:
            self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads_1, dropout=dropout_1)
            if num_layers == 2:
                self.gat2 = GATConv(hidden_channels * num_heads_1, hidden_channels, heads=num_heads_2)
            else:
                self.gat2 = GATConv(hidden_channels * num_heads_1, hidden_channels, heads=num_heads_1)
            if num_layers == 3:
                self.gat3 = GATConv(hidden_channels * num_heads_1, hidden_channels, heads=num_heads_2)
            if num_layers == 4:
                self.gat3 = GATConv(hidden_channels * num_heads_1, hidden_channels, heads=num_heads_1)
                self.gat4 = GATConv(hidden_channels * num_heads_1, hidden_channels, heads=num_heads_2)
        self.fc = torch.nn.Linear(1536, 768)  # Output one value for binary classification
        bert = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            hidden_dropout_prob=0.3,
            attention_probs_dropout_prob=0.3,
        )
        self.text_model = bert.bert
        self.text_pooler = self.text_model.pooler
        self.node_classifier = bert.classifier
        self.bert_dropout = bert.dropout
        self.undirected = undirected
        self.temp_edges = temp_edges
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.num_layers = num_layers
        print('Inside GATTEST init num layers is ', self.num_layers)
    


    def forward(self, data):   
        # Save the original batch size
        #batch_size = x.size(0)
        mask = data["y_mask"]
        _, edges_dic_num, conv_indices_to_keep, my_new_mask_idx = get_graph(data.x_text, mask, with_temporal_edges=self.temp_edges, undirected=self.undirected)
        assert len(edges_dic_num.keys()) <= 1, "length of edges dic num is greater than 1"
        edge_list = []
        device = get_device()
        for k in edges_dic_num.keys():
            edge_list = edges_dic_num[k]
            edge_list = sorted(edge_list, key=lambda x: (x[0], x[1]))
            edge_list = torch.tensor(edge_list).to(device)
        x = data.x
        y = data.y
        #edge_indices = data.edge_index.permute(1, 0)
        #are_equal = torch.equal(edge_list, edge_indices)
        #print(f"Are the tensors equal? {are_equal}")
        # YES. The undirected option without temporal edges is equal to edge_indices (the edge list returned by the mDT codebase.)

        texts = []
        for i in conv_indices_to_keep:
            texts.append(data.x_text[i][0]['body'])

        labels = y.long().to(device)
        encodings = self.tokenizer(texts, truncation=True, padding='max_length', max_length=300, return_tensors='pt').to(device)

        bert_output = self.text_model(
            token_type_ids=encodings["token_type_ids"],
            attention_mask=encodings["attention_mask"],
            input_ids=encodings["input_ids"],
        ).last_hidden_state # [#comments, 100, 768]

        
        g_data = Data(x=bert_output, edge_index=edge_list.t().contiguous())
        x, edge_list = g_data.x, g_data.edge_index
        x = x.to(device)
        edge_indices = edge_list.to(device)
        cls_embeddings = bert_output[:, 0, :] 
        # if there are no edges in the graph, ignore the GAT layers
        if edge_indices.numel() == 0:
            x = cls_embeddings
        else:
            # GAT layer 1
            x = self.gat1(cls_embeddings, edge_indices)
            x = F.elu(x)
            if self.num_layers > 1:
                # GAT layer 2
                x = self.gat2(x, edge_indices)
                if self.num_layers == 3:
                    # GAT layer 3
                    x = F.elu(x)
                    x = self.gat3(x, edge_indices)
                elif self.num_layers == 4:
                    # GAT layer 3
                    x = F.elu(x)
                    x = self.gat3(x, edge_indices)
                    # GAT layer 4
                    x = F.elu(x) 
                    x = self.gat4(x, edge_indices)               
        x_gemb = x[my_new_mask_idx].unsqueeze(0)
        x_emb = cls_embeddings[my_new_mask_idx].unsqueeze(0)

        concat_out = torch.cat([x_gemb, x_emb], dim=1)

        # SHAPE OF X: [#vertices, 768=input_dim] --> [#vertices, 64=hidden_dim] --> [#vertices, 1] --> [#vertices]
        

        # Classification layer (binary classification)
        out = self.fc(concat_out) 
        #out = self.text_pooler(out)
        #out = self.bert_dropout(out)
        out = self.node_classifier(out)    
        return out



# graphModel is the internal graph network used by both heterogenous and homogenous graph models.
class graphModel(torch.nn.Module):
    def __init__(self, in_channels=768, hidden_channels=768, num_heads_1=8, num_heads_2=1, dropout_1=0.6, dropout_2=0.6):
        super(graphModel, self).__init__()
        #self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads_1, dropout=dropout_1, add_self_loops=False)
        self.gat2 = GATConv(hidden_channels * num_heads_1, hidden_channels, heads=num_heads_2, add_self_loops=False)

    def forward(self, x, edge_list):   
         # GAT layer 1
        x = self.gat1(x, edge_list)
        x = F.elu(x)
         # GAT layer 2
        x = self.gat2(x, edge_list)
        return x


class HeteroGAT(torch.nn.Module):
    def __init__(self, model=GATTest, in_channels=768, hidden_channels=768, num_heads_1=8, num_heads_2=1, dropout_1=0.6, dropout_2=0.6, undirected=True, temp_edges=False):
        super(HeteroGAT, self).__init__()
        self.fc = torch.nn.Linear(1536, 768)  # Output one value for binary classification
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        bert = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            hidden_dropout_prob=0.3,
            attention_probs_dropout_prob=0.3,
        )
        # enable gradient checkpointing to reduce CUDA memory and send more work to compute 
        # this was added after I encountered CUDA OOM errors when training HeteroGAT on CAD data. 
        bert.gradient_checkpointing_enable()
        self.text_model = bert.bert
        self.text_pooler = self.text_model.pooler
        self.node_classifier = bert.classifier
        self.bert_dropout = bert.dropout
        self.undirected = undirected
        self.temp_edges = temp_edges

        self.graph_model = graphModel()

    def forward(self, data):   
        device = get_device()

        # use model parallelism to split the data and steps on different devices 
        # Move the data object to the CPU at the beginning
        data = data.to('cpu')
        mask = data["y_mask"]

        num_comment_nodes, comments_edges_dic_num, num_users, user_to_comments_edges, conv_indices_to_keep, my_new_mask_idx = get_hetero_graph(data.x_text, mask, with_temporal_edges=self.temp_edges)
        assert conv_indices_to_keep[my_new_mask_idx] == mask.nonzero(as_tuple=True)[0], "error: the new index should be equivalent to the old one."
        user_to_comments_edges = torch.tensor(user_to_comments_edges, dtype=torch.long).permute(1,0).to('cpu')
        comments_to_user_edges = user_to_comments_edges.flip(0).to('cpu')

        hetero_graph = HeteroData()

        # Add node indices:
        hetero_graph["comment"].node_id = torch.arange(num_comment_nodes)
        hetero_graph["user"].node_id = torch.arange(num_users)

        # Add edge indices:
        if len(user_to_comments_edges) > 0:
            hetero_graph["user", "posts", "comment"].edge_index = user_to_comments_edges #.t().contiguous()
        if len(comments_to_user_edges) > 0:    
            hetero_graph["comment", "posted_by", "user"].edge_index = comments_to_user_edges
            
        assert len(comments_edges_dic_num.keys()) == 1, "There should be only one dictionary key"
        comments_edge_list = []
        for k in comments_edges_dic_num.keys():
            comments_edge_list = comments_edges_dic_num[k]
            comments_edge_list = sorted(comments_edge_list, key=lambda x: (x[0], x[1]))
            if len(comments_edge_list) > 0:
                comments_edge_list = torch.tensor(comments_edge_list, dtype=torch.long).permute(1,0).to('cpu')
        if len(comments_edge_list) > 0:
            hetero_graph["comment", "replies", "comment"].edge_index = comments_edge_list #.t().contiguous()
  
        # We also need to make sure to add the reverse edges from movies to users
        # in order to let a GNN be able to pass messages in both directions.
        # We can leverage the `T.ToUndirected()` transform for this from PyG:
        if self.undirected:
            hetero_graph = T.ToUndirected()(hetero_graph)
        x = data.x

        # Model //sm: BERT Model runs on cuda:2
        device_bert = torch.device('cuda:2')
        texts = []
        for i in conv_indices_to_keep:
            texts.append(data.x_text[i][0]['body'])
        y = data.y

        labels = y.long().to(device)
        encodings = self.tokenizer(texts, truncation=True, padding='max_length', max_length=300, return_tensors='pt').to(device_bert)

        self.text_model = self.text_model.to(device_bert)

        bert_output = self.text_model(
            token_type_ids=encodings["token_type_ids"],
            attention_mask=encodings["attention_mask"],
            input_ids=encodings["input_ids"],
        ).last_hidden_state # [#comments, 100, 768]

        #x_token_type_ids = x["token_type_ids"][conv_indices_to_keep].to(device_bert) # [#comments, 100]
        #x_attention_mask = x["attention_mask"][conv_indices_to_keep].to(device_bert) # [#comments, 100]
        #x_input_ids = x["input_ids"][conv_indices_to_keep].to(device_bert) # [#comments, 100]
        
        #bert_output = self.text_model(
        #    token_type_ids=x_token_type_ids,
        #    attention_mask=x_attention_mask,
        #    input_ids=x_input_ids,
        #).last_hidden_state # [#comments, 100, 768]
        cls_embeddings = bert_output[:, 0, :] 

        # Add node features: in CPU
        hetero_graph["comment"].x = cls_embeddings.to('cpu')
        # must be the same feature size for each node type
        hetero_graph["user"].x = torch.ones((num_users, 768)).to('cpu') 
        #TODO(celia): add scores as edge features
    
        # Model //sm: Graph Model on cuda:1
        device_graph = torch.device('cuda:1')
        hetero_graph = hetero_graph.to(device_graph, non_blocking=True)
        self.graph_model = self.graph_model.to(device_graph)
        model = to_hetero(self.graph_model, hetero_graph.metadata(), aggr='sum')
        model = model.to(device_graph)
        x = model(hetero_graph.x_dict, hetero_graph.edge_index_dict)

        comments = x['comment']
        x_gemb = comments[my_new_mask_idx].unsqueeze(0).to(device_graph) 
        x_emb = cls_embeddings[my_new_mask_idx].unsqueeze(0).to(device_graph)
        concat_out = torch.cat([x_gemb, x_emb], dim=1)

        # SHAPE OF X: [#vertices, 768=input_dim] --> [#vertices, 64=hidden_dim] --> [#vertices, 1] --> [#vertices]
        
        # Model //sm: Classification layer (binary classification) on cuda:0
        device_classification = torch.device('cuda:0')
        concat_out = concat_out.to(device_classification)
        self.fc = self.fc.to(device_classification)
        self.node_classifier = self.node_classifier.to(device_classification)
        out = self.fc(concat_out) 
        out = self.node_classifier(out)    
        return out
        
def get_model(args, model_name, hidden_channels=64, num_heads=1):
    assert model_name in all_model_names, "Invalid model name: {}".format(model_name)
    device = get_device()
    # Instantiate your model
    model = ""

    if model_name == "simple-graph":
        model = SimpleGraphModel(in_channels=768, hidden_channels=hidden_channels, num_heads=num_heads)
    elif model_name == "distil-class":
        model = DistilBERTClass()
    elif model_name == "text-class":
        model = SimpleTextClassifier()
    elif model_name == "roberta-class":
        model = RoBERTaHateClassifier()
    elif model_name == "bert-class":
        custom_config = AutoConfig.from_pretrained(
            "bert-base-uncased",
            num_labels=2,                    
            hidden_dropout_prob=0.3,         
            attention_probs_dropout_prob=0.3 
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            config=custom_config
        )
    elif model_name == "bert-concat":
        model = BERTConcat()
    elif model_name == 'fb-roberta-hate':
        model = AutoModelForSequenceClassification.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
    elif model_name == 'img-text-transformer':
        encoder = GraphormerEncoder(args, with_graph=False, enable_images=True, device=device).to(device)
        model = GraphormerModel(args, encoder)
    elif model_name == 'text-graph-transformer':
        encoder = GraphormerEncoder(args, with_graph=True, enable_images=False, device=device).to(device)
        model = GraphormerModel(args, encoder)
    elif model_name == 'multimodal-transformer':
        encoder = GraphormerEncoder(args, with_graph=True, enable_images=True, device=device).to(device)
        model = GraphormerModel(args, encoder)
    elif model_name == 'gat-model':
        model = GATModel()
        undirected = args.undirected
        temp_edges = args.temp_edges
    elif model_name == 'gat-test':
        undirected = args.undirected
        temp_edges = args.temp_edges
        num_layers = args.num_layers
        model = GATTest(undirected=undirected, temp_edges=temp_edges, num_layers=num_layers)
    elif model_name == 'hetero-graph':
        undirected = args.undirected
        temp_edges = args.temp_edges
        model = HeteroGAT(undirected=undirected, temp_edges=temp_edges)
    else:
        model = SimpleTextClassifier()
    if model_name != "hetero-graph":
        model = model.to(device)
    return model


def get_device():
    device = torch.device('cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS")
        return device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
        return device
    print("Using CPU")
    return device

def log_memory_usage():
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"Allocated Memory: {allocated_memory:.2f} GB")
    print(f"Reserved Memory: {reserved_memory:.2f} GB")
