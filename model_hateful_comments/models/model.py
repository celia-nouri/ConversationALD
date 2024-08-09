import torch
from torch_geometric.nn import RGCNConv, GraphConv, GATConv
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, DistilBertModel, RobertaModel, AutoModelForSequenceClassification
from utils.construct_graph import get_graph

from models.multimodal_transformer import GraphormerModel, GraphormerEncoder

all_model_names = ["simple-graph", "distil-class", "text-class", "roberta-class", "bert-class", "fb-roberta-hate", "img-text-transformer", "text-graph-transformer", "multimodal-transformer", "gat-model", "gat-test"]
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
    def __init__(self, in_channels=768, hidden_channels=768, num_heads_1=8, num_heads_2=1, dropout_1=0.6, dropout_2=0.6):
        super(GATTest, self).__init__()
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
        graph = get_graph(data.x_text)
        x = data.x
        device = get_device()
        #graph = get_graph()
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
        print('output of prediction: ', out)
    
        return out
def get_model(args, model_name, hidden_channels=64, num_heads=1):
    assert model_name in all_model_names, "Invalid model name: {}".format(model_name)
    device = get_device()
    # Instantiate your model
    model = SimpleTextClassifier()
    if model_name == "simple-graph":
        model = SimpleGraphModel(in_channels=768, hidden_channels=hidden_channels, num_heads=num_heads)
    elif model_name == "distil-class":
        model = DistilBERTClass()
    elif model_name == "text-class":
        model = SimpleTextClassifier()
    elif model_name == "roberta-class":
        model = RoBERTaHateClassifier()
    elif model_name == "bert-class":
        model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
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
    elif model_name == 'gat-test':
        model = GATTest()
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
    