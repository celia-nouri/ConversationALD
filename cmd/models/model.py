import torch
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, to_hetero
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, HeteroData
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import textwrap
import seaborn as sns
from networkx.drawing.nx_agraph import graphviz_layout
from transformers import DistilBertModel, RobertaModel, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from utils.construct_graph import get_graph, get_hetero_graph



all_model_names = ["simple-graph", "distil-class", "roberta-class", "bert-class", "bert-concat", "bert-ctxemb", "fb-roberta-hate", "gat-test", "hetero-graph", "longform-class", "xlmr-class", "modernbert-class"]
#var tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True);
all_base_pretrained_models = [
    "bert-base-uncased",        # BERT (English, uncased)
    "bert-base-cased",          # BERT (English, cased)
    "roberta-base",             # RoBERTa (English)
    "xlm-roberta-base",         # XLM-R (Multilingual)
    "allenai/longformer-base-4096",  # Longformer (English)
    "answerdotai/ModernBERT-base",  # Modern-BERT (English)
    "answerdotai/ModernBERT-large" # Modern-BERT (English)
]

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
        return self.fc2(x)


# Text-Concat model in the paper
class BERTConcat(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_classes=2, hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3, trim="affordance"):
        super(BERTConcat, self).__init__()
        device = get_device()
        self.trim = trim
        
        self.model_name = "bert"
        if "longformer" in pretrained_model_name:
            self.model_name = "longformer"
        elif "xlm-roberta" in pretrained_model_name:
            self.model_name = "xlm-roberta"
        elif "roberta" in pretrained_model_name:
            self.model_name = "roberta"
        print(f"initializing Text Concat model with {self.model_name} text model, pretrained model name is {pretrained_model_name}")
        
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,  # Number of classes for classification
            hidden_dropout_prob=hidden_dropout_prob,  # Dropout for hidden layers
            attention_probs_dropout_prob=attention_probs_dropout_prob  # Dropout for attention layers
        )
        
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            config=self.config
        ).to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    
    def forward(self, context_texts, target_text, labels, max_length=4096):
        device = get_device()
        # Combine context texts and target text using the [SEP] token
        input_text = self.prepare_input(context_texts, target_text, max_length)
        
        # Tokenize and encode the concatenated input text
        encodings = self.tokenizer(
            input_text,
            padding='max_length',  
            truncation=True,       
            max_length=max_length,        
            return_tensors='pt',  
        )
        encodings = encodings.to(device)
        shape_encodings = encodings["input_ids"].shape
        print(f"input ids shape for {self.model_name}'s tokenizer: {shape_encodings}")

        # Handle model-specific inputs
        if self.model_name == "longformer":
            # For Longformer, include global attention
            global_attention_mask = torch.zeros_like(encodings["attention_mask"])
            global_attention_mask[:, 0] = 1  # Apply global attention to the first token
        
            model_output = self.text_model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
                global_attention_mask=global_attention_mask, 
                labels=labels
            )  # Shape: [#comments, 100, hidden_dim]
        else:
            # For BERT, RoBERTa, and XLM-R
            model_output = self.text_model(
                token_type_ids=encodings.get("token_type_ids", None),
                attention_mask=encodings["attention_mask"],
                input_ids=encodings["input_ids"], 
                labels=labels
            )  # Shape: [#comments, 100, hidden_dim]

        return model_output

    def prepare_input(self, context_texts, target_text, max_length=512):
        # Prepare the concatenated input text with [SEP] tokens
        # Assuming context_texts is a list of strings
        combined_context = " [SEP] ".join(context_texts)
        input_text = target_text + " [SEP] " + combined_context
        return input_text

# Embed-Concat model in the paper
class BERTContextEmb(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_classes=2, hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3, max_length=512, trim="affordance"):
        super(BERTContextEmb, self).__init__()
        device = get_device()
        self.trim = trim
        #self.fc_context = torch.nn.Linear(19200, 768).to(device)
        self.fc = torch.nn.Linear(1536, 768).to(device)  # Output one value for binary classification
        # Determine model type based on the pretrained model name
        self.model_name = "bert"
        if "longformer" in pretrained_model_name:
            self.model_name = "longformer"
        elif "xlm-roberta" in pretrained_model_name:
            self.model_name = "xlm-roberta"
        elif "roberta" in pretrained_model_name:
            self.model_name = "roberta"

        # Define a custom configuration for BERT with dropout
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,  # Number of classes for classification
            hidden_dropout_prob=hidden_dropout_prob,  # Dropout for hidden layers
            attention_probs_dropout_prob=attention_probs_dropout_prob  # Dropout for attention layers
        )
        # Load pre-trained text model for sequence classification with the custom config
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            config=self.config
        )

        # Extract model-specific components
        if self.model_name == "longformer":
            self.text_model = model.longformer.to(device)
        elif self.model_name == "bert":
            self.text_model = model.bert.to(device)
        elif self.model_name == "roberta":
            self.text_model = model.roberta.to(device)
        elif self.model_name == "xlm-roberta":
            self.text_model = model.roberta.to(device)  # XLM-Roberta shares architecture with Roberta

        print(f"initializing Context Embed model with {self.model_name} text model, pretrained model name is {pretrained_model_name}")
        
        # Tokenizer for encoding text inputs
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        self.node_classifier = model.classifier.to(device)

    def forward(self, data, target_text, labels, max_length=512):
        device = get_device()
        data = data.to(device)
        mask = data["y_mask"]

        _, _, conv_indices_to_keep, _ = get_graph(data.x_text, mask, with_temporal_edges=False, undirected=False, trim=self.trim)
        all_conv_texts = []
        for i in conv_indices_to_keep:
            all_conv_texts.append(data.x_text[i][0]['body'])

        # construct the conv context embeddings
        cls_embeddings_list = []  # To store CLS embeddings for each text
        for text in all_conv_texts:
            # Tokenize and encode the input text
            encodings = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)
            
            if self.model_name == "longformer":
                global_attention_mask = torch.zeros_like(encodings["attention_mask"])
                global_attention_mask[:, 0] = 1  # Apply global attention to the first token

                model_output = self.text_model(
                    input_ids=encodings["input_ids"],
                    attention_mask=encodings["attention_mask"],
                    global_attention_mask=global_attention_mask
                ).last_hidden_state
            else:
                model_output = self.text_model(
                    input_ids=encodings["input_ids"],
                    attention_mask=encodings["attention_mask"],
                    token_type_ids=encodings.get("token_type_ids", None)
                ).last_hidden_state
            
            # Extract the CLS embedding (first token)
            cls_embeddings = model_output[:, 0, :]
            cls_embeddings_list.append(cls_embeddings)

        context_embed = cls_embeddings_list[0]
        for context in cls_embeddings_list[1:]:
            concat_context = torch.cat([context_embed, context], dim=1)
            context_embed = self.fc(concat_context)
        
        # generate target text embedding
        target_encodings = self.tokenizer(
            target_text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)
            
        if self.model_name == "longformer":
            global_attention_mask = torch.zeros_like(target_encodings["attention_mask"])
            global_attention_mask[:, 0] = 1  # Apply global attention to the first token

            model_output = self.text_model(
                input_ids=target_encodings["input_ids"],
                attention_mask=target_encodings["attention_mask"],
                global_attention_mask=global_attention_mask
            ).last_hidden_state
        else:
            model_output = self.text_model(
                input_ids=target_encodings["input_ids"],
                attention_mask=target_encodings["attention_mask"],
                token_type_ids=target_encodings.get("token_type_ids", None)
            ).last_hidden_state

        target_cls_embeddings = model_output[:, 0, :]
        combined_embed = torch.cat([target_cls_embeddings, context_embed], dim=1)  # Shape: [batch_size, 1536]
        combined_out = self.fc(combined_embed)  # Shape: [batch_size, 768]
        
        if "roberta" in self.model_name or "longformer" == self.model_name:
            combined_out = combined_out.unsqueeze(1)  # Add sequence dimension: [batch_size, seq_length=1, hidden_dim]
            combined_out = self.node_classifier(combined_out).squeeze(1)  # Remove sequence dimension after classification
        else:
            combined_out = self.node_classifier(combined_out)  # Directly use for BERT
        return combined_out


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

# GAT model in the paper
class GATModel(torch.nn.Module):
    def __init__(self, in_channels=768, hidden_channels=768, num_heads_1=8, num_heads_2=1, dropout_1=0.4, dropout_2=0.4, undirected=True, temp_edges=False, num_layers=2, pretrained_model_name='bert-base-uncased', num_classes=2, hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2, trim="affordance", new_trim=False):
        super(GATModel, self).__init__()
        #self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.trim = trim
        self.new_trim = new_trim
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
            if num_layers == 5:
                self.gat3 = GATConv(hidden_channels * num_heads_1, hidden_channels, heads=num_heads_1)
                self.gat4 = GATConv(hidden_channels * num_heads_1, hidden_channels, heads=num_heads_1)
                self.gat5 = GATConv(hidden_channels * num_heads_1, hidden_channels, heads=num_heads_2)
        self.fc = torch.nn.Linear(1536, 768)  # Output one value for binary classification
        # Determine model type based on the pretrained model name
        self.model_name = "bert"
        if "longformer" in pretrained_model_name:
            self.model_name = "longformer"
        elif "xlm-roberta" in pretrained_model_name:
            self.model_name = "xlm-roberta"
        elif "roberta" in pretrained_model_name:
            self.model_name = "roberta"
        elif "answerdotai" in pretrained_model_name:
            self.model_name = "modernbert"
        print(f"initializing gat-test model with {self.model_name} text model, pretrained model name is {pretrained_model_name}")
       

        # Define a custom configuration for BERT with dropout
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,  # Number of classes for classification
            hidden_dropout_prob=hidden_dropout_prob,  # Dropout for hidden layers
            attention_probs_dropout_prob=attention_probs_dropout_prob  # Dropout for attention layers
        )
        # Load pre-trained text model for sequence classification with the custom config
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            config=self.config
        )

        # Extract model-specific components
        if self.model_name == "longformer":
            self.text_model = model.longformer
        elif self.model_name == "bert":
            self.text_model = model.bert
        elif self.model_name == "roberta":
            self.text_model = model.roberta
        elif self.model_name == "xlm-roberta":
            self.text_model = model.roberta  # XLM-Roberta shares architecture with Roberta

        # Other shared components
        self.node_classifier = model.classifier
                
        # Tokenizer for encoding text inputs
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        self.undirected = undirected
        self.temp_edges = temp_edges
        self.num_layers = num_layers
        print('Inside GAT model init num layers is ', self.num_layers)
    

    def forward(self, data, max_length=512, gpu=False):   
        # Save the original batch size
        data = data.to('cpu')
        mask = data["y_mask"]
        _, edges_dic_num, conv_indices_to_keep, my_new_mask_idx = get_graph(data.x_text, mask, with_temporal_edges=self.temp_edges, undirected=self.undirected, trim=self.trim, new_trim=self.new_trim)
        assert len(edges_dic_num.keys()) <= 1, "length of edges dic num is greater than 1"
        edge_list = []
        for k in edges_dic_num.keys():
            edge_list = edges_dic_num[k]
            edge_list = sorted(edge_list, key=lambda x: (x[0], x[1]))
            edge_list = torch.tensor(edge_list)
        x = data.x
        y = data.y

        texts = []
        for i in conv_indices_to_keep:
            texts.append(data.x_text[i][0]['body'])

        device_bert = get_device()
        if gpu: 
            device_bert = torch.device('cuda:2')
        self.text_model = self.text_model.to(device_bert)

        encodings = self.tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt').to(device_bert)

        # Handle model-specific inputs
        if self.model_name == "longformer":
            # For Longformer, include global attention
            global_attention_mask = torch.zeros_like(encodings["attention_mask"])
            global_attention_mask[:, 0] = 1  # Apply global attention to the first token

            model_output = self.text_model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
                global_attention_mask=global_attention_mask
            ).last_hidden_state  # Shape: [#comments, 100, hidden_dim]
        else:
            # For BERT, RoBERTa, and XLM-R
            model_output = self.text_model(
                token_type_ids=encodings.get("token_type_ids", None),
                attention_mask=encodings["attention_mask"],
                input_ids=encodings["input_ids"]
            ).last_hidden_state  # Shape: [#comments, 100, hidden_dim]

        model_output = model_output.to('cpu')
        g_data = Data(x=model_output, edge_index=edge_list.t().contiguous())        
        x, edge_list = g_data.x, g_data.edge_index
        cls_embeddings = model_output[:, 0, :] 
        edge_indices = edge_list
        
        device_graph = get_device() 
        if gpu:
            device_graph = torch.device('cuda:1')

        # if there are no edges in the graph, ignore the GAT layers
        if edge_indices.numel() == 0:
            x = cls_embeddings
        else:
            # GAT layer 1
            self.gat1 = self.gat1.to(device_graph)
            cls_embeddings = cls_embeddings.to(device_graph)
            edge_indices = edge_indices.to(device_graph)
            x, att_weight = self.gat1(cls_embeddings, edge_indices, return_attention_weights=True)
            draw_graphattention(x, edge_indices, att_weight, texts, out_name="layer1")
            x = F.elu(x)
            if self.num_layers > 1:
                # GAT layer 2
                self.gat2 = self.gat2.to(device_graph)
                x, att_weight2 = self.gat2(x, edge_indices, return_attention_weights=True)
                draw_graphattention(x, edge_indices, att_weight2, texts, out_name="layer2")
                if self.num_layers == 3:
                    self.gat3 = self.gat3.to(device_graph)
                    # GAT layer 3
                    x = F.elu(x)
                    x, att_weight3 = self.gat3(x, edge_indices, return_attention_weights=True)
                    draw_graphattention(x, edge_indices, att_weight3, texts, out_name="layer3")

                elif self.num_layers == 4:
                    self.gat3 = self.gat3.to(device_graph)
                    self.gat4 = self.gat4.to(device_graph)
                    # GAT layer 3
                    x = F.elu(x)
                    x, att_weight3 = self.gat3(x, edge_indices, return_attention_weights=True)
                    # GAT layer 4
                    x = F.elu(x) 
                    x, att_weight4 = self.gat4(x, edge_indices, return_attention_weights=True)
                elif self.num_layers == 5:
                    self.gat3 = self.gat3.to(device_graph)
                    self.gat4 = self.gat4.to(device_graph)
                    self.gat5 = self.gat5.to(device_graph)
                    # GAT layer 3
                    x = F.elu(x)
                    x, att_weight3 = self.gat3(x, edge_indices, return_attention_weights=True)
                    # GAT layer 4
                    x = F.elu(x) 
                    x, att_weight4 = self.gat4(x, edge_indices, return_attention_weights=True)
                    # GAT layer 5
                    x = F.elu(x)
                    x, att_weight5 = self.gat5(x, edge_indices, return_attention_weights=True)
        device_classification = get_device()
        if gpu:
            device_classification = torch.device('cuda:0')               
        x_gemb = x[my_new_mask_idx].unsqueeze(0).to(device_classification)
        x_emb = cls_embeddings[my_new_mask_idx].unsqueeze(0).to(device_classification)

        concat_out = torch.cat([x_gemb, x_emb], dim=1)

        # SHAPE OF X: [#vertices, 768=input_dim] --> [#vertices, 64=hidden_dim] --> [#vertices, 1] --> [#vertices]

        # Classification layer (binary classification)
        self.fc = self.fc.to(device_classification)
        self.node_classifier = self.node_classifier.to(device_classification)
        out = self.fc(concat_out) 

        if "roberta" in self.model_name or "longformer" == self.model_name:
            out = out.unsqueeze(1)  # Add sequence dimension: [batch_size, seq_length=1, hidden_dim]
            out = self.node_classifier(out).squeeze(1)  # Remove sequence dimension after classification
        else:
            out = self.node_classifier(out)  # Directly use for BERT
        return out

def wrap_text(text, width=10):
    """Wrap text into multiple lines for better readability inside nodes."""
    new_text = textwrap.wrap(text[:15], width)
    if len(text) > 15:
        new_text = textwrap.wrap(text[:15] + "...", width)

    return "\n".join(new_text)


def visualize_attention_graph(x, edge_index, attn_values, node_texts, out_name="gat_attention_graph"):
    G = to_networkx(Data(x=x, edge_index=edge_index), to_undirected=False)
    attn_values = attn_values.to(torch.float32).squeeze(1)  

    edge_attention = {
        (int(edge_index[0, i]), int(edge_index[1, i])): float(attn_values[i].mean())  
        for i in range(attn_values.size(0))
    }

    num_nodes = x.shape[0]
    attention_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for (src, tgt), weight in edge_attention.items():
        attention_matrix[src, tgt] = weight

    attn_df = pd.DataFrame(attention_matrix.numpy())
    plt.figure(figsize=(10, 6), dpi=300)
    pos = graphviz_layout(G, prog="dot")  

    min_weight = min(edge_attention.values()) if edge_attention else 0
    max_weight = max(edge_attention.values()) if edge_attention else 1

    norm = plt.Normalize(vmin=min_weight, vmax=max_weight)
    edge_colors = [plt.cm.Blues(norm(edge_attention.get((u, v), 0))) for u, v in G.edges()]
    edge_widths = [2 + 3 * norm(edge_attention.get((u, v), 0)) for u, v in G.edges()]  # Adjust thickness

    last_node = list(G.nodes())[-1]  
    node_colors = ["#ADD8E6" if node != last_node else "#FFA500" for node in G.nodes()]  

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200, edgecolors="black")
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, edge_cmap=plt.cm.Blues, width=edge_widths, alpha=0.8)
    trimmed_labels = {node: wrap_text(node_texts[node]) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=trimmed_labels, font_size=6, font_family="serif", font_color="black")

    edge_labels = {k: f"{v:.2f}" for k, v in edge_attention.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, 
                                 font_family="serif", bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    plt.axis("off")

    # Save to file
    plt.savefig(f"{out_name}_attgraph.pdf", format="pdf", bbox_inches="tight", dpi=300)

    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(attn_df, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5, cbar=True)
    plt.title("Graph Attention Weights")
    
    plt.savefig(f"{out_name}_heatmap.pdf", format="pdf", bbox_inches="tight", dpi=300)


def draw_graphattention(x, edge_index, attn_weights, node_texts, out_name=""):
    edge_index, attn_values = attn_weights  
    attn_values = attn_values.to(torch.float32)
    visualize_attention_graph(x, edge_index, attn_values, node_texts, out_name=out_name)
   

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
    def __init__(self, model=GATModel, in_channels=768, hidden_channels=768, num_heads_1=8, num_heads_2=1, dropout_1=0.6, dropout_2=0.6, undirected=True, temp_edges=False):
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
    pretrained_model_name = args.pretrained_model_name
    assert pretrained_model_name in all_base_pretrained_models, "Invalid model name: {}".format(pretrained_model_name)
    device = get_device()
    # Instantiate your model
    model = ""

    if model_name == "simple-graph":
        model = SimpleGraphModel(in_channels=768, hidden_channels=hidden_channels, num_heads=num_heads)
    elif model_name == "distil-class":
        model = DistilBERTClass()

    elif model_name == "roberta-class":
        custom_config = AutoConfig.from_pretrained(
            "roberta-base",
            num_labels=2,                    
            hidden_dropout_prob=0.3,         
            attention_probs_dropout_prob=0.3 
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base",
            config=custom_config
        )
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
    elif model_name == "longform-class":
        custom_config = AutoConfig.from_pretrained(
            "allenai/longformer-base-4096",  # Longformer model
            num_labels=2,                    # Number of output labels for classification
            hidden_dropout_prob=0.3,         # Dropout for hidden layers
            attention_probs_dropout_prob=0.3 # Dropout for attention layers
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            "allenai/longformer-base-4096",  # Longformer model
            config=custom_config
        )
    elif model_name == "xlmr-class":
        custom_config = AutoConfig.from_pretrained(
            "xlm-roberta-base",               # XLM-R model
            num_labels=2,                     # Number of output labels for classification
            hidden_dropout_prob=0.3,          # Dropout for hidden layers
            attention_probs_dropout_prob=0.3  # Dropout for attention layers
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            "xlm-roberta-base",  # XLM-R model
            config=custom_config
        )
    elif model_name == "modernbert-class":
        custom_config = AutoConfig.from_pretrained(
            "answerdotai/ModernBERT-base",               # Modern BERT model
            num_labels=2,                     # Number of output labels for classification
            hidden_dropout_prob=0.3,          # Dropout for hidden layers
            attention_probs_dropout_prob=0.3  # Dropout for attention layers
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            "answerdotai/ModernBERT-base",  #ModernBERT model
            config=custom_config,
        )

    elif model_name == "bert-concat":
        trim = args.trim
        model = BERTConcat(pretrained_model_name=pretrained_model_name, trim=trim)

    elif model_name == "bert-ctxemb":
        trim = args.trim
        model = BERTContextEmb(pretrained_model_name=pretrained_model_name, trim=trim)

    elif model_name == 'fb-roberta-hate':
        model = AutoModelForSequenceClassification.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")

    elif model_name == 'gat-test':
        undirected = args.undirected
        temp_edges = args.temp_edges
        num_layers = args.num_layers
        trim = args.trim
        new_trim = args.new_trim
        model = GATModel(undirected=undirected, temp_edges=temp_edges, num_layers=num_layers, pretrained_model_name=pretrained_model_name, trim=trim, new_trim=new_trim)
    elif model_name == 'hetero-graph':
        undirected = args.undirected
        temp_edges = args.temp_edges
        model = HeteroGAT(undirected=undirected, temp_edges=temp_edges)
    else:
        print(f"Model name {model_name} did not match any of the available models. \n Use a model name from this list {all_model_names}")
        return None
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
