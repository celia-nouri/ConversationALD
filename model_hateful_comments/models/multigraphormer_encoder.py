from typing import Optional, Tuple

import torch
import torch.nn as nn
from fairseq.modules import FairseqDropout, LayerDropModuleList, LayerNorm
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

from .multihead_attention import MultiheadAttention
from .graphormer_layers import GraphNodeFeature, GraphAttnBias
from .graphormer_graph_encoder_layer import  GraphEncoderStack
from .fusionlayers import GraphFusionStack
from transformers import AutoModel, AutoModelForSequenceClassification


def init_graphormer_params(module):
    """
    Initialize the weights specific to the Graphormer Model.
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)

class MultiGraphormerGraphEncoder(nn.Module):
    def __init__(
        self,
        num_atoms: int,
        num_in_degree: int,
        num_out_degree: int,
        num_edges: int,
        num_spatial: int,
        num_edge_dis: int,        
        num_bottle_neck: int,
        num_fusion_layers: int,
        edge_type: str,
        multi_hop_max_dist: int,
        num_fusion_stack: int = 1,
        num_graph_stack: int = 1,
        num_encoder_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,
        num_attention_heads: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        encoder_normalize_before: bool = False,
        pre_layernorm: bool = False,
        apply_graphormer_init: bool = False,
        activation_fn: str = "gelu",
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        freeze_initial_encoders: bool = False,
        device=None,
        with_graph: bool = False,
        enable_images: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.with_graph = with_graph
        self.enable_images = enable_images
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = layerdrop
        self.embedding_dim = embedding_dim
        self.apply_graphormer_init = apply_graphormer_init
        self.traceable = traceable
        num_encoder_layers = num_fusion_layers

        # define the graphormer components if with_graph is True
        if with_graph:
            self.graph_node_feature = GraphNodeFeature(
                num_heads=num_attention_heads,
                num_atoms=num_atoms,
                num_in_degree=num_in_degree,
                num_out_degree=num_out_degree,
                hidden_dim=embedding_dim,
                n_layers=num_encoder_layers,
            )

            self.graph_attn_bias = GraphAttnBias(
                num_heads=num_attention_heads,
                num_atoms=num_atoms,
                num_edges=num_edges,
                num_spatial=num_spatial,
                num_edge_dis=num_edge_dis,
                edge_type=edge_type,
                multi_hop_max_dist=multi_hop_max_dist,
                hidden_dim=embedding_dim,
                n_layers=num_encoder_layers,
            )

        self.embed_scale = embed_scale

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        if pre_layernorm:
            self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        (
            self.vit_model,
            self.vit_pooler,
            vit_other_layers,
            self.text_model,
            self.text_pooler,
            text_other_layers,
            self.node_classifier,
            self.text_dropout,
        ) = self.build_vit_bert_encoders(
            num_fusion_layers + 1, attention_dropout, activation_dropout
        )
        # self.bert = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
        self.fusion_layers = nn.ModuleList([])
        text_other_layers = [
            text_other_layers[i * num_fusion_stack : (i + 1) * num_fusion_stack]
            for i in range(
                (len(text_other_layers) + num_fusion_stack - 1)
                // num_fusion_stack
            )
        ]
        vit_other_layers = [
            vit_other_layers[i * num_fusion_stack : (i + 1) * num_fusion_stack]
            for i in range(
                (len(vit_other_layers) + num_fusion_stack - 1)
                // num_fusion_stack
            )
        ]
        self.fusion_layers.extend(
            [
                GraphFusionStack(
                    text_layer, vit_layer, num_bottle_neck, use_projection=True
                )
                for text_layer, vit_layer in zip(
                    text_other_layers, vit_other_layers
                )
            ]
        )

        if with_graph:
            self.layers.extend(
                [
                    self.build_graphormer_graph_encoder_layer(
                        num_layers=num_graph_stack,
                        embedding_dim=self.embedding_dim,
                        ffn_embedding_dim=ffn_embedding_dim,
                        num_attention_heads=num_attention_heads,
                        dropout=self.dropout_module.p,
                        attention_dropout=attention_dropout,
                        activation_dropout=activation_dropout,
                        activation_fn=activation_fn,
                        export=export,
                        q_noise=q_noise,
                        qn_block_size=qn_block_size,
                        pre_layernorm=pre_layernorm,
                    )
                    for _ in range(len(self.fusion_layers) + 1)
                ]
            )

        self.comment_encoder_stack = nn.ModuleList(
            [
                self.text_pooler,
                self.text_dropout,
                self.node_classifier,
            ]
        )

        print("NUMBER OF FUSION:", num_fusion_layers)
        print(f"NUMBER OF GRAPH LAYERS: {len(self.layers)} (t: {len(self.layers) * num_graph_stack})")
        print(f"NUMBER OF FUSION LAYERS: {len(self.fusion_layers)} (t: {len(self.fusion_layers) * num_fusion_stack})")

        self.num_bottle_neck = num_bottle_neck
        self.bottle_neck = nn.Embedding(num_bottle_neck, 768)

        # Apply initialization of model params after building the model
        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        def unfreeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = True

        if freeze_embeddings:
            raise NotImplementedError(
                "Freezing embeddings is not implemented yet."
            )

        if freeze_initial_encoders:
            freeze_module_params(self.text_model)
            freeze_module_params(self.vit_model)
            unfreeze_module_params(self.node_classifier)
            unfreeze_module_params(self.text_pooler)
            unfreeze_module_params(self.vit_pooler)

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

    def build_vit_bert_encoders(
        self, num_fusion_layers, attention_dropout, activation_dropout
    ):
        vit_model = AutoModel.from_pretrained(
            "google/vit-base-patch16-224",
            hidden_dropout_prob=activation_dropout,
            attention_probs_dropout_prob=attention_dropout,
        )
        bert = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            hidden_dropout_prob=activation_dropout,
            attention_probs_dropout_prob=attention_dropout,
        )
        bert_model = bert.bert
        vit_pooler = vit_model.pooler
        bert_pooler = bert_model.pooler
        if num_fusion_layers == 0:
            vit_other_layers = []
            bert_other_layers = []
        else:
            vit_other_layers = vit_model.encoder.layer[-num_fusion_layers:]
            vit_model.encoder.layer = vit_model.encoder.layer[
                :-num_fusion_layers
            ]
            bert_other_layers = bert_model.encoder.layer[-num_fusion_layers:]
            bert_model.encoder.layer = bert_model.encoder.layer[
                :-num_fusion_layers
            ]
        # note: this still includes the layernorm and pooler layers at the end, may want to remove
        vit_model = vit_model
        # bert_model = AutoModel.from_pretrained('microsoft/MiniLM-L12-H384-uncased')
        node_classifier = bert.classifier
        bert_dropout = bert.dropout
        bert_model = bert_model

        # this has a pooler at the end
        return (
            vit_model,
            vit_pooler,
            vit_other_layers,
            bert_model,
            bert_pooler,
            bert_other_layers,
            node_classifier,
            bert_dropout,
        )

    def build_graphormer_graph_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        export,
        q_noise,
        qn_block_size,
        pre_layernorm,
        num_layers=1,
    ):
        return GraphEncoderStack(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            pre_layernorm=pre_layernorm,
        )

    def forward(
        self,
        batched_data,
        last_state_only: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # print the shape of all tensors in batched_data
      
        mask = batched_data["y_mask"] # [#comments]
        x = batched_data["x"] 
        # x["token_type_ids"] is [#comments, 100]
        x_token_type_ids = x["token_type_ids"][mask, :] # [1, 100]
        x_attention_mask = x["attention_mask"][mask, :] # [1, 100]
        x_input_ids = x["input_ids"][mask, :] # [1, 100]
        bert_output = self.text_model(
            token_type_ids=x_token_type_ids,
            attention_mask=x_attention_mask,
            input_ids=x_input_ids,
        ).last_hidden_state # [1, 100, 768]
        
        n_graph = 1
        n_node = bert_output.size()[0]

        if self.enable_images and batched_data["x_images"] is not None:
            vit_output = self.vit_model(
                batched_data["x_images"]
            ).last_hidden_state # [num_images, 197, 768]
            # Take only the first image, otherwise the sizes are not matching
            # TODO(celia): double check this and try understanding what is wrong
            image = vit_output[0, :, :] 
            vit_output = image.unsqueeze(0) # [1, 197, 768]]
        else:
            vit_output = None
        
        bottle_neck = self.bottle_neck.weight.unsqueeze(0).repeat(n_node, 1, 1) # [1, 2, 764]

        added_mask = (
            torch.Tensor([1] * self.num_bottle_neck)
            .unsqueeze(0)
            .repeat(n_node, 1)
            .unsqueeze(1)
        ).to(self.device)

        #added_mask = (
        #    torch.Tensor([1] * self.num_bottle_neck)
        #    .unsqueeze(0)
        #    .unsqueeze(-1)
        #    .repeat(1, 1, 100)
        #).to(self.device)
        x_attention_mask = x_attention_mask.unsqueeze(1).repeat(1, 1, 1)
        
        x_attention_mask = torch.cat([added_mask, x_attention_mask], dim=2)
        extended_attention_mask = x_attention_mask
        extended_attention_mask = (1.0 - extended_attention_mask) #.torch.finfo( torch.half).min
  
        bert_output, vit_output, bottle_neck = self.fusion_layers[0](
            bert_output,
            vit_output,
            bottle_neck,
            bert_attention_mask=extended_attention_mask,
            x_image_indexes=batched_data["x_image_index"].long(),
        )
        if not self.with_graph:
            return bert_output, vit_output, bottle_neck

        bottle_neck_nodes = bottle_neck[:, 0, :]
        
        shape = x["token_type_ids"].unsqueeze(0).shape

        graph_data = (
            torch.zeros((shape[0], shape[1], 768))
            .to(bottle_neck_nodes.dtype)
        ).to(self.device) # [1, #comments, 768]
        graph_data[:, mask, :] = bottle_neck_nodes

        # compute padding mask. This is needed for multi-head attention
        x = graph_data

        n_graph, n_node = shape[:2]
        padding_mask = (x[:, :, 0]).eq(0)  # B x T x 1
        padding_mask_cls = torch.zeros(
            n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype
        )
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        mask = mask.unsqueeze(0)
        mask = torch.cat((padding_mask_cls, mask), dim=1)
        # B x (T+1) x 1

        x = self.graph_node_feature(
            x, batched_data["in_degree"], batched_data["out_degree"]
        ) # [1, #comments + 1, 768]

        # x: B x T x C

        attn_bias = self.graph_attn_bias(batched_data)

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for g_layer, f_layer in zip(self.layers, self.fusion_layers[1:]):
            x, _ = g_layer(
                x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
            )

            # extract bottle_neck tokens
            # T x B x C -> B x T x C
            x = x.transpose(0, 1)

            bottle_neck[:, 0, :] = x[mask, :]

            bert_output, vit_output, bottle_neck = f_layer(
                bert_output,
                vit_output,
                bottle_neck,
                extended_attention_mask,
                batched_data["x_image_indexes"],
            )

            x[mask, :] = bottle_neck[:, 0, :]
            # B x T x C -> T x B x C
            x = x.transpose(0, 1)
            if not last_state_only:
                inner_states.append(x)

        x, _ = self.layers[-1](
            x,
            self_attn_padding_mask=padding_mask,
            self_attn_mask=attn_mask,
            self_attn_bias=attn_bias,
        )

        if last_state_only:
            inner_states = [x]
        
        global_embedding = x[0, :, :]
        # bert_output [1, 100, 768]; bottle_neck [1, 2, 768]; global_embedding [1, 768]
                
        return bert_output, bottle_neck, global_embedding


