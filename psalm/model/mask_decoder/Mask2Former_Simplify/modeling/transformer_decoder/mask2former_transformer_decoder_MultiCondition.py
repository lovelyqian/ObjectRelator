# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .position_encoding import PositionEmbeddingSine


import torch
import torch.nn as nn
import torch.nn.functional as F


# default setting: exp0, cosine dis, without L2 norm before loss{}
def calculate_region_embedding_dis(embedding_list1, embedding_list2, sim_type="cos", L2_norm=False):
    # Ensure the lists have the same length
    assert len(embedding_list1) == len(embedding_list2), "Embedding lists must have the same length."
    for i in range(len(embedding_list1)):
        assert embedding_list1[i].shape == embedding_list2[i].shape
    
    # Initialize a list to store similarity or distance scores for each pair
    similarity_scores = []

    for emb1, emb2 in zip(embedding_list1, embedding_list2):
        # 如果 L2_norm 为 True，则对 emb1 和 emb2 进行归一化
        if L2_norm:
            emb1 = F.normalize(emb1, p=2, dim=-1)
            emb2 = F.normalize(emb2, p=2, dim=-1)

        # 根据相似度类型进行计算
        if sim_type == "cos":
            # 计算余弦相似度并取平均值
            sim = F.cosine_similarity(emb1, emb2, dim=-1)
            similarity_scores.append(sim.mean())
        
        elif sim_type == "ecu":
            # 计算欧氏距离并取平均值
            dist = torch.norm(emb1 - emb2, p=2, dim=-1)
            similarity_scores.append(dist.mean())
    
    # 根据 sim_type 获取最终的平均相似度或距离
    if sim_type == "cos":
        avg_similarity = torch.mean(torch.stack(similarity_scores))
        loss = 1 - avg_similarity  # 余弦相似度损失，越相似损失越小
    elif sim_type == "ecu":
        avg_distance = torch.mean(torch.stack(similarity_scores))
        loss = avg_distance  # 欧氏距离损失，越接近损失越小

    return loss



class FuseMultiCondition(nn.Module):
    def __init__(self, dim, fuse_method="add", learnable_weight=False):
        super(FuseMultiCondition, self).__init__()
        
        # 可学习权重参数初始化
        if learnable_weight:
            self.weight = nn.Parameter(torch.tensor(0.5))  # 设置初始权重
        else:
            self.weight = None
        
        # 融合方法选择
        self.fuse_method = fuse_method
        
        # Cross-Attention 的 Query、Key、Value 层
        print('CURRENT fuse_method for multicondition is:', )
        if fuse_method == "cross_attention":
            self.query_layer = nn.Linear(dim, dim)
            self.key_layer = nn.Linear(dim, dim)
            self.value_layer = nn.Linear(dim, dim)

    def fuse_multicondition(self, SEG_embedding, region_embedding_list):
        if SEG_embedding == None:
            return region_embedding_list
        new_region_embedding_list = []
        
        for seg_embed, region_embedding in zip(SEG_embedding, region_embedding_list):
            #print('seg_embed:', seg_embed.shape, 'region_embedding:', region_embedding.shape)
            if self.fuse_method == "add":
                if self.weight is not None:  # 如果有可学习权重
                    fused_region = self.weight * seg_embed + (1 - self.weight) * region_embedding
                else:
                    #print("SEG_embedding:", SEG_embedding)
                    #print("region_embeddnig:", region_embedding)
                    #fused_region = seg_embed + region_embedding #debug
                    fused_region = 0.2*seg_embed + 0.8*region_embedding
            
            elif self.fuse_method == "concat":
                fused_region = torch.cat([seg_embed.expand(region_embedding.shape[0], -1), region_embedding], dim=-1)
            
            elif self.fuse_method == "cross_attention":
                # 使用 SEG_embedding 作为 Query，Region 嵌入作为 Key 和 Value
                query = self.query_layer(seg_embed)  # [1, dim]
                key = self.key_layer(region_embedding)  # [Num of regions, dim]
                value = self.value_layer(region_embedding)  # [Num of regions, dim]
                
                # 计算注意力权重并应用于 Value
                attention_scores = F.softmax(query @ key.transpose(-2, -1), dim=-1)  # Shape [1, 4]
                fused_region = (attention_scores.transpose(-2, -1) * value)  # Shape [4, 256]
            elif self.fuse_method == "cross_attention_withoutpara":
                attention_scores = F.softmax(seg_embed @ region_embedding.transpose(-2, -1), dim=-1)  # Shape [1, 4]
                fused_region = (attention_scores.transpose(-2, -1) * region_embedding)  # Shape [4, 256]
            
            # 将融合结果添加到新列表
            new_region_embedding_list.append(fused_region)
        #for new_re in new_region_embedding_list:
        #    print(new_re.shape)
        return new_region_embedding_list


class FuseMultiConditionNew(nn.Module):
    def __init__(self, dim, fuse_method="add", learnable_weight=False, num_heads=1, fixed_k=None, learnable_fuse_ratios=False):
        super(FuseMultiConditionNew, self).__init__()
        
        self.k = None

        # 设置可学习权重参数 k
        if learnable_weight:
            #self.k = nn.Parameter(torch.tensor(0.5))  # 设置初始权重
            self.k = nn.Parameter(torch.tensor(0.5, dtype=torch.float)) 
            #self.register_parameter("k", nn.Parameter(torch.tensor(0.5, dtype=torch.float)))
            # 使用 nn.Embedding 来定义一个单独的可学习参数
        if fixed_k:
            self.k = fixed_k

        if learnable_fuse_ratios:
            self.fuse_k1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float)) 
            self.fuse_k2 = nn.Parameter(torch.tensor(1.0, dtype=torch.float)) 

        
        # 融合方法选择
        self.fuse_method = fuse_method
        
        # Cross Attention 设置，使用 PyTorch 的 MultiheadAttention
        if fuse_method == "CA":
            self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        
        if "CAandAdd" in fuse_method:
            self.multihead_attn1 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
            self.multihead_attn2 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)

    def fuse_multicondition(self, SEG_embedding, region_embedding_list):
        if SEG_embedding is None:
            return region_embedding_list
        
        new_region_embedding_list = []
        
        for seg_embed, region_embedding in zip(SEG_embedding, region_embedding_list):
            if self.fuse_method == "add":
                fused_region = seg_embed + region_embedding
            
            elif self.fuse_method == "concat": # not used for now
                fused_region = torch.cat([seg_embed.expand(region_embedding.shape[0], -1), region_embedding], dim=-1)
            
            elif self.fuse_method == "CA":
                # 将 seg_embed 扩展为与 region_embedding 相同的形状
                #print('1:', region_embedding.shape)
                seg_embed_expanded = seg_embed.expand(region_embedding.shape[0], -1).unsqueeze(1)
                region_embedding = region_embedding.unsqueeze(1)  # [num_regions, 1, dim]
                
                # Cross Attention：使用 seg_embed_expanded 作为 Query
                fused_region, _ = self.multihead_attn(query=seg_embed_expanded, key=region_embedding, value=region_embedding)
                fused_region = fused_region.squeeze(1)  # 移除中间维度
                #print('2:', fused_region.shape)

            elif self.fuse_method == "CAandAdd":
                # 将 seg_embed 扩展为与 region_embedding 相同的形状
                #print('1:', region_embedding.shape)
                seg_embed_expanded = seg_embed.expand(region_embedding.shape[0], -1).unsqueeze(1)
                region_embedding = region_embedding.unsqueeze(1)  # [num_regions, 1, dim]
                # Cross Attention：使用 seg_embed_expanded 作为 Query
                fused_region, _ = self.multihead_attn1(query=seg_embed_expanded, key=region_embedding, value=region_embedding)
                fused_region = fused_region.squeeze(1)  # 移除中间维度

                
                # region_embedding： N，1, dim
                #seg_embed_expanded: N,1,dm
                # Cross Attention：使用 region_emb_expanded 作为 Query，seg_emb_expanded 作为 Key 和 Value
                fused_seg, _ = self.multihead_attn2(query=region_embedding, key=seg_embed_expanded, value=seg_embed_expanded)
                fused_seg = fused_seg.squeeze(1)  # 最终形状为 [N, dim]

                # add 
                fused_region = fused_seg + fused_region


            elif self.fuse_method == "CAandAdd_learnableWeight":
                # 将 seg_embed 扩展为与 region_embedding 相同的形状
                #print('1:', region_embedding.shape)
                seg_embed_expanded = seg_embed.expand(region_embedding.shape[0], -1).unsqueeze(1)
                region_embedding = region_embedding.unsqueeze(1)  # [num_regions, 1, dim]
                
                # Cross Attention：使用 seg_embed_expanded 作为 Query
                fused_region, _ = self.multihead_attn1(query=seg_embed_expanded, key=region_embedding, value=region_embedding)
                fused_region = fused_region.squeeze(1)  # 移除中间维度
                
                #
                fused_seg, _ = self.multihead_attn2(query=region_embedding, key=seg_embed_expanded, value=seg_embed_expanded)
                fused_seg = fused_seg.squeeze(1)  # 最终形状为 [N, dim]

                # add 
                fused_region = self.fuse_k1 * fused_seg + self.fuse_k2*fused_region


            elif self.fuse_method == "CA_withoutpara":
                attention_scores = F.softmax(seg_embed @ region_embedding.transpose(-2, -1), dim=-1)  # Shape [1, num_regions]
                fused_region = (attention_scores.transpose(-2, -1) * region_embedding)  # Shape [num_regions, dim]


            elif self.fuse_method == 'CANonParamterandAdd':
                attention_scores = F.softmax(seg_embed @ region_embedding.transpose(-2, -1), dim=-1)  # Shape [1, num_regions]
                fused_region = (attention_scores.transpose(-2, -1) * region_embedding)  # Shape [num_regions, dim]

                attention_scores = F.softmax(region_embedding @ seg_embed.transpose(-2, -1), dim=0)  # Shape [N, 1]
                fused_seg = (attention_scores * seg_embed).sum(dim=0, keepdim=True)  # Shape [1, dim]
                print('fused_seh:', fused_seg.shape, 'fused_region:', fused_region.shape)

                # add 
                fused_region = fused_seg + fused_region

            elif self.fuse_method == 'CANonParamterandAdd_learnableWeight':
                attention_scores = F.softmax(seg_embed @ region_embedding.transpose(-2, -1), dim=-1)  # Shape [1, num_regions]
                fused_region = (attention_scores.transpose(-2, -1) * region_embedding)  # Shape [num_regions, dim]

                attention_scores = F.softmax(region_embedding @ seg_embed.transpose(-2, -1), dim=0)  # Shape [N, 1]
                fused_seg = (attention_scores * seg_embed).sum(dim=0, keepdim=True)  # Shape [1, dim]
                print('fused_seh:', fused_seg.shape, 'fused_region:', fused_region.shape)
                print('fuse_k1:', self.fuse_k1, 'fuse_k2:', self.fuse_k2)
                # add 
                fused_region = self.fuse_k1 * fused_seg + self.fuse_k2*fused_region

            #print('self.k' ,self.k)
            #残差链接
            if self.k is not None:
                fused_region = self.k * region_embedding.squeeze(1) + (1-self.k) * fused_region

            new_region_embedding_list.append(fused_region)
        
        return new_region_embedding_list


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x





class MultiScaleMaskedTransformerDecoder(nn.Module):
    def __init__(
            self,
            in_channels,
            num_classes,
            mask_classification=True,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=10,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False
    ):
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, x, mask_features, mask=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                               attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                                   attn_mask_target_size=size_list[(
                                                                                                                           i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask.float(), size=attn_mask_target_size, mode="bilinear",
                                  align_corners=False).to(mask_embed.dtype)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                         1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


class MultiScaleMaskedTransformerDecoderForOPTPreTrainMultiCondition(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=10,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False,
            seg_norm=False,
            seg_concat=True,
            seg_proj=True,
            seg_fuse_score=False
    ):
        print('MultiScaleMaskedTransformerDecoderForOPTPreTrainMultiCondition 0: you are using the MultiCondition Predictor.')
        nn.Module.__init__(self)
        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        # newly added for multicondition fusion
        #self.fuse_model = FuseMultiCondition(dim=hidden_dim, fuse_method="add", learnable_weight=False)  #new baseline
        #self.fuse_model =  FuseMultiCondition(dim=hidden_dim, fuse_method="cross_attention_withoutpara", learnable_weight=False)  #new baseline

        # New:
        #self.fuse_model = FuseMultiConditionNew(dim=hidden_dim, fuse_method="add", learnable_weight=False)  # addbaseline,  no residual connection,  seg+region
        #self.fuse_model = FuseMultiConditionNew(dim=hidden_dim, fuse_method="CA_withoutpara", learnable_weight=False) #CA_withoutparameters baseline,  no residual connection, att*value
        #self.fuse_model = FuseMultiConditionNew(dim=hidden_dim, fuse_method="add", learnable_weight=True) # add method,  with learnable residual connection
        
        #OURS Here
        #self.fuse_model = FuseMultiConditionNew(dim=hidden_dim, fuse_method="CA", learnable_weight=True, num_heads=1) #ours: CA method,  with learnable residual connection, num_heads =1 
        
        #self.fuse_model = FuseMultiConditionNew(dim=hidden_dim, fuse_method="CA", learnable_weight=True, num_heads=4) #CA method,  with learnable residual connection, num_heads =4

        #NEW2:
        #self.fuse_model = FuseMultiConditionNew(dim=hidden_dim, fuse_method="CANonParamterandAdd", learnable_weight=False) 
        
        #self.fuse_model = FuseMultiConditionNew(dim=hidden_dim, fuse_method="CANonParamterandAdd_learnableWeight", learnable_weight=False, learnable_fuse_ratios=True) 
    

        # New3
        #self.fuse_model = FuseMultiConditionNew(dim=hidden_dim, fuse_method="CA_withoutpara", learnable_weight=False, fixed_k=0.2) #CA_withoutparameters baseline
       
        # CAbaselines without residucal connection 
        #self.fuse_model = FuseMultiConditionNew(dim=hidden_dim, fuse_method="CA", learnable_weight=False, num_heads=1) #CA, without redisual connection 

        # different fixed k, if 43.2 is the best one ,run this:
        #self.fuse_model = FuseMultiConditionNew(dim=hidden_dim, fuse_method="CA", learnable_weight=False, fixed_k=0.2, num_heads=1) #CA, without redisual connection 
        #self.fuse_model = FuseMultiConditionNew(dim=hidden_dim, fuse_method="CA", learnable_weight=False, fixed_k=0.5, num_heads=1) #CA, without redisual connection 
        #self.fuse_model = FuseMultiConditionNew(dim=hidden_dim, fuse_method="CA", learnable_weight=False, fixed_k=0.8, num_heads=1) #CA, without redisual connection 

        # different fixed k, if 43.2 is NOT the best one ,run this:
        #self.fuse_model = FuseMultiConditionNew(dim=hidden_dim, fuse_method="CAandAdd", learnable_weight=False,  num_heads=1) 
        #self.fuse_model = FuseMultiConditionNew(dim=hidden_dim, fuse_method="CAandAdd_learnableWeight", learnable_weight=False,  num_heads=1, learnable_fuse_ratios=True) 

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.seg_norm = seg_norm
        self.seg_concat = seg_concat
        self.seg_proj = seg_proj
        self.seg_fuse_score = seg_fuse_score
        if self.seg_norm:
            print('add seg norm for [SEG]')
            self.seg_proj_after_norm = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
            self.class_name_proj_after_norm = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
            self.SEG_norm = nn.LayerNorm(hidden_dim)
            self.class_name_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.SEG_query_embed = nn.Embedding(num_queries + 1, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.SEG_proj = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        self.CLASS_proj = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        self.REGION_proj = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        # self.class_embed = nn.Linear(hidden_dim, 1)
        # self.class_embed_sim = nn.Linear(hidden_dim, 81)

    def forward(self, x, mask_features, mask=None, seg_query=None, SEG_embedding=None, class_name_embedding=None, region_embedding_list=None):
        #print('MultiScaleMaskedTransformerDecoderForOPTPreTrainMultiCondition, 0, self.seg_concat:', self.seg_concat)
        if self.seg_concat:
            return self.forward_concat(x, mask_features, mask, seg_query, SEG_embedding, class_name_embedding, region_embedding_list)
        else:
            return self.forward_woconcat(x, mask_features, mask, seg_query, SEG_embedding, class_name_embedding, region_embedding_list)

    '''
    def forward_withSSLloss(self, x, mask_features, mask=None, seg_query=None, SEG_embedding=None, class_name_embedding=None, region_embedding_list=None, region_embedding_list_exo=None):
        #print('MultiScaleMaskedTransformerDecoderForOPTPreTrainMultiCondition, 0, self.seg_concat:', self.seg_concat)
        if self.seg_concat:
            return self.forward_concat(x, mask_features, mask, seg_query, SEG_embedding, class_name_embedding, region_embedding_list)
        else:
            return self.forward_woconcat_withSSLloss(x, mask_features, mask, seg_query, SEG_embedding, class_name_embedding, region_embedding_list, region_embedding_list_exo)


    def forward_woconcat_withSSLloss(self, x, mask_features, mask=None, seg_query=None, SEG_embedding=None,
                         class_name_embedding=None, region_embedding_list=None, region_embedding_list_exo=None):
        
        #('MultiScaleMaskedTransformerDecoderForOPTPreTrainMultiCondition, 1, SEG_embedding:', SEG_embedding.shape, 'region_embedding_list:', len(region_embedding_list), region_embedding_list[0].shape, region_embedding_list[1].shape, region_embedding_list[2].shape, region_embedding_list[3].shape)
        #print('MultiScaleMaskedTransformerDecoderForOPTPreTrainMultiCondition:', 'forward_woconcat_withSSLloss():', len(region_embedding_list), len(region_embedding_list_exo))
      
        
        # fuse multicondition {ego, seg}
        #region_embedding_list = self.fuse_model.fuse_multicondition(SEG_embedding, region_embedding_list)

        # fuse multicondition {exo, seg}
        #region_embedding_list_exo = self.fuse_model.fuse_multicondition(SEG_embedding, region_embedding_list_exo)

        # sue multi
        SEG_embedding = None     # not activaing the SEG path separtely 

        # calculate SSL loss
        loss_region_emb_SSL = calculate_region_embedding_dis(region_embedding_list, region_embedding_list_exo, sim_type="ecu")
        #loss_region_emb_SSL = calculate_region_embedding_dis(region_embedding_list, region_embedding_list_exo, sim_type="cos")   #xiugai
        #('loss_region_emb_SSL:', loss_region_emb_SSL)

        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2).to(x[i].dtype))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        if seg_query is None:
            output = self.new_query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        else:
            output = seg_query.permute(1, 0, 2)

        predictions_SEG_class = []
        predictions_class_name_class = []
        predictions_region_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        SEG_class, class_name_class, outputs_mask, attn_mask, region_class_list = self.forward_prediction_heads(output,
                                                                                                                mask_features,
                                                                                                                attn_mask_target_size=
                                                                                                                size_list[
                                                                                                                    0],
                                                                                                                SEG_embedding=SEG_embedding,
                                                                                                                class_name_embedding=class_name_embedding,
                                                                                                                region_embedding_list=region_embedding_list)

        predictions_SEG_class.append(SEG_class)
        predictions_class_name_class.append(class_name_class)
        predictions_mask.append(outputs_mask)
        predictions_region_class.append(region_class_list)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            SEG_class, class_name_class, outputs_mask, attn_mask, region_class_list = self.forward_prediction_heads(
                output, mask_features,
                attn_mask_target_size=
                size_list[(
                                  i + 1) % self.num_feature_levels],
                SEG_embedding=SEG_embedding,
                class_name_embedding=class_name_embedding,
                region_embedding_list=region_embedding_list
                )
            predictions_SEG_class.append(SEG_class)
            predictions_class_name_class.append(class_name_class)
            predictions_mask.append(outputs_mask)
            predictions_region_class.append(region_class_list)

        assert len(predictions_SEG_class) == self.num_layers + 1

        out = {
            'loss_region_emb_SSL': loss_region_emb_SSL, #xiugai
            'pred_SEG_logits': predictions_SEG_class[-1],
            'pred_class_name_logits': predictions_class_name_class[-1],
            'pred_region_logits': predictions_region_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_SEG_class, predictions_class_name_class, predictions_mask, predictions_region_class
            )
        }
        return out
    '''

    def forward_concat(self, x, mask_features, mask=None, seg_query=None, SEG_embedding=None,
                       class_name_embedding=None, region_embedding_list=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2).to(x[i].dtype))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.SEG_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        if seg_query is None:
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        else:
            output = seg_query.permute(1, 0, 2)

        predictions_SEG_class = []
        predictions_class_name_class = []
        predictions_region_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        SEG_class, class_name_class, outputs_mask, attn_mask, region_class_list = self.forward_prediction_heads(output, mask_features,
                                                                                             attn_mask_target_size=
                                                                                             size_list[0],
                                                                                             SEG_embedding=SEG_embedding,
                                                                                             class_name_embedding=class_name_embedding,
                                                                                             region_embedding_list = region_embedding_list)
        predictions_SEG_class.append(SEG_class)
        predictions_class_name_class.append(class_name_class)
        predictions_mask.append(outputs_mask)
        predictions_region_class.append(region_class_list)

        for i in range(self.num_layers):

            output = torch.cat([SEG_embedding.transpose(0, 1), output], 0)
            SEG_mask = torch.zeros((attn_mask.shape[0], 1, attn_mask.shape[-1]), dtype=torch.bool,
                                   device=attn_mask.device)
            attn_mask = torch.cat([SEG_mask, attn_mask], dim=1)

            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            output = output[1:]
            SEG_embedding = output[0].unsqueeze(0).transpose(0, 1)
            SEG_class, class_name_class, outputs_mask, attn_mask, region_class_list = self.forward_prediction_heads(
                output, mask_features,
                attn_mask_target_size=
                size_list[(
                                  i + 1) % self.num_feature_levels],
                SEG_embedding=SEG_embedding,
                class_name_embedding=class_name_embedding,
                region_embedding_list=region_embedding_list
                )
            predictions_SEG_class.append(SEG_class)
            predictions_class_name_class.append(class_name_class)
            predictions_mask.append(outputs_mask)
            predictions_region_class.append(region_class_list)

        assert len(predictions_SEG_class) == self.num_layers + 1

        out = {
            'pred_SEG_logits': predictions_SEG_class[-1],
            'pred_class_name_logits': predictions_class_name_class[-1],
            'pred_region_logits': predictions_region_class[-1] if predictions_region_class is not None else None,
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_SEG_class, predictions_class_name_class, predictions_mask, predictions_region_class
            )
        }
        return out

    def forward_woconcat(self, x, mask_features, mask=None, seg_query=None, SEG_embedding=None,
                         class_name_embedding=None, region_embedding_list=None):
        
        #('MultiScaleMaskedTransformerDecoderForOPTPreTrainMultiCondition, 1, SEG_embedding:', SEG_embedding.shape, 'region_embedding_list:', len(region_embedding_list), region_embedding_list[0].shape, region_embedding_list[1].shape, region_embedding_list[2].shape, region_embedding_list[3].shape)
        
        # fuse multicondition
        #print('111')
        # ours
        #region_embedding_list = self.fuse_model.fuse_multicondition(SEG_embedding, region_embedding_list)
        #print('222')
        SEG_embedding = None     # not activaing the SEG path separtely 

        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2).to(x[i].dtype))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        if seg_query is None:
            output = self.new_query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        else:
            output = seg_query.permute(1, 0, 2)

        predictions_SEG_class = []
        predictions_class_name_class = []
        predictions_region_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        SEG_class, class_name_class, outputs_mask, attn_mask, region_class_list = self.forward_prediction_heads(output,
                                                                                                                mask_features,
                                                                                                                attn_mask_target_size=
                                                                                                                size_list[
                                                                                                                    0],
                                                                                                                SEG_embedding=SEG_embedding,
                                                                                                                class_name_embedding=class_name_embedding,
                                                                                                                region_embedding_list=region_embedding_list)

        predictions_SEG_class.append(SEG_class)
        predictions_class_name_class.append(class_name_class)
        predictions_mask.append(outputs_mask)
        predictions_region_class.append(region_class_list)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            SEG_class, class_name_class, outputs_mask, attn_mask, region_class_list = self.forward_prediction_heads(
                output, mask_features,
                attn_mask_target_size=
                size_list[(
                                  i + 1) % self.num_feature_levels],
                SEG_embedding=SEG_embedding,
                class_name_embedding=class_name_embedding,
                region_embedding_list=region_embedding_list
                )
            predictions_SEG_class.append(SEG_class)
            predictions_class_name_class.append(class_name_class)
            predictions_mask.append(outputs_mask)
            predictions_region_class.append(region_class_list)

        assert len(predictions_SEG_class) == self.num_layers + 1

        out = {
            'pred_SEG_logits': predictions_SEG_class[-1],
            'pred_class_name_logits': predictions_class_name_class[-1],
            'pred_region_logits': predictions_region_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_SEG_class, predictions_class_name_class, predictions_mask, predictions_region_class
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, SEG_embedding=None,
                                 class_name_embedding=None, region_embedding_list=None):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        # SEG_embedding = self.SEG_norm(SEG_embedding).expand_as(decoder_output)
        # SEG_embedding = SEG_embedding.expand_as(decoder_output)
        if SEG_embedding is not None:
            if self.seg_proj:
                decoder_seg_output = self.SEG_proj(decoder_output)
            else:
                decoder_seg_output = decoder_output
            if self.seg_norm:
                SEG_embedding = self.SEG_norm(SEG_embedding)
                SEG_embedding = self.seg_proj_after_norm(SEG_embedding)
            SEG_class = torch.einsum('bld,bcd->blc', decoder_seg_output, SEG_embedding)
        else:
            SEG_class = None
        # SEG_class = F.cosine_similarity(decoder_seg_output, SEG_embedding, dim=-1, eps=1e-6).unsqueeze(-1)
        if class_name_embedding is not None:
            # decoder_class_output = decoder_output.detach()
            decoder_class_output = decoder_output
            if self.seg_proj:
                decoder_class_output = self.CLASS_proj(decoder_class_output)
            else:
                decoder_class_output = decoder_class_output
            if self.seg_norm:
                class_name_embedding = self.class_name_norm(class_name_embedding)
                class_name_embedding = self.class_name_proj_after_norm(class_name_embedding)
            dot_product = torch.einsum('bld,bcd->blc', decoder_class_output, class_name_embedding)
            # dot_product = self.class_embed_sim(decoder_class_output)
            # decoder_output_mag = torch.norm(decoder_output, dim=-1, keepdim=True)
            # class_name_embedding_mag = torch.norm(class_name_embedding, dim=-1, keepdim=True)
            # class_name_class = dot_product / (decoder_output_mag * class_name_embedding_mag.transpose(-1, -2) + 1e-8)
            if self.seg_fuse_score:
                class_SEG_class = SEG_class.expand_as(dot_product)
                reverse_bg_mask = torch.ones_like(class_SEG_class).to(dtype=class_SEG_class.dtype,device=class_SEG_class.device)
                reverse_bg_mask[:,:,-1] = -reverse_bg_mask[:,:,-1]
                class_name_class = dot_product * class_SEG_class * reverse_bg_mask
            else:
                class_name_class = dot_product
        else:
            class_name_class = None
        if region_embedding_list is not None:
            if self.seg_proj:
                decoder_region_output = self.REGION_proj(decoder_output)
            else:
                decoder_region_output = decoder_output
            region_class_list = []
            for sample_decoder_output, region_embedding in zip(decoder_region_output, region_embedding_list):
                sample_region_class = torch.einsum('kd,ld->kl', region_embedding, sample_decoder_output)
                region_class_list.append(sample_region_class)
        else:
            region_class_list = None

        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask.float(), size=attn_mask_target_size, mode="bilinear",
                                  align_corners=False).to(mask_embed.dtype)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                         1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return SEG_class, class_name_class, outputs_mask, attn_mask, region_class_list

    @torch.jit.unused
    def _set_aux_loss(self, outputs_SEG_class, outputs_class_name_class, outputs_seg_masks, predictions_region_class):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # if self.mask_classification:
        #     return [
        #         {"pred_logits": a, "pred_masks": b}
        #         for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        #     ]
        # else:
        #     return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

        return [
            {"pred_SEG_logits": a, "pred_class_name_logits": b, "pred_masks": c, "pred_region_logits": d}
            for a, b, c, d in zip(outputs_SEG_class[:-1], outputs_class_name_class[:-1], outputs_seg_masks[:-1],
                                  predictions_region_class[:-1])
        ]






