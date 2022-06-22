# -*- coding: utf-8 -*-
"""
@Time:Created on 2020/8/23 10:10
@author: Qichang Zhao
@Filename: model.py
@Software: PyCharm
"""
import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.init as init
from typing import Tuple
import torch.nn.functional as F
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

class View(nn.Module):
    def __init__(self, shape: tuple, contiguous: bool = False):
        super(View, self).__init__()
        self.shape = shape
        self.contiguous = contiguous

    def forward(self, x: Tensor) -> Tensor:
        if self.contiguous:
            x = x.contiguous()

        return x.view(*self.shape)

class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)

class FeedForwardModule(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 512,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)

class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 5,
            dropout_p: float = 0.1,
    ) -> None:
        super(ConvModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_channels=in_channels,out_channels=in_channels*2,
            kernel_size=1,stride=1,padding=0,bias=True,),
            Swish(),
            nn.Conv1d(in_channels=in_channels*2,out_channels=in_channels*2,kernel_size=kernel_size,
                stride=1,padding=(kernel_size - 1) // 2,bias=True,
            ),
            GLU(dim=1),
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=1, stride=1, padding=0, bias=True, ),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs).transpose(1, 2)

class RelativeMultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            mask: Tensor,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            # mask = mask.unsqueeze(1)
            # score.masked_fill_(mask, -1e9)
            mask = mask.unsqueeze(1).unsqueeze(2)
            score.masked_fill(mask == 0, -1e9)
        attn = F.softmax(score, -1)
        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score

class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1, max_len: int = 1000):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model,max_len=max_len)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size, seq_length, _ = inputs.size()
        pos_embedding = self.positional_encoding(seq_length)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)

        return self.dropout(outputs)

class CNNformerBlock(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 512,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            max_len: int = 1000
    ):
        super(CNNformerBlock, self).__init__()

        self.MHSA_model = MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                    max_len = max_len
                )
        self.CNN_model = ConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    dropout_p=conv_dropout_p,
                )
        self.FF_model = FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                )

    def forward(self, inputs: Tensor,mask: Tensor) -> Tuple[Tensor, Tensor]:
        MHSA_out = self.MHSA_model(inputs,mask) + inputs
        CNN_out = self.CNN_model(MHSA_out) + MHSA_out
        FFout = 0.5 * self.FF_model(CNN_out) + 0.5 * CNN_out
        return FFout

class CNNformerEncoder(nn.Module):
    def __init__(
            self,
            max_len: int=1000,
            encoder_dim: int = 512,
            num_layers: int = 17,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
    ):
        super(CNNformerEncoder, self).__init__()
        self.CNNformerlayers = nn.ModuleList([CNNformerBlock(
            encoder_dim=encoder_dim,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            max_len = max_len
        ) for _ in range(num_layers)])

        self.FFlayers = nn.ModuleList([FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        ) for _ in range(num_layers)])

    def forward(self, inputs: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        CNNformeroutputs = inputs
        for num in range(len(self.CNNformerlayers)):
            FF_output = 0.5 * self.FFlayers[num](CNNformeroutputs) + 0.5 * CNNformeroutputs
            CNNformeroutputs = self.CNNformerlayers[num](FF_output,mask)
        return CNNformeroutputs

class CNNFormerDTI(nn.Module):
    def __init__(self, drug_dict, protein_dict):
        super(CNNFormerDTI, self).__init__()
        self.protein_embed = nn.Embedding(protein_dict['embeding_num'], drug_dict['embeding_dim'], padding_idx=0)
        self.drug_embed = nn.Embedding(drug_dict['embeding_num'], protein_dict['embeding_dim'], padding_idx=0)
        self.protein_F = CNNformerEncoder(
            max_len = 1000,
            encoder_dim=256,
            num_layers=3,
            num_attention_heads=8,
            feed_forward_expansion_factor=4,
            feed_forward_dropout_p=0.1,
            attention_dropout_p=0.1,
            conv_dropout_p=0.1,
            conv_kernel_size=5)
        self.drug_F = CNNformerEncoder(
            max_len = 100,
            encoder_dim=256,
            num_layers=3,
            num_attention_heads=8,
            feed_forward_expansion_factor=4,
            feed_forward_dropout_p=0.1,
            attention_dropout_p=0.1,
            conv_dropout_p=0.1,
            conv_kernel_size=5)
        self.protein_attention_layer = nn.Linear(protein_dict["encoder_dim"], protein_dict["encoder_dim"])
        self.drug_attention_layer = nn.Linear(drug_dict["encoder_dim"], drug_dict["encoder_dim"] )
        self.protein_key_layer = nn.Linear(protein_dict["encoder_dim"], protein_dict["encoder_dim"])
        self.drug_key_layer = nn.Linear(drug_dict["encoder_dim"], drug_dict["encoder_dim"])
        self.dropout1 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(drug_dict["encoder_dim"] * 6, 1024)
        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout3 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, drug, protein, drug_mask, protein_mask):
        drug_embed = self.drug_embed(drug)
        protein_embed = self.protein_embed(protein)
        """Feature extractor"""
        drug_feature = self.drug_F(drug_embed, drug_mask)
        protein_feature = self.protein_F(protein_embed, protein_mask)

        """Attention block"""
        drug_attention_key = F.leaky_relu(self.drug_key_layer(drug_feature),0.01)
        protein_attention_key = F.leaky_relu(self.protein_key_layer(protein_feature),0.01)
        Attention_matrix = torch.tanh(torch.einsum('baf,bfc->bac', drug_attention_key, protein_attention_key.permute(0, 2, 1)))
        drug_attention = torch.tanh(torch.sum(Attention_matrix, 2))
        protein_attention = torch.tanh(torch.sum(Attention_matrix, 1))

        # Attention
        drug_feature_a = F.leaky_relu(self.drug_attention_layer(drug_feature), 0.01)
        protein_feature_a = F.leaky_relu(self.protein_attention_layer(protein_feature), 0.01)
        drug_feature_a = drug_feature_a * drug_attention.unsqueeze(2)
        protein_feature_a = protein_feature_a * protein_attention.unsqueeze(2)

        """"Predictor"""
        d_max_len = drug_feature.shape[1]
        d_max_feature_a = F.max_pool1d(drug_feature_a.permute(0, 2, 1), kernel_size=d_max_len, stride=1,
                                     padding=0, dilation=1, ceil_mode=False,
                                     return_indices=False).squeeze(2)
        d_avg_feature_a = F.avg_pool1d(drug_feature_a.permute(0, 2, 1), kernel_size=d_max_len, stride=1,
                                     padding=0, ceil_mode=False).squeeze(2)
        drug_feature_a = torch.cat([d_max_feature_a, d_avg_feature_a], dim=1)

        d_max_feature = F.max_pool1d(drug_feature.permute(0, 2, 1), kernel_size=d_max_len, stride=1,
                                       padding=0, dilation=1, ceil_mode=False,
                                       return_indices=False).squeeze(2)
        d_avg_feature = F.avg_pool1d(drug_feature.permute(0, 2, 1), kernel_size=d_max_len, stride=1,
                                       padding=0, ceil_mode=False).squeeze(2)
        drug_feature = torch.cat([d_max_feature, d_avg_feature], dim=1)


        p_max_len = protein_feature.shape[1]
        p_max_feature_a = F.max_pool1d(protein_feature_a.permute(0, 2, 1), kernel_size=p_max_len, stride=1,
                                     padding=0, dilation=1, ceil_mode=False,
                                     return_indices=False).squeeze(2)
        p_avg_feature_a = F.avg_pool1d(protein_feature_a.permute(0, 2, 1), kernel_size=p_max_len, stride=1,
                                     padding=0, ceil_mode=False).squeeze(2)
        protein_feature_a = torch.cat([p_max_feature_a, p_avg_feature_a], dim=1)
        p_max_feature = F.max_pool1d(protein_feature.permute(0, 2, 1), kernel_size=p_max_len, stride=1,
                                       padding=0, dilation=1, ceil_mode=False,
                                       return_indices=False).squeeze(2)
        p_avg_feature = F.avg_pool1d(protein_feature.permute(0, 2, 1), kernel_size=p_max_len, stride=1,
                                       padding=0, ceil_mode=False).squeeze(2)
        protein_feature = torch.cat([p_max_feature, p_avg_feature], dim=1)

        iner_f = torch.mul(drug_feature_a, protein_feature_a)
        pair = torch.cat([drug_feature, iner_f, protein_feature], dim=1)
        pair = self.dropout1(pair)
        fully1 = F.leaky_relu(self.fc1(pair),0.01)
        fully1 = self.dropout2(fully1)
        fully2 = F.leaky_relu(self.fc2(fully1),0.01)
        fully2 = self.dropout3(fully2)
        fully3 = F.leaky_relu(self.fc3(fully2),0.01)
        predict = self.out(fully3)
        return predict

if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    drug_dict = {'max_len': 100,
                 'encoder_dim': 256,
                 'embeding_dim': 256,
                 'embeding_num': 65,
                 'num_layers': 3,
                 'conv_kernel_size': 5,
                 'feed_forward_expansion_factor': 4,
                 'num_attention_heads':8,
                 'attention_dropout_p': 0.1,
                 'conv_dropout_p': 0.1
                 }
    protein_dict = {'max_len': 1000,
                 'encoder_dim': 256,
                 'embeding_dim': 256,
                 'embeding_num': 26,
                 'num_layers': 3,
                 'conv_kernel_size': 5,
                 'feed_forward_expansion_factor': 4,
                 'num_attention_heads':8,
                 'attention_dropout_p': 0.1,
                 'conv_dropout_p': 0.1}

    drug_input = torch.ones([16, 100], dtype=torch.long).cuda()
    drug_mask = torch.ones([16, 100], dtype=torch.long).cuda()

    protein_input = torch.ones([16, 1000], dtype=torch.long).cuda()
    protein_mask = torch.ones([16, 1000], dtype=torch.long).cuda()
    model = CNNFormerDTI(drug_dict,protein_dict).cuda()
    total = sum([param.nelement() for param in model.parameters()])  # 计算总参数量
    print("Number of parameter: %.3f M" % (total/1000000))  # 输出

    out = model(drug_input, protein_input, drug_mask, protein_mask)
    print(out.shape)