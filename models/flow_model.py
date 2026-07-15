import torch
import torch.nn as nn
from torch import Tensor

from models.modules import DenseNetwork, TimestepEmbedder
from models.attention import (
    Attention,
)
from models.norms import RMSNorm


from torch.nn.attention.flex_attention import create_block_mask


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class GLU(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        activation: str = "SiLU",
        dropout: float = 0.0,
        bias: bool = True,
        gated: bool = False,
    ):
        """Dense update with gated linear unit.

        See [2002.05202](https://arxiv.org/abs/2002.05202).

        Parameters
        ----------
        embed_dim : int
            Dimension of the input and output.
        hidden_dim : int | None, optional
            Dimension of the hidden layer. If None, defaults to embed_dim * 2.
        activation : str, optional
            Activation function.
        dropout : float, optional
            Dropout rate.
        bias : bool, optional
            Whether to include bias in the linear layers.
        gated : bool, optional
            Whether to gate the output of the hidden layer.
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim * 2

        self.gated = gated
        self.embed_dim = embed_dim
        self.in_proj = nn.Linear(embed_dim, hidden_dim + hidden_dim * gated, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        if self.gated:
            x1, x2 = x.chunk(2, dim=-1)
            x = self.activation(x1) * x2
        else:
            x = self.activation(x)
        x = self.drop(x)
        return self.out_proj(x)


class CALayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mha_config: dict = None,
        mlp_ratio: float = 4.0,
        do_selfattn: bool = False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.norm_k = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = GLU(hidden_dim, mlp_hidden_dim, bias=False, gated=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(3 * hidden_dim, hidden_dim * 6, bias=True)
        )
        if mha_config is None:
            mha_config = {}

        self.attn = Attention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            do_selfattn=do_selfattn,
            **mha_config,
        )

    def forward(self, seq_q, seq_k, ctxt, mask_k=None):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(ctxt).chunk(6, dim=-1)

        seq_q = seq_q + gate_msa.unsqueeze(1) * self.attn(
            x=modulate(self.norm1(seq_q), shift_msa, scale_msa),
            kv=self.norm_k(seq_k),
            kv_mask=mask_k,
        )

        seq_q = seq_q + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(seq_q), shift_mlp, scale_mlp)
        )
        return seq_q

    def reset_parameters(self):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)


class CABlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        mha_config=None,
        mlp_ratio=4.0,
        is_last=False,
    ):
        super().__init__()
        self.is_last = is_last
        if not is_last:
            self.layer_a = CALayer(
                hidden_dim,
                num_heads,
                mha_config,
                mlp_ratio,
            )
        self.layer_b = CALayer(
            hidden_dim,
            num_heads,
            mha_config,
            mlp_ratio,
        )

    def forward(self, seq_a, seq_b, ctxt, mask_a=None, mask_b=None):
        seq_b = self.layer_b(seq_q=seq_b, seq_k=seq_a, ctxt=ctxt, mask_k=mask_a)
        if not self.is_last:
            seq_a = self.layer_a(seq_q=seq_a, seq_k=seq_b, ctxt=ctxt, mask_k=mask_b)

        return seq_a, seq_b

    def reset_parameters(self):
        if not self.is_last:
            self.layer_a.reset_parameters()
        self.layer_b.reset_parameters()


class FlowNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = config["hidden_dim"]
        num_heads = config["num_heads"]
        num_dit_layers = config["num_dit_layers"]

        self.hidden_dim = hidden_dim
        self.act = config["act"]

        self.use_flex = (
            self.config["mha_config"].get("attn_type", "torch-meff") == "torch-flex"
        )

        self.truth_variables = self.config["truth_variables"]
        self.pflow_variables = self.config["pflow_variables"]

        self.truth_in_dim = len(self.truth_variables) + 5
        self.fs_in_dim = len(self.pflow_variables) + 5
        self.global_in_dim = config.get("global_dim", 6)
        self.fs_out_dim = self.fs_in_dim

        self.time_embedding = TimestepEmbedder(
            hidden_dim, time_factor=float(config.get("time_factor", 1.0))
        )
        self.truth_init = DenseNetwork(
            inpt_dim=self.truth_in_dim,  # + (hidden_dim // 4 if self.use_pos_embd else 0),
            outp_dim=hidden_dim,
            hddn_dim=[hidden_dim, hidden_dim],
            act_h=self.act,
        )
        self.fs_init = DenseNetwork(
            inpt_dim=self.fs_in_dim,
            outp_dim=hidden_dim,
            hddn_dim=[hidden_dim, hidden_dim],
            act_h=self.act,
        )
        self.global_embedding = DenseNetwork(
            inpt_dim=self.global_in_dim,
            outp_dim=hidden_dim,
            hddn_dim=[hidden_dim, hidden_dim],
            act_h=self.act,
        )
        self.ca_blocks = nn.ModuleList(
            [
                CABlock(
                    hidden_dim,
                    num_heads,
                    mha_config=self.config.get("mha_config", None),
                    is_last=i == num_dit_layers - 1,
                )
                for i in range(num_dit_layers)
            ]
        )
        self.final_layer = DenseNetwork(
            inpt_dim=hidden_dim,
            outp_dim=self.fs_out_dim,
            hddn_dim=[hidden_dim, hidden_dim, hidden_dim],
            act_h=self.act,
            ctxt_dim=3 * hidden_dim,
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.ca_blocks:
            block.reset_parameters()

    def create_padding_mask(self, pads_kv):
        def padding(b, h, q_idx, kv_idx):
            return ~pads_kv[b, kv_idx]

        return padding

    def forward(
        self,
        fs_data,
        truth_data,
        mask,
        timestep,
        global_data=None,
    ):
        # truth_mask = mask[..., 0]
        # fs_mask = mask[..., 1]
        truth_mask = mask[..., 0].bool()
        fs_mask = mask[..., 1].bool()

        fs_data_ = fs_data
        truth_data_ = truth_data

        truth_embd = self.truth_init(truth_data_)

        truth_ctxt = torch.sum(
            truth_embd * truth_mask.unsqueeze(-1), dim=1
        ) / torch.sum(truth_mask, dim=1, keepdim=True)
        time_embd = self.time_embedding(timestep)
        ctxt = torch.cat(
            [
                truth_ctxt,
                self.global_embedding(global_data),
                time_embd,
            ],
            -1,
        )

        fs_embd = self.fs_init(fs_data_)

        fs_embd = fs_embd * fs_mask.unsqueeze(-1)
        truth_embd = truth_embd * truth_mask.unsqueeze(-1)

        if self.use_flex:
            B, S = truth_mask.shape
            attn_mask_tr = create_block_mask(
                self.create_padding_mask(~truth_mask),
                B=B,
                H=None,
                Q_LEN=S,
                KV_LEN=S,
                _compile=True,
            )
            attn_mask_fs = create_block_mask(
                self.create_padding_mask(~fs_mask),
                B=B,
                H=None,
                Q_LEN=S,
                KV_LEN=S,
                _compile=True,
            )
        else:
            attn_mask_tr = ~truth_mask
            attn_mask_fs = ~fs_mask

        for block in self.ca_blocks:
            truth_embd, fs_embd = block(
                seq_a=truth_embd,
                seq_b=fs_embd,
                ctxt=ctxt,
                mask_a=attn_mask_tr,  # we need to invert the mask to satisfy the Attention convention
                mask_b=attn_mask_fs,
            )
        fs_out = self.final_layer(fs_embd, ctxt)

        return fs_out
