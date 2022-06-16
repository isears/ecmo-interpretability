# From https://github.com/gzerveas/mvts_transformer, with some modifications
import torch
import math
from torch.nn.modules import (
    MultiheadAttention,
    Linear,
    Dropout,
    BatchNorm1d,
    TransformerEncoderLayer,
)
import pandas as pd
from tabsep.modeling import EarlyStopping

# Could also be learnable:
# https://github.com/gzerveas/mvts_transformer/blob/fe3b539ccc2162f55cf7196c8edc7b46b41e7267/src/models/ts_transformer.py#L105
class FixedPositionalEncoding(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer(
            "pe", pe
        )  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TSTransformerEncoderClassiregressor(torch.nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(
        self,
        feat_dim,
        max_len,
        d_model,
        n_heads,
        num_layers,
        dim_feedforward,
        num_classes,
        dropout=0.1,
        activation="gelu",
        freeze=False,
    ):
        super(TSTransformerEncoderClassiregressor, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = torch.nn.Linear(feat_dim, d_model)
        self.pos_enc = FixedPositionalEncoding(
            d_model, dropout=dropout * (1.0 - freeze), max_len=max_len
        )

        encoder_layer = TransformerEncoderLayer(
            d_model,
            self.n_heads,
            dim_feedforward,
            dropout * (1.0 - freeze),
            activation=activation,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers
        )

        self.act = torch.nn.functional.gelu  # Or relu

        self.dropout1 = torch.nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, max_len, num_classes)
        self.single_class_output = torch.nn.Sigmoid()

    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = torch.nn.Linear(d_model * max_len, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model
        )  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(
            inp, src_key_padding_mask=~padding_masks
        )  # (seq_length, batch_size, d_model)
        output = self.act(
            output
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # Output
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(
            output.shape[0], -1
        )  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)

        # Attempt to shoehorn the model into single-class classification
        output = self.single_class_output(output)

        return output


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup=0
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, warmup=warmup
        )
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))
                exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))

                denom = exp_avg_sq.sqrt().add_(group["eps"])
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if group["warmup"] > state["step"]:
                    scheduled_lr = 1e-8 + state["step"] * group["lr"] / group["warmup"]
                else:
                    scheduled_lr = group["lr"]

                step_size = (
                    scheduled_lr * math.sqrt(bias_correction2) / bias_correction1
                )

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(-group["weight_decay"] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(exp_avg, denom, value=(-step_size))

                p.data.copy_(p_data_fp32)

        return loss


class TstOneInput(TSTransformerEncoderClassiregressor):
    def forward(self, X):
        X_unpacked, padding_masks = (
            X[:, :, 0:-1],
            X[:, :, -1] == 1,
        )

        return super().forward(X_unpacked, padding_masks)

    # def __call__(self, X):
    #     return self.forward(X)


def model_factory(max_seq_len):
    # Other options (for now defaults from paper)
    d_model = 64  # Should be 64, but gpu mem can't handle it
    n_heads = 8
    num_layers = 3
    dim_feedforward = 256
    num_classes = 1

    transformer_model = TSTransformerEncoderClassiregressor(
        feat_dim=621,  # TODO: generate this dynamically?
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        max_len=max_seq_len,
        n_heads=n_heads,
        num_classes=num_classes,
        num_layers=num_layers,
    )

    optimizer = AdamW(transformer_model.parameters())

    return transformer_model, optimizer


if __name__ == "__main__":
    pass
