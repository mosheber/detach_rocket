import numpy as np
import pandas as pd
import torch_frame
from typing import Any

import torch
from torch import Tensor

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.encoder.stype_encoder import StypeEncoder, LinearEncoder
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
from torch import nn
from torch_frame.nn import Trompt
from torch.nn import LayerNorm, ModuleList, Parameter, ReLU, Sequential
from torch_frame.nn.encoder.stype_encoder import EmbeddingEncoder
from torch_frame.typing import NAStrategy
from torch_frame.nn.conv import TromptConv
from torch_frame.nn.decoder import TromptDecoder
from torch_frame.data.stats import compute_col_stats
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, EpochTimer, PrintLog
from skorch.dataset import get_len

from tqdm import tqdm

def col_stats_from_mat(X):
    col_stats = {str(i): compute_col_stats(pd.Series(X[:,i]), stype=torch_frame.numerical) for i in range(X.shape[1])}
    col_names_dict = {torch_frame.numerical:[str(x) for x in list(range(X.shape[1]))]}
    return col_stats, col_names_dict

class GumbelFeatureImportance(torch.nn.Module):
    def __init__(self, 
             channels: int,
             out_channels: int,
             **kwargs
        ):
        super().__init__()

        self.proj = torch.nn.Linear(channels, 1)
        self.gumbel_softmax_hard = kwargs.get("gumbel_softmax_hard", False)

    def forward(self, x):
        x_proj = self.proj(x).squeeze(-1)
        x_proj_soft = torch.nn.functional.gumbel_softmax(x_proj, tau=1.0, hard=self.gumbel_softmax_hard)
        self.x_proj_soft = torch.clone(x_proj_soft)
        return x_proj_soft[...,None] * x

POST_ENCODER_FEAT_IMP_MAP = {
    "gumbel": GumbelFeatureImportance
}

class FeatureImportTrompt(Trompt):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_prompts: int,
        num_layers: int,
        # kwargs for encoder
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dicts: list[dict[torch_frame.stype, StypeEncoder]],
        feature_importance_type="NONE",
        **kwargs,
    ) -> None:
        torch.nn.Module.__init__(self)
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")

        self.channels = channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        num_cols = sum(
            [len(col_names) for col_names in col_names_dict.values()])

        encoder_type_map = StypeWiseFeatureEncoder

        self.x_prompt = Parameter(torch.empty(num_prompts, channels))
        self.encoders = ModuleList()
        self.trompt_convs = ModuleList()
        for i in range(num_layers):
            if stype_encoder_dicts is None:
                stype_encoder_dict_layer = {
                    stype.categorical:
                    EmbeddingEncoder(
                        post_module=LayerNorm(channels),
                        na_strategy=NAStrategy.MOST_FREQUENT,
                    ),
                    stype.numerical:
                    LinearEncoder(
                        post_module=Sequential(
                            ReLU(),
                            LayerNorm(channels),
                        ),
                        na_strategy=NAStrategy.MEAN,
                    ),
                }
            else:
                stype_encoder_dict_layer = stype_encoder_dicts[i]

            self.encoders.append(
                encoder_type_map(
                    out_channels=channels,
                    col_stats=col_stats,
                    col_names_dict=col_names_dict,
                    stype_encoder_dict=stype_encoder_dict_layer,
                    **kwargs
                ))
            self.trompt_convs.append(
                TromptConv(channels, num_cols, num_prompts))
        # Decoder is shared across layers.
        self.trompt_decoder = TromptDecoder(channels, out_channels,
                                            num_prompts)

        self.post_enc_feat_imp_layer = None
        if feature_importance_type != "NONE":
            self.post_enc_feat_imp_layer = POST_ENCODER_FEAT_IMP_MAP[feature_importance_type](
                channels,
                out_channels,
                **kwargs
            )
        self.reset_parameters()

    def forward_stacked(self, tf: TensorFrame) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into a series of output
        predictions at each layer. Used during training to compute layer-wise
        loss.

        Args:
            tf (:class:`torch_frame.TensorFrame`):
                Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output predictions stacked across layers. The
                shape is :obj:`[batch_size, num_layers, out_channels]`.
        """
        batch_size = len(tf)
        outs = []
        # [batch_size, num_prompts, channels]
        x_prompt = self.x_prompt.repeat(batch_size, 1, 1)
        for i in range(self.num_layers):
            # [batch_size, num_cols, channels]
            x, _ = self.encoders[i](tf)

            ############ APPLY Feat Imp
            if self.post_enc_feat_imp_layer is not None:
                x = self.post_enc_feat_imp_layer(x)
            
            # [batch_size, num_prompts, channels]
            x_prompt = self.trompt_convs[i](x, x_prompt)
            # [batch_size, out_channels]
            out = self.trompt_decoder(x_prompt)
            # [batch_size, 1, out_channels]
            out = out.view(batch_size, 1, self.out_channels)
            outs.append(out)
        # [batch_size, num_layers, out_channels]
        stacked_out = torch.cat(outs, dim=1)
        return stacked_out

class TromptModel(FeatureImportTrompt):
    def __init__(self,
                 channels: int,
                 out_channels: int,
                 num_prompts: int,
                 num_layers: int,
                 X_sample: np.array,
                 feature_importance_type="NONE"):
        col_stats, col_names_dict = col_stats_from_mat(X_sample)
        super().__init__(
            channels=channels,
            out_channels=out_channels,
            num_prompts=num_prompts,
            num_layers=num_layers,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dicts=None,
            feature_importance_type=feature_importance_type
        )
        self.col_names_dict = col_names_dict
        self.dtype = next(self.parameters()).dtype

    def forward(self, X, labels=None):
        tf = torch_frame.TensorFrame(
            feat_dict = {
                torch_frame.numerical: X.to(self.dtype),
            },
            col_names_dict = self.col_names_dict
        )
        return super().forward(tf)

    def get_feature_importance(self):
        if hasattr(self, "post_enc_feat_imp_layer") and hasattr(self.post_enc_feat_imp_layer, "x_proj_soft"):
            return self.post_enc_feat_imp_layer.x_proj_soft.mean(dim=0).detach().cpu().numpy()
        return None

class BatchNeuralNetClassifier(NeuralNetClassifier):
    def run_single_epoch(self, iterator, training, prefix, step_fn, **fit_params):
        """Compute a single epoch of train or validation.

        Parameters
        ----------
        iterator : torch DataLoader or None
          The initialized ``DataLoader`` to loop over. If None, skip this step.

        training : bool
          Whether to set the module to train mode or not.

        prefix : str
          Prefix to use when saving to the history.

        step_fn : callable
          Function to call for each batch.

        **fit_params : dict
          Additional parameters passed to the ``step_fn``.

        """
        if iterator is None:
            return
        batch_count = 0
        for batch in tqdm(iterator):
            self.notify("on_batch_begin", batch=batch, training=training)
            step = step_fn(batch, **fit_params)
            self.history.record_batch(prefix + "_loss", step["loss"].item())
            batch_size = (get_len(batch[0]) if isinstance(batch, (tuple, list))
                          else get_len(batch))
            self.history.record_batch(prefix + "_batch_size", batch_size)
            self.notify("on_batch_end", batch=batch, training=training, **step)
            batch_count += 1
        self.history.record(prefix + "_batch_count", batch_count)

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        # we can assume that the attribute criterion_ exists; if users define
        # custom criteria, they have to override get_loss anyway
        self.criterion_ = self.criterion_.to(y_pred.dtype)
        return super().get_loss(y_pred, y_true, *args, **kwargs)

class FeatureImpNeuralNetClassifier(BatchNeuralNetClassifier):
    @property
    def feature_importances_(self):
        return self.module_.get_feature_importance()
    
    def _check_settable_attr(self, name, attr):
        return
    
    @property
    def _default_callbacks(self):
        return [
            ('epoch_timer', EpochTimer()),
            ('valid_f1', EpochScoring(
                'f1',
                name='valid_f1',
                lower_is_better=False,
            )),
            ('print_log', PrintLog()),
        ]
        # return [
        #     ('epoch_timer', EpochTimer()),
        #     ('train_loss', PassthroughScoring(
        #         name='train_loss',
        #         on_train=True,
        #     )),
        #     ('valid_loss', PassthroughScoring(
        #         name='valid_loss',
        #     )),
        #     ('valid_acc', EpochScoring(
        #         'accuracy',
        #         name='valid_acc',
        #         lower_is_better=False,
        #     )),
        #     ('print_log', PrintLog()),
        # ]