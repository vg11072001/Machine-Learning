from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


def similiarity(x1, x2):
    # refer: https://www.kaggle.com/code/hengck23/mvccl-model-for-admani-dataset

    p12 = (x1 * x2).sum(-1)
    p1 = torch.sqrt((x1 * x1).sum(-1))
    p2 = torch.sqrt((x2 * x2).sum(-1))
    s = p12 / (p1 * p2 + 1e-6)
    return s


def criterion_global_consistency(x1, x1_projection, x2, x2_projection, alpha=-0.5):
    # refer: https://www.kaggle.com/code/hengck23/mvccl-model-for-admani-dataset

    loss = alpha * (similiarity(x1, x1_projection) + similiarity(x2, x2_projection))
    return loss


class BFWithLogitsLoss(nn.Module):
    """Binary Focal Loss"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = 'mean') -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(input, target, self.alpha, self.gamma, self.reduction)


class ConsistencyLoss(nn.Module):

    def __init__(self, alpha: float = -0.5, reduction: str = 'mean'):
        super().__init__()
        self._alpha = alpha
        self._reduction = reduction

    def forward(self,
                x1: torch.Tensor,
                x1_projection: torch.Tensor,
                x2: torch.Tensor,
                x2_projection: torch.Tensor) -> torch.Tensor:

        loss = criterion_global_consistency(x1, x1_projection, x2, x2_projection, self._alpha)
        if self._reduction == 'mean':
            return torch.mean(loss)
        elif self._reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class FocalLoss(nn.Module):
    # reference: https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (torch.Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class HierarchicalCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight: Optional[list[float]] = None,
        distance_matrix: list[float] = [
            [1.0, 1.0, 2.0],  # 正常・軽度
            [1.0, 1.0, 1.0],  # 中度
            [2.0, 1.0, 1.0],  # 重度
        ],
    ):
        super().__init__()
        if weight is not None:
            self.register_buffer('class_weights', torch.Tensor(weight))
        else:
            self.class_weights = None
        self.register_buffer('distance_matrix', torch.Tensor(distance_matrix))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # クロスエントロピー損失の計算（負の対数尤度）
        log_preds = torch.log_softmax(logits, dim=1)
        ce_loss = -(targets * log_preds)
        if self.class_weights is not None:
            ce_loss = ce_loss * self.class_weights
        ce_loss = ce_loss.sum(dim=1)

        # 各サンプルの予測クラスを取得
        predicted_class = torch.argmax(logits, dim=1)

        # 各サンプルの正解クラスのソフトラベルに対応する距離を計算
        distances = torch.sum(targets * self.distance_matrix[predicted_class], dim=1)

        # 距離に基づいたペナルティを損失に加える
        loss = ce_loss * distances

        return loss.mean()


class RSNA2024Loss(nn.Module):
    def __init__(
        self,
        conditions: list[str] = ['spinal_canal_stenosis', 'left_neural_foraminal_narrowing', 'right_neural_foraminal_narrowing',
                                 'left_subarticular_stenosis', 'right_subarticular_stenosis'],
        levels: list[str] = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1'],
        ce_loss: dict = dict(name='CrossEntropyLoss', weight=[1.0, 2.0, 4.0]),
        slice_ce_loss: dict = dict(name='CrossEntropyLoss', weight=[1.0, 1.0, 1.0]),
        condition_weight: Optional[list[float]] = None,
        overall_loss_weight: float = 1.0,
        level_loss_weight: float = 0.0,
        slice_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.conditions = conditions
        self.levels = levels
        self.overall_loss_weight = overall_loss_weight
        self.level_loss_weight = level_loss_weight
        self.slice_loss_weight = slice_loss_weight
        self.level_loss = nn.CrossEntropyLoss()

        ce_loss_work = ce_loss.copy()
        ce_loss_name = ce_loss_work.pop('name')
        if ce_loss_name == 'CrossEntropyLoss':
            if 'weight' in ce_loss_work:
                weight = ce_loss_work.pop('weight')
                ce_loss_work['weight'] = torch.tensor(weight)
            self.ce_loss = nn.CrossEntropyLoss(**ce_loss_work)
        elif ce_loss_name == 'HierarchicalCrossEntropyLoss':
            self.ce_loss = HierarchicalCrossEntropyLoss(**ce_loss_work)
        else:
            raise ValueError(f'{ce_loss_name} is not supported.')

        slice_ce_loss_work = slice_ce_loss.copy()
        slice_ce_loss_name = slice_ce_loss_work.pop('name')
        if slice_ce_loss_name == 'CrossEntropyLoss':
            if 'weight' in slice_ce_loss_work:
                weight = slice_ce_loss_work.pop('weight')
                slice_ce_loss_work['weight'] = torch.tensor(weight)
            self.slice_ce_loss = nn.CrossEntropyLoss(**slice_ce_loss_work, reduction='none')
        else:
            raise ValueError(f'{slice_ce_loss_name} is not supported.')

        if condition_weight is None:
            condition_weight = [1.0] * len(conditions)
        self.register_buffer('condition_weight', torch.tensor(condition_weight))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        slice_logits: Optional[torch.Tensor] = None,
        slice_labels: Optional[torch.Tensor] = None,
        slice_weights: Optional[torch.Tensor] = None,
        level_logits: Optional[torch.Tensor] = None,
        level_targets: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        losses = dict()

        partial_losses = []
        for cond_idx, cond in enumerate(self.conditions):
            for level_idx, level in enumerate(self.levels):
                logit = logits[:, cond_idx, level_idx]
                target = targets[:, cond_idx, level_idx]
                if torch.any(target != -100):
                    partial_loss = self.ce_loss(logit, target) * self.condition_weight[cond_idx]
                    partial_losses.append(partial_loss)

        overall_loss = torch.mean(torch.stack(partial_losses)) * self.overall_loss_weight
        losses['overall_loss'] = overall_loss.item()

        # level loss
        if self.level_loss_weight > 0.0:
            level_losses = []
            for level_idx, level in enumerate(self.levels):
                level_logit = level_logits[:, level_idx]
                level_target = level_targets[:, level_idx]
                if torch.any(level_target != -100):
                    level_loss = self.level_loss(level_logit, level_target)
                    level_losses.append(level_loss)

            level_loss = torch.mean(torch.stack(level_losses)) * self.level_loss_weight
            losses['level_loss'] = level_loss.item()
        else:
            level_loss = 0.0
            losses['level_loss'] = 0.0

        # slice loss
        if slice_logits is not None and self.slice_loss_weight > 0.0 and slice_weights.sum() > 0:
            bs, num_slices, num_condition, num_classes = slice_logits.shape
            slice_loss = self.slice_ce_loss(slice_logits.view(-1, num_classes), slice_labels.view(-1, num_classes))
            slice_loss = slice_loss.view(bs, num_slices, num_condition)

            # 座標教師ラベルが存在しないスライス・疾患をマスクする
            slice_loss = slice_loss * slice_weights

            # 疾患ごとの重みを適用する
            slice_loss = slice_loss * self.condition_weight.view(1, 1, -1)

            # 教師ラベルが与えられた数で平均を取る
            slice_loss = slice_loss.sum() / slice_weights.sum()
            slice_loss = slice_loss * self.slice_loss_weight
            losses['slice_loss'] = slice_loss.item()
        else:
            slice_loss = 0.0
            losses['slice_loss'] = 0.0

        # weighted sum
        losses['loss'] = overall_loss + level_loss + slice_loss

        return losses


if __name__ == '__main__':
    import torch.nn.functional as F  # noqa
    rsna2024_loss = RSNA2024Loss(ce_loss={"name": "CrossEntropyLoss", "weight": [1.0, 2.0, 4.0]},
                                 conditions=['spinal_canal_stenosis', 'left_neural_foraminal_narrowing',
                                             'right_neural_foraminal_narrowing', 'left_subarticular_stenosis', 'right_subarticular_stenosis'],
                                 condition_weight=[1.0, 0.25, 0.25, 0.25, 0.25],
                                 levels=['any'],
                                 level_loss_weight=0.5,
                                 slice_loss_weight=0.5)
    rsna2024_loss(torch.randn(2, 5, 1, 3),
                  F.one_hot(torch.randint(0, 3, (2, 5, 1)), num_classes=3).float(),
                  torch.randn(2, 40, 5, 3),
                  F.one_hot(torch.randint(0, 3, (2, 40, 5)), num_classes=3).float(),
                  torch.randint(0, 2, (2, 40, 5)).float(),
                  torch.randn(2, 1, 5),
                  torch.randint(0, 5, (2, 1)))

    h_ce_loss = HierarchicalCrossEntropyLoss()

    logits_list = [
        torch.Tensor([[-1, -1, 1], [-1, -1, 1]]),
        torch.Tensor([[-1, 1, -1], [-1, 1, -1]]),
        torch.Tensor([[1, -1, -1], [1, -1, -1]]),
    ]

    targets = torch.Tensor([[0., 0., 1.], [0., 0., 1.]])

    for i, logits in enumerate(logits_list):
        loss = h_ce_loss(logits, targets)
        print(f"Loss {i+1}: {loss.item()}")
