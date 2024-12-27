# Reference: https://github.com/allegro/allRank/blob/c88475661cb72db292d13283fdbc4f2ae6498ee4/allrank/models/losses/loss_utils.py
import torch
from itertools import zip_longest
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from loguru import logger

PADDED_Y_VALUE = -1
DEFAULT_EPS = 1e-10

def masked_max(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: Tensor of shape (N, L, hidden_dim)
        lengths: 1D tensor of shape (N,) with valid sequence lengths.

    Returns:
        max_vals: Tensor of shape (N, hidden_dim), where each row
                  is the max over the valid time steps for that sequence.
    """
    N, L, hidden_dim = x.shape
    device = x.device

    # Create a mask of shape (N, L) -> True where index < length, else False
    idxs = torch.arange(L, device=device).unsqueeze(0)  # shape (1, L)
    mask = idxs < lengths.unsqueeze(1)                 # shape (N, L)

    # Expand mask to (N, L, hidden_dim) so it can be applied to x
    mask_3d = mask.unsqueeze(-1).expand(-1, -1, hidden_dim)

    # Replace padding positions with -inf
    x_masked = x.masked_fill(~mask_3d, float('-inf'))

    # Take max along the length dimension (dim=1)
    max_vals, _ = x_masked.max(dim=1)
    return max_vals

def LN(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std

def convert_to_torch_tensor(array):
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    elif isinstance(array, torch.Tensor):
        return array
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")
    
class DictDataset(Dataset):
    def __init__(self, dict_x, dict_y, dict_z=None, dict_m=None):
        self.dict_x = dict_x
        self.dict_y = dict_y
        self.dict_z = dict_z
        self.dict_m = dict_m
        self.keys = sorted(self.dict_x.keys())
    
    def __getitem__(self, index):
        return_item = [self.dict_x[self.keys[index]], self.dict_y[self.keys[index]]]
        if self.dict_z is not None:
            return_item.append(self.dict_z[self.keys[index]])
        if self.dict_m is not None:
            return_item.append(self.dict_m[self.keys[index]])
        return return_item
    
    def __len__(self):
        return len(self.keys)

def assign_and_normalize_scores(line_label, candidate_token):
    """
    Assign and normalize scores to data points based on the following rules:
    1. For data points with line_label >= 1, their scores must be higher than those with line_label == 0.
    2. Data points with line_label == 0 must have the same score.
    3. For data points with line_label == 1, those with higher line_length get higher scores.
    4. Normalize all scores to [0, 1], with the highest score as 1.

    Args:
        line_label (list): A list of labels for each data point.
        candidate_token (list): A list of candidate tokens

    Returns:
        list: A list of normalized scores assigned to each data point.
    """
    line_length = [len(i) for i in candidate_token]
    # Initialize raw scores
    raw_scores = [0] * len(line_label)
    
    if len(line_label) == 1:
        return [1]
    # Determine the base scores for each label category
    score_zero = 1  # Base score for label == 0
    score_one_base = 10  # Base score for label == 1
    score_high_label_base = 20  # Base score for label > 1

    # Process data points with label == 0
    for i, label in enumerate(line_label):
        if label >= 0:
            raw_scores[i] = score_zero
    # Process data points with label == 1
    # Sort indices by length for label == 1
    one_indices = [i for i, label in enumerate(line_label) if label == 1]
    one_indices_sorted = sorted(one_indices, key=lambda i: line_length[i])
    for rank, i in enumerate(one_indices_sorted):
        raw_scores[i] = score_one_base + rank  # Increment score based on rank

    # Process data points with label > 1
    for i, label in enumerate(line_label):
        if label > 1:
            raw_scores[i] = score_high_label_base + label  # Ensure scores are greater than label == 0

    # Normalize raw scores to [0, 1]
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    normalized_scores = [
        (score - min_score) / (max_score - min_score) if max_score > min_score else 0
        for score in raw_scores
    ]
    
    return normalized_scores

def __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator=PADDED_Y_VALUE):
    mask = y_true == padding_indicator

    y_pred[mask] = float('-inf')
    y_true[mask] = 0.0

    _, indices = y_pred.sort(descending=True, dim=-1)
    return torch.gather(y_true, dim=1, index=indices)
                        
def dcg(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=PADDED_Y_VALUE):
    """
    Discounted Cumulative Gain at k.
    
    Compute DCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for DCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: DCG values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    actual_length = y_true.shape[1]

    if ats is None:
        ats = [actual_length]
    ats = [min(at, actual_length) for at in ats]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    discounts = (torch.tensor(1) / torch.log2(torch.arange(true_sorted_by_preds.shape[1], dtype=torch.float) + 2.0)).to(
        device=true_sorted_by_preds.device)

    gains = gain_function(true_sorted_by_preds)

    discounted_gains = (gains * discounts)[:, :np.max(ats)]

    cum_dcg = torch.cumsum(discounted_gains, dim=1)

    ats_tensor = torch.tensor(ats, dtype=torch.long) - torch.tensor(1)

    dcg = cum_dcg[:, ats_tensor]

    return dcg

def sinkhorn_scaling(mat, mask=None, tol=1e-6, max_iter=50):
    """
    Sinkhorn scaling procedure.
    :param mat: a tensor of square matrices of shape N x M x M, where N is batch size
    :param mask: a tensor of masks of shape N x M
    :param tol: Sinkhorn scaling tolerance
    :param max_iter: maximum number of iterations of the Sinkhorn scaling
    :return: a tensor of (approximately) doubly stochastic matrices
    """
    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)
        mat = mat.masked_fill(mask[:, None, :] & mask[:, :, None], 1.0)

    for _ in range(max_iter):
        mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=DEFAULT_EPS)
        mat = mat / mat.sum(dim=2, keepdim=True).clamp(min=DEFAULT_EPS)

        if torch.max(torch.abs(mat.sum(dim=2) - 1.)) < tol and torch.max(torch.abs(mat.sum(dim=1) - 1.)) < tol:
            break

    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)

    return mat

def deterministic_neural_sort(s, tau, mask):
    """
    Deterministic neural sort.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param s: values to sort, shape [batch_size, slate_length]
    :param tau: temperature for the final softmax function
    :param mask: mask indicating padded elements
    :return: approximate permutation matrices of shape [batch_size, slate_length, slate_length]
    """
    dev = s.device

    n = s.size()[1]
    one = torch.ones((n, 1), dtype=torch.float32, device=dev)
    s = s.masked_fill(mask[:, :, None], -1e8)
    A_s = torch.abs(s - s.permute(0, 2, 1))
    A_s = A_s.masked_fill(mask[:, :, None] | mask[:, None, :], 0.0)

    B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1)))

    temp = [n - m + 1 - 2 * (torch.arange(n - m, device=dev) + 1) for m in mask.squeeze(-1).sum(dim=1)]
    temp = [t.type(torch.float32) for t in temp]
    temp = [torch.cat((t, torch.zeros(n - len(t), device=dev))) for t in temp]
    scaling = torch.stack(temp).type(torch.float32).to(dev)  # type: ignore

    s = s.masked_fill(mask[:, :, None], 0.0)
    C = torch.matmul(s, scaling.unsqueeze(-2))

    P_max = (C - B).permute(0, 2, 1)
    P_max = P_max.masked_fill(mask[:, :, None] | mask[:, None, :], -np.inf)
    P_max = P_max.masked_fill(mask[:, :, None] & mask[:, None, :], 1.0)
    sm = torch.nn.Softmax(-1)
    P_hat = sm(P_max / tau)
    return P_hat

def sample_gumbel(samples_shape, device, eps=1e-10) -> torch.Tensor:
    """
    Sampling from Gumbel distribution.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param samples_shape: shape of the output samples tensor
    :param device: device of the output samples tensor
    :param eps: epsilon for the logarithm function
    :return: Gumbel samples tensor of shape samples_shape
    """
    U = torch.rand(samples_shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def stochastic_neural_sort(s, n_samples, tau, mask, beta=1.0, log_scores=True, eps=1e-10):
    """
    Stochastic neural sort. Please note that memory complexity grows by factor n_samples.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param s: values to sort, shape [batch_size, slate_length]
    :param n_samples: number of samples (approximations) for each permutation matrix
    :param tau: temperature for the final softmax function
    :param mask: mask indicating padded elements
    :param beta: scale parameter for the Gumbel distribution
    :param log_scores: whether to apply the logarithm function to scores prior to Gumbel perturbation
    :param eps: epsilon for the logarithm function
    :return: approximate permutation matrices of shape [n_samples, batch_size, slate_length, slate_length]
    """
    dev = s.device

    batch_size = s.size()[0]
    n = s.size()[1]
    s_positive = s + torch.abs(s.min())
    samples = beta * sample_gumbel([n_samples, batch_size, n, 1], device=dev)
    if log_scores:
        s_positive = torch.log(s_positive + eps)

    s_perturb = (s_positive + samples).view(n_samples * batch_size, n, 1)
    mask_repeated = mask.repeat_interleave(n_samples, dim=0)

    P_hat = deterministic_neural_sort(s_perturb, tau, mask_repeated)
    P_hat = P_hat.view(n_samples, batch_size, n, n)
    return P_hat

def neuralNDCG(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE, temperature=1., powered_relevancies=True, k=None,
               stochastic=False, n_samples=32, beta=0.1, log_scores=True):
    """
    NeuralNDCG loss introduced in "NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable
    Relaxation of Sorting" - https://arxiv.org/abs/2102.07831. Based on the NeuralSort algorithm.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param temperature: temperature for the NeuralSort algorithm
    :param powered_relevancies: whether to apply 2^x - 1 gain function, x otherwise
    :param k: rank at which the loss is truncated
    :param stochastic: whether to calculate the stochastic variant
    :param n_samples: how many stochastic samples are taken, used if stochastic == True
    :param beta: beta parameter for NeuralSort algorithm, used if stochastic == True
    :param log_scores: log_scores parameter for NeuralSort algorithm, used if stochastic == True
    :return: loss value, a torch.Tensor
    """

    if k is None:
        k = y_true.shape[1]

    mask = (y_true == padded_value_indicator)
    # Choose the deterministic/stochastic variant
    if stochastic:
        P_hat = stochastic_neural_sort(y_pred.unsqueeze(-1), n_samples=n_samples, tau=temperature, mask=mask,
                                       beta=beta, log_scores=log_scores)
    else:
        P_hat = deterministic_neural_sort(y_pred.unsqueeze(-1), tau=temperature, mask=mask).unsqueeze(0)

    # Perform sinkhorn scaling to obtain doubly stochastic permutation matrices
    P_hat = sinkhorn_scaling(P_hat.view(P_hat.shape[0] * P_hat.shape[1], P_hat.shape[2], P_hat.shape[3]),
                             mask.repeat_interleave(P_hat.shape[0], dim=0), tol=1e-6, max_iter=50)
    P_hat = P_hat.view(int(P_hat.shape[0] / y_pred.shape[0]), y_pred.shape[0], P_hat.shape[1], P_hat.shape[2])

    # Mask P_hat and apply to true labels, ie approximately sort them
    P_hat = P_hat.masked_fill(mask[None, :, :, None] | mask[None, :, None, :], 0.)
    y_true_masked = y_true.masked_fill(mask, 0.).unsqueeze(-1).unsqueeze(0)
    if powered_relevancies:
        y_true_masked = torch.pow(2., y_true_masked) - 1.

    ground_truth = torch.matmul(P_hat, y_true_masked).squeeze(-1)
    discounts = (torch.tensor(1.) / torch.log2(torch.arange(y_true.shape[-1], dtype=torch.float) + 2.)).to(y_pred.device)
    discounted_gains = ground_truth * discounts

    if powered_relevancies:
        idcg = dcg(y_true, y_true, ats=[k]).permute(1, 0)
    else:
        idcg = dcg(y_true, y_true, ats=[k], gain_function=lambda x: x).permute(1, 0)

    discounted_gains = discounted_gains[:, :, :k]
    ndcg = discounted_gains.sum(dim=-1) / (idcg + DEFAULT_EPS)
    idcg_mask = idcg == 0.
    ndcg = ndcg.masked_fill(idcg_mask.repeat(ndcg.shape[0], 1), 0.)

    assert (ndcg < 0.).sum() >= 0, "every ndcg should be non-negative"
    if idcg_mask.all():
        return torch.tensor(0.)

    mean_ndcg = ndcg.sum() / ((~idcg_mask).sum() * ndcg.shape[0])  # type: ignore
    return -1. * mean_ndcg  # -1 cause we want to maximize NDCG

def approxNDCGLoss(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, alpha=1.):
    """
    Loss based on approximate NDCG introduced in "A General Approximation Framework for Direct Optimization of
    Information Retrieval Measures". Please note that this method does not implement any kind of truncation.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param alpha: score difference weight used in the sigmoid function
    :return: loss value, a torch.Tensor
    """
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)
    padded_pairs_mask.diagonal(dim1=-2, dim2=-1).zero_()

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum((torch.pow(2, y_true_sorted) - 1) / D, dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    # Here we approximate the ranking positions according to Eqs 19-20 and later approximate NDCG (Eq 21)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :])
    scores_diffs[~padded_pairs_mask] = 0.
    approx_pos = 1. + torch.sum(padded_pairs_mask.float() * (torch.sigmoid(-alpha * scores_diffs).clamp(min=eps)), dim=-1)
    approx_D = torch.log2(1. + approx_pos)
    approx_NDCG = torch.sum((G / approx_D), dim=-1)

    return -torch.mean(approx_NDCG)

# Reference: https://github.com/Cadene/block.bootstrap.pytorch/blob/e938e9e269d2ec67499db327b903edbe821fed5f/block/models/networks/fusions/fusions.py
class MLB(nn.Module):
    def __init__(self,
            n_inputs,
            hidden_dim=1200,
            output_dim=1,
            activ_input='relu',
            activ_output='relu',
            normalize=False,
            dropout_input=0.,
            dropout_pre_lin=0.,
            dropout_output=0.):
        super(MLB, self).__init__()
        self.n_inputs = n_inputs
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        # Modules
        self.linear0 = nn.Linear(n_inputs, hidden_dim)
        self.linear1 = nn.Linear(n_inputs, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_x):
        x, first_token = input_x
        x0 = self.linear0(x)
        x1 = self.linear1(first_token)

        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        z = x0 * x1

        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z,p=2)

        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)

        z = self.linear_out(z)

        if self.activ_output:
            z = getattr(F, self.activ_output)(z)

        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    def forward(self, Q, K, V):
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        # Weighted sum of values
        output = torch.matmul(attn_weights, V)
        return output, attn_weights
    
class RankNet(nn.Module):
    def __init__(self, 
                 n_inputs, 
                 layer_num, 
                 hidden_size,
                 enable_ln=False,
                 enable_attn=False,
                 enable_res=False,
                 enable_mlb=False,
                 enable_classifier=False,
                 code_num=32,
                 drop_out_rate=0,
                 act="relu",
                 device="cuda", 
                 ):
        super(RankNet, self).__init__()
        self.n_inputs = n_inputs
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.enable_attn = enable_attn
        self.enable_res = enable_res
        self.enable_mlb = enable_mlb
        self.res = None
        self.attention = None
        self.V = None
        self.K = None
        self.output_proj = None
        self.drop_out_rate = drop_out_rate
        self.enable_ln = enable_ln
        self.code_num = code_num
        self.enable_classifier = enable_classifier
        self.agg = None
        self.act = act
        act_func = getattr(nn, act)
        if enable_classifier:
            self.classfier = nn.Linear(hidden_size, 2)
            
        if enable_attn:
            self.down_sample = nn.Sequential(*[nn.Linear(n_inputs, hidden_size), act_func(inplace=True)])
            self.attention = ScaledDotProductAttention(hidden_size)
            self.V = torch.nn.Parameter(torch.randn(code_num, hidden_size))
            self.K = torch.nn.Parameter(torch.randn(code_num, hidden_size))
            self.output_proj = nn.Bilinear(hidden_size, hidden_size, hidden_size)

            blocks = []
            for _ in range(layer_num-2):
                blocks.append(act_func(inplace=True))
                if self.drop_out_rate > 0:
                    blocks.append(nn.Dropout(p=self.drop_out_rate))
                blocks.append(nn.Linear(hidden_size, hidden_size))
            blocks.append(act_func(inplace=True))
            blocks.append(nn.Linear(hidden_size, 1))
            self.blocks = nn.Sequential(*blocks)
            #self.attention = LearnableAttention(n_inputs, hidden_size)
            #blocks = [nn.Linear(hidden_size, hidden_size)]
            #for _ in range(layer_num-2):
            #    blocks.append(nn.ReLU(inplace=True))
            #    blocks.append(nn.Linear(hidden_size, hidden_size))
            #blocks.append(nn.ReLU(inplace=True))
            #blocks.append(nn.Linear(hidden_size, 1))
            #self.blocks = nn.Sequential(*blocks)
        elif enable_res:
            self.res = nn.Sequential(*[nn.Linear(n_inputs, hidden_size), act_func(inplace=True)])
            blocks = [nn.Linear(hidden_size, hidden_size)]
            for _ in range(layer_num-2):
                blocks.append(act_func(inplace=True))
                blocks.append(nn.Linear(hidden_size, hidden_size))
            blocks.append(act_func(inplace=True))
            blocks.append(nn.Linear(hidden_size, 1))
            self.blocks = nn.Sequential(*blocks)
        elif enable_mlb:
            blocks = [MLB(n_inputs, hidden_size, output_dim=hidden_size, dropout_pre_lin=self.drop_out_rate)]
            for _ in range(layer_num-2):
                blocks.append(nn.Linear(hidden_size, hidden_size))
                blocks.append(act_func(inplace=True))
            blocks.append(nn.Linear(hidden_size, 1))
            self.blocks = nn.Sequential(*blocks)
        else:
            blocks = [nn.Linear(n_inputs, hidden_size)]
            for _ in range(layer_num-1):
                blocks.append(act_func(inplace=True))
                if self.drop_out_rate > 0:
                    blocks.append(nn.Dropout(p=self.drop_out_rate))
                blocks.append(nn.Linear(hidden_size, hidden_size))
            blocks.append(act_func(inplace=True))
            blocks.append(nn.Linear(hidden_size, 1))
            self.blocks = nn.Sequential(*blocks)
            #self.attention = None
            #self.V = None
            #self.K = None
            #self.output_proj = None
        
        self.sigmoid = nn.Sigmoid()
        
        self.device = device
        self.to(device)
    
    def share_forward(self, input_1, first_token=None):
        input_1 = input_1.to(self.device)
        if self.enable_ln is True:
            input_1, mu, std = LN(input_1)
        if self.enable_attn:
            assert first_token is not None
            first_token = first_token.to(self.device)
            if len(first_token.shape) == 1:
                first_token = first_token.unsqueeze(0)
            if len(input_1.shape) == 2:
                input_1 = input_1.unsqueeze(0)
            first_token = self.down_sample(first_token)
            input_1 = self.down_sample(input_1)
            attn_output, attn_weights = self.attention(first_token, self.K, self.V)
            attn_output = attn_output.reshape(-1, 1, self.hidden_size)
            repeat_counts = torch.tensor([input_1.shape[1]]).to(self.device)
            attn_output = torch.repeat_interleave(attn_output, repeat_counts, dim=1)
            output = self.output_proj(input_1, attn_output)
            output = output + input_1
        elif self.enable_res:
            first_proj = self.res(first_token)
            # (1, hidden_size)
            input_1 = self.res(input_1)
            # (L,)
            output = input_1 + first_proj
        elif self.enable_mlb:
            output = [input_1, first_token]
        else:
            output = input_1
        return self.blocks[:-1](output)
        
    def forward_both(self, input_1, first_token=None, length=None):
        """
        Args
        ---
            input_1: (batch_size, length, hidden_dim) or (length, hidden_dim)
            first_token: (batch_size, hidden_dim) or (hidden_dim)
            length: (batch_size,) or None
            agg: aggregation method. Either "last" or "mean"
        """
        agg = self.agg
        assert agg in ["last", "mean", "rank", "max"]
        if length is None:
            assert len(input_1.shape) == 2
        else:
            length = convert_to_torch_tensor(length)
        penu_output = self.share_forward(input_1, first_token=first_token)
        ranking_output = torch.squeeze(self.sigmoid(self.blocks[-1](penu_output)), dim=-1) #(N, L)
        if agg == "last":
            if length is None:
                penu_output = penu_output.reshape(1, -1, self.hidden_size)
                last_tokens = penu_output[:, -1]
            else:
                batch_indices = torch.arange(input_1.shape[0])
                last_token_indices = length - 1  # Indices of the last valid token
                last_tokens = penu_output[batch_indices, last_token_indices]
            classification = self.classfier(last_tokens)
        elif agg == "max":
            if length is None:
                classification = self.classfier(penu_output.max(dim=0).values)
            else:
                max_value = masked_max(penu_output, length)
                classification = self.classfier(max_value)
        elif agg == "mean":
            if length is None:
                classification = self.classfier(penu_output.mean(dim=0))
            else:
                masks = torch.arange(input_1.shape[1]).unsqueeze(0).to(length.device) < length.unsqueeze(1)  # Create masks for valid positions
                masked_input = penu_output * masks.unsqueeze(-1) 
                sums = masked_input.sum(dim=1)  # Sum over the length dimension
                averages = sums / length.unsqueeze(-1)  # Divide by lengths
                classification = self.classfier(averages)
        elif agg == "rank":
            ranking_output = ranking_output.squeeze(0)
            _, max_indices = torch.max(ranking_output, dim=-1)  # max_indices: Shape (N,)
            if length is None:
                classification = self.classfier(penu_output[max_indices])
            else:
                classification = self.classfier(penu_output[torch.arange(input_1.shape[0]), max_indices])
        #class_scores = nn.functional.log_softmax(classification, dim=-1)
        return ranking_output, classification
    
    
    
    def forward(self, input_1, first_token=None):
        penu_output = self.share_forward(input_1, first_token=first_token)
        output = self.blocks[-1](penu_output)
        return torch.squeeze(self.sigmoid(output), dim=-1)
    
    def fit(self, train_x, train_y, candidate_token_dict, verbose=False, learning_param={}):
        for param in self.parameters():
            param.requires_grad = True
        self.train()
        dataset = DictDataset(train_x, train_y, candidate_token_dict)
        loss_fn = learning_param.get("loss_fn", "neuralNDCG")
        num_epochs = learning_param.get("num_epochs", 20)
        batch_size = learning_param.get("batch_size", 2)
        beta1 = learning_param.get("beta1", 0.9)
        weight_decay = learning_param.get("weight_decay", 0)
        lr = learning_param.get("lr", 1e-4)
        alpha = learning_param.get("alpha", 1)
        co_training = learning_param.get("co_training", False)
        if self.enable_classifier:
            agg = learning_param.get("agg", "last")
            class_loss_fn = nn.CrossEntropyLoss()
            self.agg = agg
        if loss_fn == "approxNDCGLoss":
            loss_function = approxNDCGLoss
        elif loss_fn == "neuralNDCG":
            loss_function = neuralNDCG
        else:
            raise ValueError("Invalid loss function. Must be either approxNDCGLoss or neuralNDCG")
        collate_fn = self.get_collate_fn(enable_attn=self.enable_attn)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), 
                                     lr=lr, 
                                     betas=(beta1, 0.999), 
                                     weight_decay=weight_decay)
        loss_list = list()
        for epoch in (range(num_epochs)):
            total_loss = 0
            inst_counter = 0
            #for x, y, first_token in train_loader:
            for x, y in train_loader:
                if x.shape[1] == 1 and ((self.enable_classifier is False) or (self.enable_classifier is True and co_training is False)):
                    # The input has only one element. Skip this iteration.
                    continue
                class_label = ((y > 0).sum(dim=-1) > 0).long()
                length = (y >= 0).sum(dim=-1) 
                ranking_idx = torch.logical_and(class_label, length>1)
                if sum(ranking_idx) == 0 and co_training is False:
                    continue
                optimizer.zero_grad()
                
                if self.enable_classifier and co_training is True:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    ranking_output, class_scores = self.forward_both(x, length=length)
                    class_loss = class_loss_fn(class_scores, class_label)
                    ranking_idx = torch.logical_and(class_label, length>1)
                    if sum(ranking_idx) > 0:
                        ranking_loss = loss_function(ranking_output[ranking_idx], y[ranking_idx])
                    else:
                        ranking_loss = 0
                    loss = alpha * ranking_loss + class_loss
                else:
                    x = x[ranking_idx].to(self.device)
                    y = y[ranking_idx].to(self.device)
                    output = self(x)
                    loss = loss_function(output, y)
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(am.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                #print(compute_grad_norm(am))
                total_loss += loss.item()
                inst_counter += len(x)
            if verbose:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / inst_counter}")
            loss_list.append(total_loss / inst_counter)
        self.eval()
        return loss_list
    
    def fit_classifier(self, train_x, train_y, candidate_token_dict, first_token_dict=None, verbose=False, learning_param={}):
        # Freeze layer1 parameters
        freeze = learning_param.get("freeze", False)
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            for param in self.classfier.parameters():
                param.requires_grad = True
        self.train()
        dataset = DictDataset(train_x, train_y, candidate_token_dict, first_token_dict)
        num_epochs = learning_param.get("num_epochs", 20)
        batch_size = learning_param.get("batch_size", 2)
        beta1 = learning_param.get("beta1", 0.9)
        weight_decay = learning_param.get("weight_decay", 1e-4)
        lr = learning_param.get("lr", 1e-4)
        agg = learning_param.get("agg", "last")
        self.agg = agg
        class_loss_fn = nn.CrossEntropyLoss()
        collate_fn = self.get_collate_fn(enable_attn=self.enable_attn)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), 
                                     lr=lr, 
                                     betas=(beta1, 0.999), 
                                     weight_decay=weight_decay
                                     )
        loss_list = list()
        for epoch in (range(num_epochs)):
            total_loss = 0
            inst_counter = 0
            for item in train_loader:
                x = item[0]
                y = item[1]
                if first_token_dict is not None:
                    first_token = item[2]
                else:
                    first_token = None
                optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                length = (y >= 0).sum(dim=-1) 
                class_label = ((y > 0).sum(dim=-1) > 0).long()
                ranking_output, class_scores = self.forward_both(x, length=length, first_token=first_token)
                class_loss = class_loss_fn(class_scores, class_label)
                class_loss.backward()
                optimizer.step()
                total_loss += class_loss.item()
                inst_counter += len(x)
            if verbose:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / inst_counter}")
            loss_list.append(total_loss / inst_counter)
        self.eval()
        return loss_list
    
    def get_collate_fn(self, fillvalue=-1, enable_attn=False):
        def collate_fn(batch):
            if not enable_attn:
                x, y, candidate_token = zip(*batch)
                first_token = None
            else:
                x, y, candidate_token, first_token = zip(*batch)
                first_token = torch.stack(first_token)
            #lengths = [len(seq) for seq in x]
            rank_y = []
            for idx, item in enumerate(y):
                rank_y.append(assign_and_normalize_scores(item, candidate_token[idx]))
                
            # Find the maximum length
            max_length = max(len(item) for item in rank_y)

            # Pad sequences with zeros and convert to tensor
            padded_rank_y = torch.tensor(
                list(zip_longest(*rank_y, fillvalue=fillvalue)), dtype=torch.float32
            ).transpose(0, 1)  # Transpose to get (batch_size, max_length)
            
            padded_x = pad_sequence(x, batch_first=True, padding_value=-1)
            
            if first_token is not None:
                return padded_x, padded_rank_y, first_token
            else:
                return padded_x, padded_rank_y
        return collate_fn
    
    def predict(self, x, first_token=None):
        with torch.inference_mode():
            x = x.to(self.device)
            if first_token is not None:
                first_token = first_token.to(self.device)
            if self.enable_classifier:
                ranking_output, class_scores = self.forward_both(x, first_token=first_token)
                ranking_output = ranking_output.cpu()
                class_scores = class_scores.cpu()
            else:
                ranking_output = self(x, first_token=first_token).cpu()
                class_scores = None
            ranking_output = ranking_output.squeeze(0)
            ranking = torch.sort(ranking_output, dim=-1, descending=True).indices.tolist()
            return ranking, ranking_output, class_scores
    
    def predict_proba(self, first_token, x):
        x = convert_to_torch_tensor(x)
        x = x.to(self.device)
        ranking, y, class_scores = self.predict(x, first_token=first_token)
        return ranking, y, class_scores
    

    @classmethod
    def load_from_memory(cls, loaded_info, device=None):
        n_inputs = loaded_info["meta"]["n_inputs"]
        layer_num = loaded_info["meta"]["layer_num"]
        hidden_size = loaded_info["meta"]["hidden_size"]
        enable_attn = loaded_info["meta"].get("enable_attn", False)
        enable_res = loaded_info["meta"].get("enable_res", False)
        enable_mlb = loaded_info["meta"].get("enable_mlb", False)
        enable_ln = loaded_info["meta"].get("enable_ln", False)
        enable_classifier = loaded_info["meta"].get("enable_classifier", False)
        drop_out_rate = loaded_info["meta"].get("drop_out_rate", 0)
        code_num = loaded_info["meta"].get("code_num", 32)
        agg = loaded_info["meta"].get("agg", None)
        act = loaded_info["meta"].get("act", "relu")
        parameter_dict = {
            "n_inputs": n_inputs,
            "layer_num": layer_num,
            "hidden_size": hidden_size,
            "enable_attn": enable_attn,
            "enable_res": enable_res,
            "enable_mlb": enable_mlb,
            "enable_ln": enable_ln,
            "enable_classifier": enable_classifier,
            "drop_out_rate": drop_out_rate,
            "code_num": code_num,
            "act": act
        }
        if device is not None:
            parameter_dict["device"] = device
        model = cls(**parameter_dict)
        model.load_state_dict(loaded_info["state_dict"])
        #model.encoder = encoder.to(device)
        model.agg = agg
        return model
    
    @classmethod
    def load(cls, path, device=None):
        loaded_info = torch.load(path)
        n_inputs = loaded_info["meta"]["n_inputs"]
        layer_num = loaded_info["meta"]["layer_num"]
        hidden_size = loaded_info["meta"]["hidden_size"]
        enable_attn = loaded_info["meta"].get("enable_attn", False)
        enable_res = loaded_info["meta"].get("enable_res", False)
        enable_mlb = loaded_info["meta"].get("enable_mlb", False)
        enable_ln = loaded_info["meta"].get("enable_ln", False)
        enable_classifier = loaded_info["meta"].get("enable_classifier", False)
        drop_out_rate = loaded_info["meta"].get("drop_out_rate", 0)
        code_num = loaded_info["meta"].get("code_num", 32)
        agg = loaded_info["meta"].get("agg", "last")
        act = loaded_info["meta"].get("act", "relu")
        parameter_dict = {
            "n_inputs": n_inputs,
            "layer_num": layer_num,
            "hidden_size": hidden_size,
            "enable_attn": enable_attn,
            "enable_res": enable_res,
            "enable_mlb": enable_mlb,
            "enable_ln": enable_ln,
            "enable_classifier": enable_classifier,
            "drop_out_rate": drop_out_rate,
            "code_num": code_num,
            "agg": agg,
            "act": act
        }
        if device is not None:
            parameter_dict["device"] = device
        model = cls(**parameter_dict)
        model.load_state_dict(loaded_info["state_dict"])
        #model.encoder = encoder.to(device)
        return model
    
    def pack(self):
        return {
            "meta": {
                "n_inputs": self.n_inputs,
                "layer_num": self.layer_num,
                "hidden_size": self.hidden_size,
                "enable_attn": self.enable_attn,
                "enable_res": self.enable_res,
                "enable_mlb": self.enable_mlb,
                "enable_ln": self.enable_ln,
                "enable_classifier": self.enable_classifier,
                "drop_out_rate": self.drop_out_rate,
                "code_num": self.code_num,
                "agg": self.agg,
                "act": self.act
            },
            "state_dict": self.state_dict()
        }
        
    def save(self, path):
        torch.save(self.pack(), path)
    