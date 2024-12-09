import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn import svm
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE
from torch.nn.utils.rnn import pad_sequence
from loguru import logger
from torch.utils.data import Dataset
import sispca
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC
from abc import abstractmethod
from itertools import zip_longest
from torch.nn.utils.rnn import pad_sequence

from method.sae_model import Autoencoder, NaiveAutoEncoder
# from sparse_autoencoder import LN

PADDED_Y_VALUE = -1
DEFAULT_EPS = 1e-5

def collect_attention_map(attn_snapshot, layer, input_token_length, output_seg, numeric_stability=1e-7):
    """
    Collect features as hallucination detection mentioned in the paper.
    ---
    Args:
        attn_snapshot: Tuple[Tensor]. Each layer's multi-head attention map, 
            which has the shape [1, head_num, token_num, token_num]. It should be the full input version.
        layer: int. The index of the selected layer as feature.
        input_token_length: int. The token length of the input prompt.
        output_seg: List[List]: List of segment tokens. It should be str_output version
    Return:
        vt: torch.tensor. vt in the paper with the shape [multi_head_num]
    """
    
    al_context = attn_snapshot[layer][0][:, -1, :input_token_length]
    al_context = torch.mean(al_context, dim=-1)
    al_result = list()
    for seg in output_seg:
        real_seg = [i + input_token_length for i in seg]
        attn_seg = attn_snapshot[layer][0][:, -1, real_seg]
        al_attn_score = torch.mean(attn_seg, dim=-1)
        al_attn_score = al_context / (al_context + al_attn_score + numeric_stability)
        al_result.append(al_attn_score)
    #al_new = attn_snapshot[layer][0][:, -1, input_token_length:]
    #al_new = torch.mean(al_new, dim=-1)
    
    #lr = al_context / (al_context + al_new)
    #return lr
    return al_result
    
def collect_hidden_states(hidden_map, input_token_length, output_seg, encoder, before_enc=None):
    """
    Collect hidden states as features for hallucination detection.
    ---
    Args:
        hidden_map: Tuple[Tensor]. Each layer's hidden states, which has the shape [1, token_num, hidden_size].
        output_seg: List[List]: List of segment tokens. It should be str_output version
    """
    if before_enc is None:
        hidden_all = list()
        for seg in output_seg:
            real_seg = [i + input_token_length  for i in seg]
            line_split_token = max(real_seg)
            hidden_seg = hidden_map[line_split_token] # (token_num, hidden_size)
            hidden_all.append(hidden_seg)
        hidden_all = torch.stack(hidden_all)
    else:
        if isinstance(before_enc, np.ndarray):
            before_enc = torch.from_numpy(before_enc)
        hidden_all = before_enc
    if encoder is not None:
        with torch.inference_mode():
            hidden_all = hidden_all.to(encoder.device)
            latent_activations, info = encoder.encode(hidden_all)
        return latent_activations.cpu(), hidden_all.cpu()
    else:
        return None, hidden_all.cpu()

def flat_data_dict(data_dict):
    x = dict()
    for dataset in data_dict:
        for key in data_dict[dataset]:
            x[f"{dataset}_{key}"] = data_dict[dataset][key]
    return x
"""
def flat_data_dict(data_dict_x, data_dict_y):
    x = dict()
    y = dict()
    for dataset in data_dict_x:
        for key in data_dict_x[dataset]:
            x[f"{dataset}_{key}"] = data_dict_x[dataset][key]  
            y[f"{dataset}_{key}"] = data_dict_y[dataset][key]
    return x, y
"""

def convert_to_torch_tensor(array):
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    elif isinstance(array, torch.Tensor):
        return array
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")

class SupervisedPrjection():
    def __init__(self):
        self.projection_matrix = None
        self.latent_info = None
        self.key_order = None
        self.n_latent_sub = None
        self.mean = None
        self.std = None
        self.vector_norm = None
    
    def fit_projector(self, 
                      supervised_x, 
                      latent_info,
                      key_order, 
                      batch_size=2048,
                      max_epochs=500, 
                      early_stopping_patience=5,
                      vector_norm=True):
        """
        latent_info: dict. key -> [target_type, label, latent_subspace]
            target_type: str. "continuous" or "categorial"
        """
        self.vector_norm = vector_norm
        if vector_norm:
            supervised_x = self.norm_x(supervised_x)
        sdata = sispca.SISPCADataset(
            data = supervised_x, # (n_sample, n_feature)
            target_supervision_list = [
                sispca.Supervision(target_data=latent_info[key][1], target_type=latent_info[key][0])
                for key in key_order
            ]
        )
        n_latent_sub = [latent_info[key][2] for key in key_order]
        sispca_model = sispca.SISPCA(
            sdata, 
            n_latent_sub=n_latent_sub,
            lambda_contrast=10,
            kernel_subspace='linear',
            solver='eig'
        )
        sispca_model.fit(batch_size = batch_size, 
                         max_epochs = max_epochs, 
                         early_stopping_patience = early_stopping_patience, 
                         accelerator="auto"
                         )
        
        self.projection_matrix = sispca_model.U
        self.latent_info = latent_info
        self.key_order = key_order
        self.n_latent_sub = n_latent_sub
        self.mean = sispca_model._dataset.mean
        self.std = sispca_model._dataset.std
        del sispca_model
    
    def forward(self, x, feature_idx=0, topk=None, norm=True):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        # https://github.com/JiayuSuPKU/sispca/blob/539dae1c9cc02b6e575c8c9157926665d7504e3f/sispca/data.py#L96
        if self.vector_norm and norm == True:
            # Enable norm in training
            # and do not explicitly disable it in prediction
            x = self.norm_x(x)
        if self.mean is not None:
            x = x - self.mean
        if self.std is not None:
            x = x / self.std
        #x = sispca.utils.normalize_col(x, center=True, scale=False)
        proj = self.projection_matrix[:, :self.n_latent_sub[feature_idx]]
        if topk is not None:
            proj = proj[:, :topk]
        return x @ proj # We Assume the first latent subspace is the most important one

    def norm_x(self, vector_data):
        return vector_data / np.linalg.norm(vector_data, axis=1, keepdims=True)
    
    def save(self, path):
        torch.save({
            "projection_matrix": self.projection_matrix,
            "latent_info": self.latent_info,
            "key_order": self.key_order,
            "n_latent_sub": self.n_latent_sub,
            "mean": self.mean,
            "std": self.std,
            "vector_norm": self.vector_norm
            }, 
            path
        )
    
    @classmethod
    def load(cls, path):
        loaded_info = torch.load(path)
        model = cls()
        model.projection_matrix = loaded_info["projection_matrix"]
        model.latent_info = loaded_info["latent_info"]
        model.key_order = loaded_info["key_order"]
        model.n_latent_sub = loaded_info["n_latent_sub"]
        model.mean = loaded_info["mean"]
        model.std = loaded_info["std"]
        model.vector_norm = loaded_info["vector_norm"]
        return model
        

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
        
class AttentionModule(nn.Module):
    def __init__(self, 
                 code_num, 
                 hidden_dim, 
                 K, 
                 output_dim=2,
                 enable_input_proj=False,
                 ln=True):
        super(AttentionModule, self).__init__()
        self.attention = ScaledDotProductAttention(hidden_dim)
        self.V = torch.nn.Parameter(torch.randn(code_num, hidden_dim))
        self.register_buffer('K', K)
        #self.K = K # Constant
        self.K.requires_grad = False
        self.output_proj = nn.Bilinear(hidden_dim, hidden_dim, output_dim)  # Bilinear layer
        self.hidden_dim = hidden_dim
        self.enable_input_proj = enable_input_proj
        self.left_proj = nn.Linear(hidden_dim, hidden_dim)
        self.right_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ln = ln
        self.relu = nn.ReLU()
        if enable_input_proj:
            self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.input_proj = None

    def LN(self, x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = x.mean(dim=-1, keepdim=True)
        x = x - mu
        std = x.std(dim=-1, keepdim=True)
        x = x / (std + eps)
        return x, mu, std
        
    def forward(self, context_vec, X, repeat_counts=None):
        # X: M * N
        # Context_vec: (1, hidden_dim)
        if repeat_counts == None:
            repeat_counts = torch.tensor([len(X)]).to(self.device)
        attn_output, attn_weights = self.attention(context_vec, self.K, self.V) # Output: 1*hidden_dim
        
        # Project the output back to d_model (optional, for integration into larger models)
        if self.ln is True:
            X, mu, std = self.LN(X)
        if self.input_proj is None:
            output = X
        else:
            output = self.input_proj(X) # M*hidden_dim
        #attn_output_expanded = attn_output.unsqueeze(0)
        #attn_output_expanded = attn_output.reshape(-1, output.size(0), self.hidden_dim)  # (batch_size, M, hidden_dim)
        # (M*hidden_dim), (1*hidden_dim).T
        attn_output = torch.repeat_interleave(attn_output, repeat_counts, dim=0)
        #attn_output = attn_output.repeat(output.shape[0], 1)
        output = self.left_proj(output)
        output = self.relu(output)

        attn_output = self.right_proj(attn_output)
        attn_output = self.relu(attn_output)
        output = self.output_proj(output, attn_output)
        #output = self.relu(output)
        output = nn.functional.log_softmax(output, dim=-1)
        return output, attn_weights # M * 1

    def predict_proba(self, context_vec, X, repeat_counts=None):
        self.eval()
        context_vec = convert_to_torch_tensor(context_vec)
        if repeat_counts is not None:
            repeat_counts = convert_to_torch_tensor(repeat_counts)
            repeat_counts.to(self.device)
        X = convert_to_torch_tensor(X).to(self.device)
        if len(context_vec.shape) < 2:
            context_vec = context_vec.unsqueeze(0)
        context_vec = context_vec.to(self.device)
        with torch.inference_mode():
            output, _ = self.forward(context_vec, X, repeat_counts)
            probabilities = torch.exp(output)
        return probabilities.cpu().squeeze(0)

    @property
    def device(self):
        return next(self.parameters()).device

class RiskPredictor(ABC):
    @abstractmethod
    def fit(self):
        return NotImplemented
    
    def save(self, path):
        return NotImplemented
    
    @abstractmethod
    def predict(self):
        return NotImplemented
    
    @abstractmethod
    def predict_using_recorder(self):
        return NotImplemented
    
    @abstractmethod
    def evaluate(self):
        return NotImplemented
    

class RankNet(nn.Module):
    def __init__(self, n_inputs, layer_num, hidden_size, device="cuda", lr=1e-4, num_epochs=20, batch_size=4):
        super(RankNet, self).__init__()
        self.n_inputs = n_inputs
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        blocks = [nn.Linear(n_inputs, hidden_size)]
        for _ in range(layer_num-1):
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.Linear(hidden_size, hidden_size))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Linear(hidden_size, 1))
        self.blocks = nn.Sequential(*blocks)
        
        self.sigmoid = nn.Sigmoid()
        
        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.to(device)
        

    def forward(self, input_1):
        return torch.squeeze(self.sigmoid(self.blocks(input_1)), dim=-1)
    
    def fit(self, train_x, train_y, candidate_token_dict):
        dataset = DictDataset(train_x, train_y, candidate_token_dict)
        
        collate_fn = self.get_collate_fn()
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        loss_function = RankNet.approxNDCGLoss
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        for epoch in range(self.num_epochs):
            total_loss = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                #context = context.to(device)
                #lengths = lengths.to(device)

                output = self(x)
                
                loss = loss_function(output, y)
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(am.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                #print(compute_grad_norm(am))
                total_loss += loss.item()
    
    def get_collate_fn(self, fillvalue=-1):
        def collate_fn(batch):
            x, y, candidate_token = zip(*batch)
            #lengths = [len(seq) for seq in x]
            rank_y = []
            for idx, item in enumerate(y):
                rank_y.append(RankNet.assign_and_normalize_scores(item, candidate_token[idx]))
                
            # Find the maximum length
            max_length = max(len(item) for item in rank_y)

            # Pad sequences with zeros and convert to tensor
            padded_rank_y = torch.tensor(
                list(zip_longest(*rank_y, fillvalue=fillvalue)), dtype=torch.float32
            ).transpose(0, 1)  # Transpose to get (batch_size, max_length)
            
            padded_x = pad_sequence(x, batch_first=True, padding_value=-1)
            
            return padded_x, padded_rank_y
        return collate_fn
    
    def predict(self, x):
        with torch.inference_mode():
            x = x.to(self.device)
            y = self(x).cpu()
            ranking = torch.sort(y, dim=-1, descending=True).indices.tolist()
            return ranking, y
    
    def predict_proba(self, x):
        x = convert_to_torch_tensor(x)
        x = x.to(self.device)
        ranking, y = self.predict(x)
        return ranking, y
    

    @classmethod
    def load(cls, path, device=None):
        loaded_info = torch.load(path)
        n_inputs = loaded_info["meta"]["n_inputs"]
        layer_num = loaded_info["meta"]["layer_num"]
        hidden_size = loaded_info["meta"]["hidden_size"]
        lr = loaded_info["meta"]["lr"]
        num_epochs = loaded_info["meta"]["num_epochs"]
        batch_size = loaded_info["meta"]["batch_size"]
        parameter_dict = {
            "n_inputs": n_inputs,
            "layer_num": layer_num,
            "hidden_size": hidden_size,
            "lr": lr,
            "num_epochs": num_epochs,
            "batch_size": batch_size
        }
        if device is not None:
            parameter_dict["device"] = device
        model = cls(**parameter_dict)
        model.load_state_dict(loaded_info["state_dict"])
        #model.encoder = encoder.to(device)
        return model
    
    def save(self, path):
        torch.save({
            "meta": {
                "n_inputs": self.n_inputs,
                "layer_num": self.layer_num,
                "hidden_size": self.hidden_size,
                "lr": self.lr,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size
            },
            "state_dict": self.state_dict()
        }, path)
    
    @staticmethod
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

    @staticmethod
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


        
class EncoderClassifier(RiskPredictor):
    def __init__(self):
        self.fit_model_param = None
        self.clf = None
        self.model_type = None
        self.cache = None
        self.vector_norm = None
        self.external_dim_red_flag = False
        self.dim_red = None
        self.hidden_first = None
        self.encoder = None
        self.sorting_code = False
        
    def fit(self,
            train_info,  
            model_type, 
            fit_model_param,
            encoder,
            vector_norm=False,
            external_proj=None 
            ):
        self.fit_model_param = fit_model_param
        self.vector_norm = vector_norm
        already_norm = False
        train_x = train_info["train_x"]
        train_y = train_info["train_y"]
        if vector_norm is True:
            train_x = train_x / np.linalg.norm(train_x, axis=1, keepdims=True)
            already_norm = True
        if "dimension_reduction" in self.fit_model_param:
            dim_red_param = self.fit_model_param.pop("dimension_reduction")
            if "svd" in dim_red_param:
                self.dim_red = TruncatedSVD(**dim_red_param["svd"])
                train_x = self.dim_red.fit_transform(train_x)
        if external_proj is not None:
            self.dim_red = external_proj
            self.external_dim_red_flag = True
            train_x = self.dim_red.forward(train_x, norm=not already_norm).numpy()
            #train_x = self.dim_red.fit_transform(train_x, norm=not already_norm)
                
        if "balanced" in self.fit_model_param:
            balanced = self.fit_model_param.pop("balanced")
            if balanced:
                smote = SMOTE()
                train_x, train_y = smote.fit_resample(train_x, train_y)
        if "sorting_code" in self.fit_model_param:
            self.sorting_code = self.fit_model_param.pop("sorting_code")
            train_x = obtain_sorted_code(train_x, encoder.activation.k)
        else:
            self.sorting_code = False
        self.model_type = model_type
        self.encoder = encoder
        if self.model_type.lower() == "logistic":
            self.clf = LogisticRegression(**fit_model_param).fit(train_x, train_y)
        elif self.model_type.lower() == "svm":
            self.clf = svm.SVC(probability=True, **fit_model_param).fit(train_x, train_y)
        elif self.model_type.lower() == "mlp":
            self.clf = MLPClassifier(**fit_model_param).fit(train_x, train_y)
        elif self.model_type.lower() == "lstm":
            self.clf = LSTMPredictor(**fit_model_param).fit(train_x, train_y)
        elif self.model_type.lower() == "attn":
            self.clf = AttentionModule(**fit_model_param)
        
    def save(self, path):
        if self.encoder is not None:
            encoder_device = self.encoder.device
            encoder_state_dict = self.encoder.cpu().state_dict()
            encoder_type = self.encoder.__class__.__name__
        else:
            encoder_device = None
            encoder_state_dict = None
            encoder_type = None
        torch.save({"model": self.clf,
                     "dim_red": self.dim_red, 
                     "param": self.fit_model_param, 
                     "model_type": self.model_type,
                     "device": encoder_device,
                     "encoder": encoder_state_dict,
                     "encoder_type": encoder_type,
                     "sorting_code": self.sorting_code,
                     "vector_norm": self.vector_norm,
                     "external_dim_red_flag": self.external_dim_red_flag
                     }, 
                    path
                    )
    
    @classmethod
    def load(cls, path):
        loaded_info = torch.load(path)
        model = cls()
        model.clf = loaded_info["model"]
        model.fit_model_param = loaded_info["param"]
        model.model_type = loaded_info["model_type"]
        model.dim_red = loaded_info["dim_red"]
        device = loaded_info["device"]
        model.sorting_code = loaded_info["sorting_code"] if "sorting_code" in loaded_info else False
        encoder_type = loaded_info.get("encoder_type", "Autoencoder")
        if loaded_info["encoder"] is not None:
            if encoder_type == "Autoencoder":
                encoder = Autoencoder.from_state_dict(loaded_info["encoder"])
            elif encoder_type == "NaiveAutoEncoder":
                encoder = NaiveAutoEncoder.from_state_dict(loaded_info["encoder"])
            model.encoder = encoder.to(device)
        else:
            model.encoder = None
        model.vector_norm = loaded_info.get("vector_norm", False)
            
        model.external_dim_red_flag = loaded_info.get("external_dim_red_flag", False)
        #model.encoder = encoder.to(device)
        return model

    def to(self, device):
        if self.encoder is not None:
            self.encoder = self.encoder.to(device)
        return self
    
    def predict(self, 
                layer_info=None, 
                input_token_length=None, 
                candidate_tokens=None, 
                x=None,
                first_token=None):
        assert (layer_info is not None or x is not None), "Either layer_info or before_enc should be provided."
        if layer_info is not None:
            self.hidden_first = layer_info.cpu()[input_token_length]
        else:
            self.hidden_first = first_token
        self.hidden_first, _ = self.encoder.encode(self.hidden_first)
        latent, before_enc = collect_hidden_states(layer_info, input_token_length, candidate_tokens, self.encoder, before_enc=x)
        latent = latent.numpy().astype(np.float32)
        self.cache = before_enc.numpy().astype(np.float32)
        if self.sorting_code:
            latent = obtain_sorted_code(latent, self.encoder.activation.k)
        
        already_norm = False
        if self.vector_norm is True:
            already_norm = True
            latent = latent / np.linalg.norm(latent, axis=1, keepdims=True)
            
        if self.dim_red is not None:
            if self.external_dim_red_flag == True:
                latent = self.dim_red.forward(latent, norm=not already_norm).numpy()
            else:
                latent = self.dim_red.transform(latent)
        if self.model_type == "attn":
            y = self.clf.predict_proba(self.hidden_first, latent)
            pred_result = y[:, 1]
            rank_per_line = sorted(list(zip(pred_result, [i for i in range(len(pred_result))])), reverse=True)
            rank_per_line = [i[1] for i in rank_per_line]
        elif self.model_type == "ranker":
            rank_per_line, y = self.clf.predict_proba(latent)
        else:
            y = self.clf.predict_proba(latent)
            pred_result = y[:, 1]
            rank_per_line = sorted(list(zip(pred_result, [i for i in range(len(pred_result))])), reverse=True)
            rank_per_line = [i[1] for i in rank_per_line]
    
        return rank_per_line, y
    
    def predict_using_recorder(self, recorder, tokenized_info, input_token_length, candidate_tokens):
        with torch.inference_mode():
            _, record_dict = recorder.forward(tokenized_info)
            hidden_states = record_dict[list(recorder.layer_names.keys())[0]] # Only one layer
            hidden_states = hidden_states[0][0] # (token_num, hidden_size)
            recorder.clear_cache()
            rank_per_line, y = self.predict(hidden_states, input_token_length, candidate_tokens)
            #latent_activations, _ = collect_hidden_states(hidden_states, input_token_length, candidate_tokens, self.encoder).numpy()
            #y = self.clf.predict_proba(latent_activations)
            return rank_per_line, y, self.cache
    
    def evaluate(self, x, y, context=None):
        pred_profiler = []
        if self.model_type.lower() != "lstm":
            pred_label_recoreder = []
            for idx, item in enumerate(x):
                item = item.reshape(1, -1)
                if self.sorting_code:
                    item = obtain_sorted_code(item, self.encoder.activation.k)
                if self.vector_norm:
                    item = item / np.linalg.norm(item, axis=1, keepdims=True)
                if self.dim_red is not None:
                    if self.external_dim_red_flag == True:
                        item = self.dim_red.forward(item, norm = not self.vector_norm).numpy()
                    else:
                        item = self.dim_red.transform(item)
                if self.model_type.lower() == "attn":
                    hidden_first = context[idx]
                    pred = self.clf.predict_proba(hidden_first, item)
                else:
                    pred = self.clf.predict_proba(item)
                pred_result = pred[:, 1]
                pred_profiler.append(pred_result)
                pred_label = pred_result > 0.5
                pred_label_recoreder.append(pred_label)
        else:
            pred_label_recoreder = dict()
            key_list = list(x.keys())
            for key in key_list:
                item = x[key].reshape(1, -1)
                if self.sorting_code:
                    item = obtain_sorted_code(item, self.encoder.activation.k)
                if self.vector_norm:
                    item = item / np.linalg.norm(item, axis=1, keepdims=True)
                if self.dim_red is not None:
                    if self.external_dim_red_flag == True:
                        item = self.dim_red.forward(item, norm = not self.vector_norm).numpy()
                    else:
                        item = self.dim_red.transform(item)
                pred = self.clf.predict_proba(item)
                pred_result = pred[:, 1]
                pred_profiler.append(pred_result)
                pred_label = pred_result > 0.5
                pred_label_recoreder[key] = pred_label
            y = [y[key] for key in key_list]
            pred_label_recoreder = [pred_label_recoreder[key] for key in key_list]
        return sklearn.metrics.accuracy_score(y, pred_label_recoreder), pred_profiler
            

class LBLRegression():
    def __init__(self):
        self.fit_model_param = None
        self.clf = None
        self.attn_layer = None
        self.model_type = None
        self.cache = None
        self.hidden_first = None
        
    def fit(self, train_info, model_type, fit_model_param, attn_layer):
        self.fit_model_param = fit_model_param
        self.attn_layer = attn_layer
        self.model_type = model_type
        train_x = train_info["train_x"]
        train_y = train_info["train_y"]
        if self.model_type.lower() == "logistic":
            self.clf = LogisticRegression(**fit_model_param).fit(train_x, train_y)
        elif self.model_type.lower() == "svm":
            self.clf = svm.SVC(probability=True, **fit_model_param).fit(train_x, train_y)
        elif self.model_type.lower() == "mlp":
            self.clf = MLPClassifier(**fit_model_param).fit(train_x, train_y)
        elif self.model_type.lower() == "lstm":
            self.clf = LSTMPredictor(**fit_model_param).fit(train_x, train_y)
        
    def save(self, path):
        joblib.dump({"model": self.clf, "param": self.fit_model_param, "attn_layer": self.attn_layer, "model_type": self.model_type}, path)
    
    @classmethod
    def load(cls, path):
        loaded_info = joblib.load(path)
        model = cls()
        model.clf = loaded_info["model"]
        model.fit_model_param = loaded_info["param"]
        model.attn_layer = loaded_info["attn_layer"]
        model.model_type = loaded_info["model_type"]
        return model
    
    def predict(self, attn_snapshot=None, input_token_length=None, candidate_tokens=None, x=None):
        assert attn_snapshot is not None or x is not None, "Either attn_snapshot or before_enc should be provided."
        if x is None:
            al_result = collect_attention_map(attn_snapshot, self.attn_layer, input_token_length, candidate_tokens)
        self.cache = x
        x = np.stack(al_result).astype(np.float32)
        y = self.clf.predict_proba(x)

        pred_result = y[:, 1]
        rank_per_line = sorted(list(zip(pred_result, [i for i in range(len(pred_result))])), reverse=True)
        rank_per_line = [i[1] for i in rank_per_line]
        return rank_per_line, y
    
    def predict_using_recorder(self, recorder, tokenized_info, input_token_length, candidate_tokens):
        with torch.inference_mode():
            snapshot, hook_info = recorder.forward(tokenized_info)
            attn_snapshot = hook_info[list(recorder.layer_names.keys())[0]] # Only one layer
            recorder.clear_cache()

            rank_per_line, y = self.predict([attn_snapshot], input_token_length, candidate_tokens)
            
            return rank_per_line, y, self.cache

    def evaluate(self, x, y):
        if self.model_type.lower() != "lstm" or self.model_type.lower() != "attn":
            pred_label_recoreder = []
            for idx, item in enumerate(x):
                pred = self.clf.predict_proba(item.reshape(1, -1))
                pred_result = pred[:, 1]
                pred_label = pred_result > 0.5
                pred_label_recoreder.append(pred_label)
        else:
            pred_label_recoreder = dict()
            key_list = list(x.keys())
            for key in key_list:
                pred = self.clf.predict_proba(x[key].reshape(1, -1))
                pred_result = pred[:, 1]
                pred_label = pred_result > 0.5
                pred_label_recoreder[key] = pred_label
            y = [y[key] for key in key_list]
            pred_label_recoreder = [pred_label_recoreder[key] for key in key_list]
        return sklearn.metrics.accuracy_score(y, pred_label_recoreder), pred_label_recoreder

class LSTMPredictor(nn.Module):
    def __init__(self, 
                 input_dim=100, 
                 hidden_dim=128, 
                 num_layers=1, 
                 tagset_size=2, 
                 train_epoch=5,
                 lr=1e-4,
                 batch_size=4):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.lr = float(lr)
        #self.word2idx = {'<PAD>': 0, '<UNK>': 1}  # Start with special tokens
        self.tag2idx = {'<PAD>': -1}

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, tagset_size)  # Multiply by 2 for bidirectional

    def forward(self, hidden, lengths):
        # LSTM outputs
        packed_embeds = nn.utils.rnn.pack_padded_sequence(hidden, lengths, batch_first=True, enforce_sorted=False)
        packed_lstm_out, _ = self.lstm(packed_embeds)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_lstm_out, batch_first=True)
        # Fully connected layer
        fc_out = self.fc(lstm_out)
        # Apply log softmax for NLLLoss
        tag_scores = nn.functional.log_softmax(fc_out, dim=2)
        return tag_scores
    
    def get_collate_fn(self):
        def collate_fn(batch):
            sequences, tags = zip(*batch)
            lengths = [len(seq) for seq in sequences]
            padded_sequences = pad_sequence([convert_to_torch_tensor(seq).float() for seq in sequences], batch_first=True)
            padded_tags = pad_sequence([torch.tensor(seq) for seq in tags], batch_first=True, padding_value=self.tag2idx['<PAD>'])
            return padded_sequences, padded_tags, lengths
        return collate_fn

    def fit(self, train_x_dict, train_y_dict):
        local_dataset = DictDataset(train_x_dict, train_y_dict)
        train_loader = torch.utils.data.DataLoader(local_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.get_collate_fn())
        self.train()
        loss_function = nn.NLLLoss(ignore_index=self.tag2idx['<PAD>'])
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        num_epochs = self.train_epoch
        for epoch in range(num_epochs):
            total_loss = 0
            for embeddings, tags, lengths in train_loader:
                optimizer.zero_grad()
                tag_scores = self(embeddings.to(self.device), lengths)
                tag_scores = tag_scores.view(-1, self.tagset_size)
                tags = tags.view(-1)
                loss = loss_function(tag_scores, tags)
                loss.backward()
                total_loss += loss.item()
            logger.info((f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/(len(train_loader)*self.batch_size)}"))
        
        self.eval()
        return self
    
    def predict_proba(self, embeddings):
        self.eval()
        length = [len(embeddings)]
        embeddings_tensor = convert_to_torch_tensor(embeddings).float().unsqueeze(0)
        with torch.inference_mode():
            tag_scores = self(convert_to_torch_tensor(embeddings_tensor).float().to(self.device), length)
            #_, predicted_tags = torch.max(tag_scores, dim=2)
            #predicted_tags = predicted_tags.squeeze(0).tolist()
            probabilities = torch.exp(tag_scores)
        return probabilities.cpu().squeeze(0)

    @property
    def device(self):
        return next(self.parameters()).device
        

#def obtain_sorted_code(input_tensor, topk):
    # Sort the tensor and get the sorted indices
#    if isinstance(input_tensor, np.array):
#        input_tensor = torch.from_numpy(input_tensor)
#    sorted_value, sorted_indices = torch.sort(input_tensor, dim=-1, descending=True)
#    return  sorted_indices[:, :topk] / input_tensor.shape[-1], sorted_value[:, :topk]

def obtain_sorted_code(input_tensor, topk):
    # Sort the tensor and get the sorted indices
    input_tensor = input_tensor
    sorted_indices = np.argsort(input_tensor)
    sorted_indices = sorted_indices[..., ::-1][:, :topk]

    sorted_array = np.take_along_axis(input_tensor, sorted_indices, axis=-1)
    normalized_index = sorted_indices / input_tensor.shape[-1]
    
    #return normalized_index#, sorted_array
    return np.concatenate([normalized_index, sorted_array], axis=-1)

class DictDataset(Dataset):
    def __init__(self, dict_x, dict_y, dict_z=None):
        self.dict_x = dict_x
        self.dict_y = dict_y
        self.dict_z = dict_z
        self.keys = sorted(self.dict_x.keys())
    
    def __getitem__(self, index):
        return_item = [self.dict_x[self.keys[index]], self.dict_y[self.keys[index]]]
        if self.dict_z is not None:
            return_item.append(self.dict_z[self.keys[index]])
        return return_item
    
    def __len__(self):
        return len(self.keys)