import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn import svm
import torch
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE

from method.sae_model import Autoencoder

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
    
def collect_hidden_states(hidden_map, input_token_length, output_seg, encoder):
    """
    Collect hidden states as features for hallucination detection.
    ---
    Args:
        hidden_map: Tuple[Tensor]. Each layer's hidden states, which has the shape [1, token_num, hidden_size].
        output_seg: List[List]: List of segment tokens. It should be str_output version
    """
    hidden_all = list()
    for seg in output_seg:
        real_seg = [i + input_token_length  for i in seg]
        line_split_token = max(real_seg)
        hidden_seg = hidden_map[line_split_token] # (token_num, hidden_size)
        hidden_all.append(hidden_seg)
    hidden_all = torch.stack(hidden_all).to(encoder.device)
    with torch.inference_mode():
        latent_activations, info = encoder.encode(hidden_all)
    return latent_activations.cpu()
    
            
class EncoderClassifier():
    def __init__(self):
        self.fit_model_param = None
        self.clf = None
        self.model_type = None
        
    def fit(self, 
            train_x, 
            train_y, 
            model_type, 
            fit_model_param,
            encoder, 
            ):
        self.fit_model_param = fit_model_param
        self.dim_red = None
        if "dimension_reduction" in self.fit_model_param:
            dim_red_param = self.fit_model_param.pop("dimension_reduction")
            if "svd" in dim_red_param:
                self.dim_red = TruncatedSVD(**dim_red_param["svd"])
                train_x = self.dim_red.fit_transform(train_x)
        if "balanced" in self.fit_model_param:
            balanced = self.fit_model_param.pop("balanced")
            if balanced:
                smote = SMOTE()
                train_x, train_y = smote.fit_resample(train_x, train_y)
        self.model_type = model_type
        self.encoder = encoder
        if self.model_type.lower() == "logistic":
            self.clf = LogisticRegression(**fit_model_param).fit(train_x, train_y)
        elif self.model_type.lower() == "svm":
            self.clf = svm.SVC(probability=True, **fit_model_param).fit(train_x, train_y)
        elif self.model_type.lower() == "mlp":
            self.clf = MLPClassifier(**fit_model_param).fit(train_x, train_y)
        
    def save(self, path):
        encoder_device = self.encoder.device
        joblib.dump({"model": self.clf,
                     "dim_red": self.dim_red, 
                     "param": self.fit_model_param, 
                     "model_type": self.model_type,
                     "device": encoder_device,
                     "encoder": self.encoder.cpu().state_dict()
                     }, 
                    path
                    )
    
    @classmethod
    def load(cls, path):
        loaded_info = joblib.load(path)
        model = cls()
        model.clf = loaded_info["model"]
        model.fit_model_param = loaded_info["param"]
        model.model_type = loaded_info["model_type"]
        model.dim_red = loaded_info["dim_red"]
        device = loaded_info["device"]
        encoder = Autoencoder.from_state_dict(loaded_info["encoder"])
        model.encoder = encoder.to(device)
        return model

    def to(self, device):
        self.encoder = self.encoder.to(device)
        return self
    
    def predict(self, layer_info, input_token_length, candidate_tokens):
        latent = collect_hidden_states(layer_info, input_token_length, candidate_tokens, self.encoder).numpy()
        if self.dim_red is not None:
            latent = self.dim_red.transform(latent)
        y = self.clf.predict_proba(latent)
        return y
    
    def predict_using_recorder(self, recorder, tokenized_info, input_token_length, candidate_tokens):
        with torch.inference_mode():
            _, record_dict = recorder.forward(tokenized_info)
            hidden_states = record_dict[list(recorder.layer_names.keys())[0]] # Only one layer
            hidden_states = hidden_states[0][0] # (token_num, hidden_size)
            recorder.clear_cache()
            y = self.predict(hidden_states, input_token_length, candidate_tokens)
            #latent_activations = collect_hidden_states(hidden_states, input_token_length, candidate_tokens, self.encoder).numpy()
            #y = self.clf.predict_proba(latent_activations)
            return y

class LBLRegression():
    def __init__(self):
        self.fit_model_param = None
        self.clf = None
        self.attn_layer = None
        self.model_type = None
        
    def fit(self, train_x, train_y, model_type, fit_model_param, attn_layer):
        self.fit_model_param = fit_model_param
        self.attn_layer = attn_layer
        self.model_type = model_type
        if self.model_type.lower() == "logistic":
            self.clf = LogisticRegression(**fit_model_param).fit(train_x, train_y)
        elif self.model_type.lower() == "svm":
            self.clf = svm.SVC(probability=True, **fit_model_param).fit(train_x, train_y)
        elif self.model_type.lower() == "mlp":
            self.clf = MLPClassifier(**fit_model_param).fit(train_x, train_y)
        
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
    
    def predict(self, attn_snapshot, input_token_length, candidate_tokens):
        al_result = collect_attention_map(attn_snapshot, self.attn_layer, input_token_length, candidate_tokens)
        x = np.stack(al_result)
        y = self.clf.predict_proba(x)
        return y
    
    def predict_using_recorder(self, recorder, tokenized_info, input_token_length, candidate_tokens):
        with torch.inference_mode():
            snapshot, hook_info = recorder.forward(tokenized_info)
            attn_snapshot = hook_info[list(recorder.layer_names.keys())[0]] # Only one layer
            recorder.clear_cache()

            pred_result = self.predict([attn_snapshot], input_token_length, candidate_tokens)
            
            return pred_result

