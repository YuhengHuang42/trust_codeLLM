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

def flat_data_dict(data_dict_x, data_dict_y):
    x = dict()
    y = dict()
    for dataset in data_dict_x:
        for key in data_dict_x[dataset]:
            x[f"{dataset}_{key}"] = data_dict_x[dataset][key]  
            y[f"{dataset}_{key}"] = data_dict_y[dataset][key]
    return x, y

def convert_to_torch_tensor(array):
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    elif isinstance(array, torch.Tensor):
        return array
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")
        
class EncoderClassifier():
    def __init__(self):
        self.fit_model_param = None
        self.clf = None
        self.model_type = None
        self.cache = None
        
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
        
    def save(self, path):
        if self.encoder is not None:
            encoder_device = self.encoder.device
            encoder_state_dict = self.encoder.cpu().state_dict()
        else:
            encoder_device = None
            encoder_state_dict = None
        torch.save({"model": self.clf,
                     "dim_red": self.dim_red, 
                     "param": self.fit_model_param, 
                     "model_type": self.model_type,
                     "device": encoder_device,
                     "encoder": encoder_state_dict,
                     "sorting_code": self.sorting_code
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
        if loaded_info["encoder"] is not None:
            encoder = Autoencoder.from_state_dict(loaded_info["encoder"])
            model.encoder = encoder.to(device)
        else:
            model.encoder = None
        #model.encoder = encoder.to(device)
        return model

    def to(self, device):
        if self.encoder is not None:
            self.encoder = self.encoder.to(device)
        return self
    
    def predict(self, layer_info=None, input_token_length=None, candidate_tokens=None, x=None):
        assert (layer_info is not None or x is not None), "Either layer_info or before_enc should be provided."
        latent, before_enc = collect_hidden_states(layer_info, input_token_length, candidate_tokens, self.encoder, before_enc=x)
        latent = latent.numpy().astype(np.float32)
        self.cache = before_enc.numpy().astype(np.float32)
        if self.dim_red is not None:
            latent = self.dim_red.transform(latent)
        if self.sorting_code:
            latent = obtain_sorted_code(latent, self.encoder.activation.k)
        y = self.clf.predict_proba(latent)
        return y
    
    def predict_using_recorder(self, recorder, tokenized_info, input_token_length, candidate_tokens):
        with torch.inference_mode():
            _, record_dict = recorder.forward(tokenized_info)
            hidden_states = record_dict[list(recorder.layer_names.keys())[0]] # Only one layer
            hidden_states = hidden_states[0][0] # (token_num, hidden_size)
            recorder.clear_cache()
            y = self.predict(hidden_states, input_token_length, candidate_tokens)
            #latent_activations, _ = collect_hidden_states(hidden_states, input_token_length, candidate_tokens, self.encoder).numpy()
            #y = self.clf.predict_proba(latent_activations)
            return y, self.cache
    
    def evaluate(self, x, y):
        pred_profiler = []
        if self.model_type.lower() != "lstm":
            pred_label_recoreder = []
            for idx, item in enumerate(x):
                item = item.reshape(1, -1)
                if self.sorting_code:
                    item = obtain_sorted_code(item, self.encoder.activation.k)
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
            x = np.stack(al_result).astype(np.float32)
        self.cache = x
        y = self.clf.predict_proba(x)
        return y
    
    def predict_using_recorder(self, recorder, tokenized_info, input_token_length, candidate_tokens):
        with torch.inference_mode():
            snapshot, hook_info = recorder.forward(tokenized_info)
            attn_snapshot = hook_info[list(recorder.layer_names.keys())[0]] # Only one layer
            recorder.clear_cache()

            pred_result = self.predict([attn_snapshot], input_token_length, candidate_tokens)
            
            return pred_result, self.cache

    def evaluate(self, x, y):
        if self.model_type.lower() != "lstm":
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
            tag_scores = nn.functional.log_softmax(tag_scores, dim=2)
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
    def __init__(self, dict_x, dict_y):
        self.dict_x = dict_x
        self.dict_y = dict_y
        self.keys = sorted(self.dict_x.keys())
    
    def __getitem__(self, index):
        return [self.dict_x[self.keys[index]], self.dict_y[self.keys[index]]]
    
    def __len__(self):
        return len(self.keys)