import torch
import copy
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.cache_utils import DynamicCache

def parse_layer_name(layer_name, base_module):
    """
    Return pytorch model layer based on layer_name and the model.
    One can register forward hook easily by using the returned module.
    ---
    Input:
        layer_name: string. Use "." to indicates the sub_module level.
            For example, features.denseblock1.denselayer1.conv in densenet.
        base_module: torch.nn.modules. DNN model. If the model is in DataParallel,
            pass model.module.
    Return:
        target_module: torch.nn.modules or None(when not found).
    """
    target_name_list = layer_name.split(".")
    target_name = target_name_list[0]
    for name, module in base_module._modules.items():
        if name == target_name:
            if len(target_name_list) == 1:
                return module
            else:
                next_level_layer = target_name_list[1:]
                next_level_layer = ".".join(next_level_layer)
                return parse_layer_name(next_level_layer, module)
    return None


class TransHookRecorder:
    """This is the hook for Transformer model.
    It is used to record the value of hidden states.
    """
    def __init__(self, layer_names: dict, model, mode="attention"):
        '''
        layer_names: dict. The key is the layer name, and the value is the configuration.
            Example: {"model.layers.47.self_attn": {"output_attentions": True}}
            Example: {'model.layers.35': {"return_first": True}}
        '''
        self.parameter_recorder = dict()
        self.recorder = dict()
        self.layer_names = layer_names
        self.mode = mode
        assert self.mode in ["attention", "plain"]
        if isinstance(model, torch.nn.DataParallel):
            self.model = model.module
        else:
            self.model = model
        self.handlers = list()
        #self.record_mode = record_mode
    
    def _register_hooker(self, name):
        self.recorder[name] = list()
        return_first = False
        if "return_first" in self.layer_names[name]:
            return_first = self.layer_names[name]["return_first"]
        def named_hooker(module, input, output):
            if return_first:
                output = output[0].cpu()
            self.recorder[name].append(output)
        return named_hooker
    
    def _register_pre_hooker(self, name):
        self.parameter_recorder[name] = dict()
        def named_hooker(module, args, kwargs):
            if len(self.parameter_recorder[name]) == 0:
                for key in kwargs:
                    self.parameter_recorder[name][key] = kwargs[key]
                #self.parameter_recorder[name]['past_key_value'] = copy.deepcopy(kwargs["past_key_value"])
                self.parameter_recorder[name]['past_key_value'] = {
                    "key_cache": copy.deepcopy(kwargs["past_key_value"].key_cache),
                    "value_cache": copy.deepcopy(kwargs["past_key_value"].value_cache),
                    "cache_type": type(kwargs["past_key_value"])
                }
        return named_hooker
    
    def register_hookers(self):
        if self.mode == "attention":
            for l_name in self.layer_names:
                module = parse_layer_name(l_name, self.model)
                if module == None:
                    raise Exception("Layer not found")
                handler_pre = module.register_forward_pre_hook(self._register_pre_hooker(l_name), with_kwargs=True)
                self.handlers.append(handler_pre)
                #handler_post = module.register_forward_hook(self._register_hooker(l_name), with_kwargs=True)
                #self.handlers.append(handler_post)
        elif self.mode == "plain":
            for l_name in self.layer_names:
                module = parse_layer_name(l_name, self.model)
                if module == None:
                    raise Exception("Layer not found")
                handler = module.register_forward_hook(self._register_hooker(l_name))
                self.handlers.append(handler)
            
    
    def forward(self, input_dict):
        if self.mode == "attention":
            return self.rerun_forward(input_dict)
        else:
            return self.plain_forward(input_dict)

    
    
    def plain_forward(self, input_dict):
        """
        
        """
        self.remove_handlers()
        self.register_hookers()
        with torch.inference_mode():
            snapshot = self.model.forward(**input_dict)
        self.remove_handlers()
        return snapshot, self.recorder
    
    def rerun_forward(self, input_dict):
        """
        Extract the internal states of the model by first dry running the model, using hooks to 
        record the intermediate input configurations, and then running the specific module again to get the profiled results.
        ---
        Args:
            input_dict: dict. The input dictionary for the model.
        Output:
            snapshot: dict. The output of the model.
            recorder: dict. The recorded intermediate states. layer_name --> output.
        """
        self.remove_handlers()
        self.register_hookers()
        with torch.inference_mode():
            snapshot = self.model.forward(**input_dict)
        self.remove_handlers()
        for l_name in self.layer_names:
            module = parse_layer_name(l_name, self.model)
            cur_input_para = self.parameter_recorder[l_name]
            restore_para = cur_input_para['past_key_value']
            temp_cache = restore_para['cache_type']()
            temp_cache.value_cache = restore_para['value_cache']
            temp_cache.key_cache = restore_para['key_cache']
            cur_input_para["past_key_value"] = temp_cache
            if "output_attentions" in self.layer_names[l_name]:
                cur_input_para["output_attentions"] = self.layer_names[l_name]["output_attentions"]
            cur_output = module(**cur_input_para)
            if cur_input_para["output_attentions"]:
                self.recorder[l_name] = cur_output[1].cpu()
            del cur_input_para
            self.parameter_recorder[l_name] = dict()
        return snapshot, self.recorder
        
    
    def get_result(self):
        return self.recorder
    
    def clear_cache(self):
        for key in self.recorder:
            self.recorder[key] = list()
        
    def remove_handlers(self):
        for i in self.handlers:
            i.remove()
        self.handlers.clear()
        
    def __del__(self):
        self.remove_handlers()
        
# This class is already introduced in v4.45.1: https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py
# But not v4.42.3. So we define it here for compatibility.
class OffloadedCache(DynamicCache):
    """
    A drop-in replacement for DynamicCache that conserves GPU memory at the expense of more CPU memory.
    Useful for generating from models with very long context.

    In addition to the default CUDA stream, where all forward() computations happen,
    this class uses another stream, the prefetch stream, which it creates itself.
    Since scheduling of operations on separate streams happens independently, this class uses
    the prefetch stream to asynchronously prefetch the KV cache of layer k+1 when layer k is executing.
    The movement of the layer k-1 cache to the CPU is handled by the default stream as a simple way to
    ensure the eviction is scheduled after all computations on that cache are finished.
    """

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("OffloadedCache can only be used with a GPU")
        super().__init__()
        self.original_device = []
        self.prefetch_stream = torch.cuda.Stream()
        self.beam_idx = None  # used to delay beam search operations

    def prefetch_layer(self, layer_idx: int):
        "Starts prefetching the next layer cache"
        if layer_idx < len(self):
            with torch.cuda.stream(self.prefetch_stream):
                # Prefetch next layer tensors to GPU
                device = self.original_device[layer_idx]
                self.key_cache[layer_idx] = self.key_cache[layer_idx].to(device, non_blocking=True)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].to(device, non_blocking=True)

    def evict_previous_layer(self, layer_idx: int):
        "Moves the previous layer cache to the CPU"
        if len(self) > 2:
            # We do it on the default stream so it occurs after all earlier computations on these tensors are done
            prev_layer_idx = (layer_idx - 1) % len(self)
            self.key_cache[prev_layer_idx] = self.key_cache[prev_layer_idx].to("cpu", non_blocking=True)
            self.value_cache[prev_layer_idx] = self.value_cache[prev_layer_idx].to("cpu", non_blocking=True)

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        "Gets the cache for this layer to the device. Prefetches the next and evicts the previous layer."
        if layer_idx < len(self):
            # Evict the previous layer if necessary
            torch.cuda.current_stream().synchronize()
            self.evict_previous_layer(layer_idx)
            # Load current layer cache to its original device if not already there
            original_device = self.original_device[layer_idx]
            self.prefetch_stream.synchronize()
            key_tensor = self.key_cache[layer_idx]
            value_tensor = self.value_cache[layer_idx]
            # Now deal with beam search ops which were delayed
            if self.beam_idx is not None:
                self.beam_idx = self.beam_idx.to(original_device)
                key_tensor = key_tensor.index_select(0, self.beam_idx)
                value_tensor = value_tensor.index_select(0, self.beam_idx)
            # Prefetch the next layer
            self.prefetch_layer((layer_idx + 1) % len(self))
            return (key_tensor, value_tensor)
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Saves the beam indices and reorders the cache when the tensor is back to its device."""
        # We delay this operation until the tensors are back to their original
        # device because performing torch.index_select on the CPU is very slow
        del self.beam_idx
        self.beam_idx = beam_idx.clone()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `OffloadedCache`.
        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.original_device.append(key_states.device)
            self.evict_previous_layer(layer_idx)
        else:
            key_tensor, value_tensor = self[layer_idx]
            self.key_cache[layer_idx] = torch.cat([key_tensor, key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([value_tensor, value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    # According to https://docs.python.org/3/library/exceptions.html#NotImplementedError
    # if a method is not supposed to be supported in a subclass we should set it to None
    from_legacy_cache = None

    to_legacy_cache = None