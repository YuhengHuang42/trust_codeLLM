# PtTrust

This is the related code relevant to the work **Risk Assessment Framework for Code LLMs via Leveraging Internal States**

## Dependency

For related Python dependencies, please refer to ``requirements.txt``. In addition, the code has been tested on:

Pytorch==2.4.1+cu121

transformers==4.42.3

```
sudo apt install subversion
```

We also rely on several external packages for dataset evaluation, this includes:

- [Code Lingua](https://github.com/codetlingua/codetlingua)

- [Defects4j](https://github.com/rjust/defects4j)

- [EditEval](https://github.com/qishenghu/InstructCoder)

For defects4j, after you install it, you can set the JAVA_PATH and DEFECTS4J_PATH in related repair evaluation configuration (.yaml) files, such as:

```
system_setting:
  JAVA_PATH: "/usr/lib/jvm/java-8-openjdk-amd64"
  DEFECTS4J_PATH: "/home/user/defects4j/framework/bin"
```

We also directly re-use the data from [LLM_repair](https://zenodo.org/records/7592886) (paper: [Automated Program Repair in the Era of Large Pre-trained Language Models](https://lingming.cs.illinois.edu/publications/icse2023a.pdf)). After you download it, specify the path in the yaml file with:

```
task_config:
  repair_data_path: "LLM_repair/Defects4j/single_function_repair.json"
  repair_loc_folder: "LLM_repair/Defects4j/location/"
```


## File Sturecture

```
├── code_completion.py
├── code_generation.py
├── code_repair.py
├── codetlingua
├── code_translation.py
├── InstructCoder # EditEval
├── LLM_repair # LLM_repair for Defects4j
├── method
├── example_model_config
├── README.md
├── utility
```

For experiments in this study, we use Yaml files to configure specific settings. We sometimes also use CLI configurations, which, in principle, have higher priority than settings in Yaml files.

### Evaluation

``code_generation.py``,  ``code_repair.py`` and ``code_translation.py`` are used to evaluate LLMs' performance on different tasks. All of them can be configured through yaml files. Please refer to examples under `example_model_config` folder (e.g., files in `llm_evaluation` )

### Method Workflow

The overall structure of PtTrust is through hidden state extraction --> sae training --> semantic binding --> evaluation.

#### Hidden State Extraction

Hidden state extraction is done through ``./method/collect_hidden_aug.py``. It is also configured through yaml files. Please refer to examples in `./example_model_config/extraction_hidden_state.yaml`. Please be aware this file is dataset specific. For example, for Leetcode dataset, we have a parameter called `mutation_prop` which indicates the proportion of code lines will be mutated.

The most important extraction utilities are under ``./method/extract``. In ``extract_util.py``, we have a class called ``TransHookRecorder``, which parses the target layer by string and extract the states given  that layer. Please refer to ``method/collect_hidden_aug.py`` for the specific usage (i.e., first define which layer to extract, then call ``recorder.forward`` with an LLM, and finally clear the cache of the recorder through ``recorder.clear_cache()``). 

We also define some IO utilities in `method/extract/naive_store.py`. They directly store the extracted data to disk instead of memory. But there is still a lot of room for improvement. 

`VariedKeyTensorStore` is index throguh key, while `NaiveTensorStore` is List.


#### SAE Training

It is done in ``./method/sae_training.py``. The example configuration file is in ``./example_model_config/sae_training.yaml``. In ``llm_config`` we have related parameters to load target LLMs (e.g., model_name, quantization level.). Under ``task_config`` we have related parameters to train the SAE. We also use Wandb to monitor the training process (``wandb_info``). 

#### Semantic Binding 

The binding process of PtTrust is done in ``./method/semantic_binding_rank.py``. The example configuration file is ``./example_model_config/PtTrust_semantic_binding.yaml``. Please specify the path of your trained encoders in this file otherwise it will fallback to normal model (Probing classifiers). Specify ``agg`` parameters for training to enable classification mode (i.e., entire code snippet correctness prediction). 

#### Evaluation

Evaluation is done in ``./method/evaluate_binding.py``. Please refer to  ``./example_model_config/eval_PtTrust_editeval.yaml`` for details. According to the difference in the semantic binding stage, the evaluation script will automatically adapt to different settings.


### Credit

- [Code Lingua](https://github.com/codetlingua/codetlingua)

- [Defects4j](https://github.com/rjust/defects4j)

- [EditEval](https://github.com/qishenghu/InstructCoder)

- [LLM_repair](https://zenodo.org/records/7592886)

- [Transformer Lens](https://github.com/TransformerLensOrg/TransformerLens)

(There might be some missing here, but there will be comments besides the code if I can find the source. )





