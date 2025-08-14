# PtTrust 

This is the related code relevant to the work **Risk Assessment Framework for Code LLMs via Leveraging Internal States**

## Dependency :lock:

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


## File Sturecture :construction:

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

## Risk Assessment Method Workflow :art:

The overall structure of PtTrust is through LLM Evaluation --> hidden state extraction --> sae training --> semantic binding --> evaluation.

### LLM Inference :racehorse:

``code_generation.py``,  ``code_repair.py`` and ``code_translation.py`` are used to evaluate LLMs' performance on different tasks. All of them can be configured through yaml files. Please refer to examples under `example_model_config` folder (e.g., files in `llm_evaluation` )

### Hidden State Extraction :wrench:

Hidden state extraction is done through ``./method/collect_hidden_aug.py``. It is also configured through YAML files. Please refer to examples in `./example_model_config/extraction_hidden_state.yaml`. Please be aware that this file is dataset-specific. For example, for the Leetcode dataset, we have a parameter called `mutation_prop` which indicates the proportion of code lines that will be mutated.

The most important extraction utilities are under ``./method/extract``. In ``extract_util.py``, we have a class called ``TransHookRecorder``, which parses the target layer by string and extracts the states given  that layer. Please refer to ``method/collect_hidden_aug.py`` for the specific usage (i.e., first define which layer to extract, then call ``recorder.forward`` with an LLM, and finally clear the cache of the recorder through ``recorder.clear_cache()``). 

We also define some IO utilities in `method/extract/naive_store.py`. They directly store the extracted data to disk instead of memory. But there is still a lot of room for improvement. 

`VariedKeyTensorStore` is indexed through key, while `NaiveTensorStore` is a List.

### SAE Training :truck:

It is done in ``./method/sae_training.py``. The example configuration file is in ``./example_model_config/sae_training.yaml``. In ``llm_config`` we have related parameters to load target LLMs (e.g., model_name, quantization level.). Under ``task_config`` we have related parameters to train the SAE. We also use Wandb to monitor the training process (``wandb_info``). 

### Semantic Binding :rocket:

The binding process of PtTrust is done in ``./method/semantic_binding_rank.py``. The example configuration file is ``./example_model_config/PtTrust_semantic_binding.yaml``. Please specify the path of your trained encoders in this file; otherwise, it will fall back to the normal model (Probing classifiers). Specify ``agg`` parameters for training to enable classification mode (i.e., entire code snippet correctness prediction). 

Labels are necessary in this stage. For prompts related to automatic labeling, please refer to: https://github.com/YuhengHuang42/trust_codeLLM/blob/main/utility/openai_utils.py

Notice that we include a few-shot example in the prompt to obtain structural output for the following automatic extraction.

### Method Evaluation :rotating_light:

Evaluation is done in ``./method/evaluate_binding.py``. Please refer to  ``./example_model_config/eval_PtTrust_editeval.yaml`` for details. According to the difference in the semantic binding stage, the evaluation script will automatically adapt to different settings.

When performing evaluations, we rely on a specific label file referred to as `important_label_path`. The data id indexes this file and contains a mapping betwen id -> which lines are incorrect. Related labels can be obtained through ``dataset_utils.find_buggy_positions`` by providing, for example, the identified incorrect code lines returned by the OpenAI API. For diff-based labels, we used `utils.get_changes_with_line_numbers` for automatic labeling.

## Example Data

We uploaded example data at: https://drive.google.com/drive/folders/1BVHheTeZqVGdEq8Jn-e6mJOEf0uKnYTx?usp=drive_link

Here, `openai_label` contains the returned labeling from the GPT-4o API.

`xxx_inference` contains the inference result of different evaluation datasets using shelve. We also have `xxx_line_error.json` files in these directories as examples for `important_label`.

## Credit :package:

- [Code Lingua](https://github.com/codetlingua/codetlingua)

- [Defects4j](https://github.com/rjust/defects4j)

- [EditEval](https://github.com/qishenghu/InstructCoder)

- [LLM_repair](https://zenodo.org/records/7592886)

- [Transformer Lens](https://github.com/TransformerLensOrg/TransformerLens)

(There might be some missing here, but there will be comments besides the code if I can find the source. )





