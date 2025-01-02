# Reference: https://github.com/bigcode-project/bigcode-evaluation-harness/blob/3c8c685b6c162f034e7e0215b19cb75917ba6672/lm_eval/tasks/humaneval.py#L88

"""Evaluating Large Language Models Trained on Code
https://arxiv.org/abs/2107.03374

The HumanEval dataset released by OpenAI includes 164 programming problems with a function signature,
docstring, body, and several unit tests. 
They were handwritten to ensure not to be included in the training set of code generation models.

Homepage: https://github.com/openai/human-eval
"""

import re
from evaluate import load


_CITATION = """
@misc{chen2021evaluating,
      title={Evaluating Large Language Models Trained on Code},
      author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
      year={2021},
      eprint={2107.03374},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""

from abc import ABC, abstractmethod
from warnings import warn
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Keyword, Name

from datasets import load_dataset
from .dataset_utils import CodeDataset

class Task(ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    DATASET_PATH: str = None

    # The name of a subset within `DATASET_PATH`.
    DATASET_NAME: str = None

    def __init__(self, stop_words=None, requires_execution=True):
        """
        :param stop_words: list
            list of stop words if the generation uses a stopping criteria during generation
        :param requires_execution: bool
            wheter the task requires code execution during evaluation or not
        """
        self.stop_words = stop_words
        self.requires_execution = requires_execution
        try:
            self.dataset = load_dataset(path=self.DATASET_PATH, name=self.DATASET_NAME)
        except:
            warn(
                "This task will use a locally downloaded dataset, not from the HF hub."
            )

    @abstractmethod
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return []

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    @abstractmethod
    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass

    @abstractmethod
    def get_reference(self, doc):
        """Builds the reference solution for the doc.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass

    @abstractmethod
    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        pass

    @abstractmethod
    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        pass
    
class HumanEval(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "openai_humaneval"

    def __init__(self):
        super().__init__(
            stop_words=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"],
            requires_execution=True,
        )
        self.code_metric = load("code_eval")

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        return doc["prompt"].strip()

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        return "\n" + test_func + "\n" + entry_point

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return (decoded_string[:min_stop_index], min_stop_index)

    def postprocess_generation(self, generation, idx, real_prompt=None):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        prompt = self.get_prompt(self.dataset["test"][idx])
        start_pos = len(prompt) if real_prompt is None else len(real_prompt)
        generation = generation[start_pos :]
        cleaned_code, min_stop_index = self._stop_at_stop_token(generation, self.stop_words)
        return (prompt + cleaned_code, min_stop_index)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        #code_metric = load("code_eval")
        results, _ = self.code_metric.compute(
            references=references,
            predictions=generations,
        )
        return results


def extract_assertions_from_humaneval(code_str):
    # Use regex to extract all assertions
    tokens = lex(code_str, PythonLexer())
    for token, value in tokens:
        # Include loop and other logic in constructing test cases.
        if token in [Keyword] and value in ['for', 'while']:
            return [code_str]
    func_name_pattern = r"check\((\w+)\)"
    matches = re.findall(func_name_pattern, code_str)
    func_name = matches[-1]
    
    assertion_pattern = r"(assert .*\n)"
    assertions = re.findall(assertion_pattern, code_str)
    
    # Replace the "candidate" with the given function name
    modified_assertions = [assertion.replace("candidate", func_name) for assertion in assertions]
    
    return modified_assertions
    
class HumanEvalDataset(CodeDataset):
    def __init__(self):
        self.task = HumanEval()
        self.problems = self.task.get_dataset()
    
    def __len__(self):
        return len(self.problems)

    def __getitem__(self, i):
        return self.problems[i]
    
    def get_prompt(self, problem_id):
        prompt = self.task.get_prompt(self.problems[problem_id])
        return prompt + "\n    "
    
    def check_result(self, generate_code, problem_id: int, completion_id=1, output_error_case=False):
        test_cases = self.task.get_reference(self.problems[problem_id])
        result = self.task.process_results([[generate_code]], [test_cases])
        return result
    
    def check_result_in_detail(self, generated_code, problem_id: int):
        test_cases = self.task.get_reference(self.problems[problem_id])
        test_cases = extract_assertions_from_humaneval(test_cases)
        
        results, detail = self.task.code_metric.compute(
            references=test_cases,
            predictions=[[generated_code] for i in range(len(test_cases))],
            num_workers=1,
            timeout=3 # https://huggingface.co/spaces/evaluate-metric/code_eval
        )
        
        failed_test_case = []
        for key in detail:
            if detail[key][0][1]['passed'] == True:
                continue
            else:
                failed_test_case.append(test_cases[key])
        
        return results, failed_test_case, detail
        #return score + results["pass@1"] / 2
        #return result
    
    def postprocess(self, generate_code, problem_id, real_prompt=None):
        code, min_stop_index = self.task.postprocess_generation(generate_code, problem_id, real_prompt)
        return code, min_stop_index