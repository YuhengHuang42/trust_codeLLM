from datasets import load_dataset
from evaluate import load
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

from .dataset_utils import CodeDataset
class HumanEvalPackRepair(CodeDataset):
    def __init__(self,
                 dataset_name: str="bigcode/humanevalpack",
                 split_token="\n",
                ):
        self.problems = load_dataset(dataset_name, "python")["test"]
        self.split_token = split_token
        code_metric = load("code_eval")
        self.code_metric = code_metric
    
    def __getitem__(self, index):
        item = self.problems[index]
        return item

    def __len__(self):
        return len(self.problems)
    
    def get_buggy_code(self, index):
        return self.problems[index]['buggy_solution']

    def get_fix_code(self, idx):
        return self.problems[idx]['canonical_solution']
    
    def get_prompt(self, index):
        entry_point = self.problems[index]['entry_point']
        declaration = self.problems[index]['declaration']
        buggy_code = self.problems[index]['buggy_solution']
        example_test = self.problems[index]['example_test']
        signature = self.problems[index]['signature']
        correct = self.problems[index]['canonical_solution']
        instruction = f"The above code is wrong and failed the test cases. Fix bugs in {entry_point}:"

        prompt = f"{declaration}\n{buggy_code}\n{example_test}\n{instruction}\n\ndef {signature}:\n"
        return prompt, correct
        
    def check_result(self, generate_code, problem_id: int, completion_id=1, output_error_case=False, input_full_code=False):
        if not input_full_code:
            full_code = self.problems[problem_id]['declaration'] + generate_code
        else:
            full_code = generate_code
        results, _ = self.code_metric.compute(
            references = [self.problems[problem_id]['test']],
            predictions = [[full_code]]
        )
        if abs(results["pass@1"] - 1.0) < 1e-5:
            return "correct"
        else:
            return "buggy"