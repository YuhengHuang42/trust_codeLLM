
from .dataset_utils import CodeDataset
from datasets import load_dataset
from codetlingua.tools.utils import check_correctness, get_problem, untrusted_check

class CodetlinguaDataset(CodeDataset):
    def __init__(self, split, source_lang, target_lang):
        self.split = split
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.problems = CodetlinguaDataset.load_codetlingua(split, source_lang)

    @staticmethod
    def load_codetlingua(split, source_lang):
        if split == "iidai/codenet":
            problems = load_dataset("iidai/codenet")['train']
            problems = [p for p in problems if p['language'] == source_lang]
        elif split == "iidai/avatar":
            problems = load_dataset("iidai/avatar")['train']
            problems = [p for p in problems if p['language'] == source_lang]
        return problems
    
    @staticmethod
    def match_translation_code(dataset_a, dataset_b):
        assert len(dataset_a) == len(dataset_b)
        result = dict()
        for item in dataset_a:
            data_id = item["id"]
            result[data_id] = {
                dataset_a.source_lang: {
                    "code": item["code"],
                    "test_IO": item["test_IO"],
                }
            }
        
        for item in dataset_b:
            data_id = item["id"]
            if data_id in result:
                result[data_id][dataset_b.source_lang] = {
                    "code": item["code"],
                    "test_IO": item["test_IO"],
                }
            else:
                result[data_id] = {
                    dataset_b.source_lang: {
                        "code": item["code"],
                        "test_IO": item["test_IO"],
                    }
                }
        return result
    
    def check_result(self, generate_code, problem_id, completion_id=1, output_error_case=False):
        result = untrusted_check(
            self.problems[problem_id],
            generate_code,
            self.target_lang,
            completion_id,
            output_error_case
        )
        if result is None:
            result = "timeout" # Killed by untrusted_check because of timeout
        return result
    
    def __getitem__(self, i):
        return self.problems[i]

    def __len__(self):
        return len(self.problems)