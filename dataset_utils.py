from datasets import load_dataset
import re
from codetlingua.tools.utils import check_correctness, get_problem, untrusted_check
import shelve

"""
CodeNet:
PLs: C, C++, Go, Java, Python
# Samples / Language: 200
# Tests / Sample: 1
AVATAR:
PLs: Java, Python
# Samples / Language: 250
# Tests / Sample: ~50
"""
class CodetlinguaDataset():
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
    
    def check_result(self, generate_code, problem_id, completion_id=1):
        result = untrusted_check(
            self.problems[problem_id],
            generate_code,
            self.target_lang,
            completion_id,
        )
        if result is None:
            result = "timeout" # Killed by untrusted_check because of timeout
        return result
    
    def __getitem__(self, i):
        return self.problems[i]

    def __len__(self):
        return len(self.problems)

def extract_code_block(text, select_idx=0):
    # Regular expression to find any code block starting and ending with ```
    pattern = r"```.*?\n(.*?)```"
    match = re.findall(pattern, text, re.DOTALL)
    
    if select_idx < 0:
        return match

    if len(match) > 0:
        return match[select_idx]
    else:
        return None

    #if match:
    #    # Return the matched code block without the delimiters
    #    return match.group(1).strip()
    #else:
    #    return None


def load_shelve(path):
    with shelve.open(path) as db:
        loaded_data = dict(db)
    return loaded_data