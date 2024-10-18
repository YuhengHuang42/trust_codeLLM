import subprocess
from subprocess import Popen, PIPE
import multiprocessing
import time
from multiprocessing import Value
import shutil
from typing import Dict, Iterable, List, Optional, Union, Any, Tuple
import numpy as np
import os
import datasets
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
PY_LANGUAGE = Language(tspython.language())

from .dataset_utils import CodeDataset
SUCCESS = "success"
RUNTIME_FAILED = "runtime_failed"
COMPILE_FAILED = "compile_failed"
INFINTIE_LOOP = "infinite_loop"
TEST_FAILED = "test_failed"

_SUCCESS = 0
_RUNTIME_FAILED = 1
_COMPILE_FAILED = 2
_INFINTIE_LOOP = 3
_TEST_FAILED = 4
_UNKNOWN = 5
COMPLETION_PLACEHOLDER = {
    "python": "# TODO: Your code here",
    "java": "/* TODO: Your code here */",
    "cpp": "/* TODO: Your code here */",
    "csharp": "/* TODO: Your code here */",
}

_mapping = {_SUCCESS: SUCCESS, _RUNTIME_FAILED: RUNTIME_FAILED, _COMPILE_FAILED: COMPILE_FAILED, _INFINTIE_LOOP: INFINTIE_LOOP, _TEST_FAILED: TEST_FAILED, _UNKNOWN: None}

class SAFIMDataset(CodeDataset):
    def __init__(self, task="block_v2", language="python"):
        self.problems = datasets.load_dataset("gonglinyuan/safim", task, split="test")
        self.language = language
        if language is not None:
            self.problems = self.problems.filter(lambda x: x["lang"] == language)
    
    def __len__(self):
        return len(self.problems)

    def __getitem__(self, i):
        return self.problems[i]

    @staticmethod
    def replace_placeholder(eval_prompt, generated_code):
        start_idx = eval_prompt.find("{{completion}}")
        assert start_idx != -1
        # Calculate the end index of {{completion}}
        end_idx = start_idx + len("{{completion}}")
        
        # Replace {{completion}} with generate_code manually
        full_code = eval_prompt[:start_idx] + generated_code + eval_prompt[end_idx:]
        
        # The start and end indices for generate_code in the full_code are the same as {{completion}}
        generate_code_start_idx = start_idx
        generate_code_end_idx = generate_code_start_idx + len(generated_code)

        # Output the modified full_code and the start/end positions of generate_code
        return full_code, (generate_code_start_idx, generate_code_end_idx)
    
    def check_result(self, processed_code, problem_id: int, completion_id=1, output_error_case=False):
        test_cases = eval(self.problems[problem_id]['unit_tests'])
        #generate_code = self.postprocess(problem_id, generate_code)
        if processed_code == self.problems[problem_id]["ground_truth"]:
            return SUCCESS
        else:
            full_code, replace_idx = SAFIMDataset.replace_placeholder(self.problems[problem_id]['eval_prompt'], processed_code)
            #full_code = self.problems[problem_id]['eval_prompt'].replace("{{completion}}", generate_code)
        result = untrusted_check(
            full_code,
            problem_id,
            test_cases
        )
        return result

    def assemble_infilling_spm_prompt(self, prefix: str, suffix: str, model_type: str) -> str:
        if "codellama" in model_type.lower():
            return "<PRE>" + " <SUF>" + suffix + " <MID>" + prefix
        elif "deepseekcoder" in model_type.lower():
            return "<｜fim▁begin｜>" + "<｜fim▁hole｜>" + suffix + "<｜fim▁end｜>" + prefix

    def get_infilling_parts(self, sample):
        parts = sample["prompt"].split(COMPLETION_PLACEHOLDER[sample["lang"]])
        assert len(parts) == 2
        return parts
    
    def get_prompt(self, problem_id: int, model_type: str):
        prefix, suffix = self.get_infilling_parts(self.problems[problem_id])
        prompt = self.assemble_infilling_spm_prompt(prefix, suffix, model_type)
        return prompt
    
    def postprocess(self, output: str, problem_id: int):
        generate_code = truncate_line_until_block(self.problems[problem_id], output)
        #full_code = self.problems[problem_id]['eval_prompt'].replace("{{completion}}", generate_code)
        return generate_code
        
    
    
    
        

def get_parser(lang):
    assert lang.lower() == "python"
    parser = Parser(PY_LANGUAGE)
    #parser.set_language(TS_LANG[lang])
    return parser

class ASTVisitor:

    def __init__(self, with_ndtypes=False, print_debug_outputs=False):
        self.with_ndtypes = with_ndtypes
        self.print_debug_outputs = print_debug_outputs
        self.stack = []
        self.ndtypes = []

    def enter(self, node) -> bool:
        return True

    def leave(self, node):
        pass

    def enter_leaf(self, node):
        pass

    def print_stack(self, node):
        depth = len(self.stack)
        print(" " * depth * 2 + node.type)

    def on_enter(self, node) -> bool:
        if self.print_debug_outputs:
            self.print_stack(node)
        if self.with_ndtypes:
            self.ndtypes.append((node.start_byte, True, node.type))
        enter_fn = getattr(self, "enter_%s" % node.type, self.enter)
        r = enter_fn(node)
        if node.child_count == 0:
            self.enter_leaf(node)
        self.stack.append(node.type)
        return r

    def on_leave(self, node):
        assert self.stack.pop() == node.type
        leave_fn = getattr(self, "leave_%s" % node.type, self.leave)
        r = leave_fn(node)
        # print("on leave ", node.type)
        if self.with_ndtypes:
            self.ndtypes.append((node.end_byte, False, node.type))
        return r

    def walk(self, root_node):
        if root_node is None:
            return

        cursor = root_node.walk()
        has_next = True

        while has_next:
            current_node = cursor.node

            # Step 1: Try to go to next child if we continue the subtree
            if self.on_enter(current_node):
                has_next = cursor.goto_first_child()
            else:
                has_next = False

            # Step 2: Try to go to next sibling
            if not has_next:
                self.on_leave(current_node)
                has_next = cursor.goto_next_sibling()

            # Step 3: Go up until sibling exists
            while not has_next and cursor.goto_parent():
                self.on_leave(cursor.node)  # We will never return to this specific parent
                has_next = cursor.goto_next_sibling()

    def __call__(self, root_node):
        return self.walk(root_node)


class ErrorCheckVisitor(ASTVisitor):
    def __init__(self, with_ndtypes=False):
        super().__init__(with_ndtypes)
        self.error_cnt = 0

    def enter_ERROR(self, node):
        if node.text.decode("utf-8") != ";":
            self.error_cnt += 1

def match_prefix_and_suffix(l1, l2):
    p = 0
    while p < len(l1) and p < len(l2):
        if l1[p] == l2[p]:
            p += 1
        else:
            break
    q = 0
    while -q < len(l1) and -q < len(l2):
        if l1[q - 1] == l2[q - 1]:
            q -= 1
        else:
            break
    return p, q

def truncate_line_until_block(sample, code):
    parser = get_parser(sample["lang"])
    lines = code.splitlines(keepends=True)
    while lines:
        eval_prefix, eval_suffix = sample['eval_prompt'].split("{{completion}}")
        eval_prefix = eval_prefix.encode("utf-8")
        eval_suffix = eval_suffix.encode("utf-8")
        completion = "".join(lines).encode("utf-8")
        if sample["lang"] == "python":
            code_bytes_0 = eval_prefix + b"pass" + eval_suffix
        else:
            code_bytes_0 = eval_prefix + eval_suffix
        code_bytes_1 = eval_prefix + completion + eval_suffix

        visitor = ErrorCheckVisitor(with_ndtypes=True)
        tree = parser.parse(code_bytes_1)
        visitor(tree)
        if visitor.error_cnt > 0:
            lines.pop()
            continue
        visitor_trace_1 = [(x, y) for _, x, y in visitor.ndtypes]

        visitor = ErrorCheckVisitor(with_ndtypes=True)
        tree = parser.parse(code_bytes_0)
        visitor(tree)
        assert visitor.error_cnt == 0
        visitor_trace_0 = [(x, y) for _, x, y in visitor.ndtypes]
        if len(visitor_trace_0) > len(visitor_trace_1):
            lines.pop()
            continue

        prefix_matched, suffix_matched = match_prefix_and_suffix(visitor_trace_0, visitor_trace_1)
        matched_diff = len(visitor_trace_0) - (prefix_matched - suffix_matched)
        if sample["lang"] == "python":
            matched_diff -= 4
        if matched_diff == 0:
            break
        else:
            lines.pop()
    return "".join(lines)

      
        
def exec_sample(code, problem_id, test_cases, stat: Value, output_error_case: bool=False,):
    os.makedirs(f"temp_dir/{problem_id}/", exist_ok=True)
     
    with open(f"temp_dir/{problem_id}/{problem_id}.py", "w") as f:
        f.write(code)

    try:
        subprocess.run(f"python3 -m py_compile temp_dir/{problem_id}/{problem_id}.py", check=True, capture_output=True, shell=True)

        #test_io = problem['test_IO']
        for i in range(len(test_cases)):
            f_in = test_cases[i]['input']
            f_out = test_cases[i]['output'][0]
            
            p = Popen(f"python3 temp_dir/{problem_id}/{problem_id}.py", stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
            
            try:
                stdout, stderr_data = p.communicate(input=f_in.encode(), timeout=100)
            except subprocess.TimeoutExpired:
                stat.value = _INFINTIE_LOOP
                break
            
            try:
                if float(stdout.decode())%1 == 0:
                    stdout = str(int(float(stdout.decode())))
                    f_out = str(int(float(f_out)))
                else:
                    # find how many decimal points are there in the output
                    stdout_temp = stdout.decode().strip()
                    f_out_temp = f_out.strip()
                    f_out_total_dec_points = len(f_out_temp.split(".")[1])
                    stdout_total_dec_points = len(stdout_temp.split(".")[1])
                    min_dec_points = min(f_out_total_dec_points, stdout_total_dec_points)

                    stdout = str(round(float(stdout.decode()), min_dec_points))
                    f_out = str(round(float(f_out), min_dec_points))

            except:
                try: # if stdout is already decoded as String, then pass
                    stdout = stdout.decode()
                except:
                    pass
            
            if(stdout.strip()==f_out.strip()):
                stat.value = _SUCCESS
                continue

            else:
                if output_error_case:
                    print("Error Test Case: {}".format(i))
                if stderr_data.decode()=='':
                    stat.value = _TEST_FAILED
                else:
                    stat.value = _RUNTIME_FAILED
                break

    except Exception as e:
        if output_error_case:
            print("Error Test Case", "Error: ", e, e.stderr)
        stat.value = _COMPILE_FAILED

    shutil.rmtree(f"temp_dir/{problem_id}")
    
# def exec_sample(code, problem_id, test_cases, stat: Value, output_error_case: bool=False,):
def untrusted_check(
    code: str,
    problem_id: int,
    test_cases,
    output_error_case: bool=False,
    env: dict=None,
) -> Tuple[str, np.ndarray]:

    # shared memory objects
    stat = Value("i", _UNKNOWN)

    p = multiprocessing.Process(
        target=exec_sample,
        args=(
            code,
            problem_id,
            test_cases,
            stat,
            output_error_case,
        ),
    )

    p.start()
    p.join(100)

    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    return _mapping[stat.value]