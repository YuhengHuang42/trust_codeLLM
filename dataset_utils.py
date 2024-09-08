from datasets import load_dataset
import re
from codetlingua.tools.utils import check_correctness, get_problem, untrusted_check
import shelve
import json
import collections
import abc
import subprocess
import os
import javalang
import time
import signal
from loguru import logger

from abc import ABC

# Defects4jDataset prompt
JAVA_LONG_VARY_PROMPT = """// Provide a fix for the buggy function

// Buggy Function
int binarySearch(int arr[], int l, int r, int x)
{{
    if (r >= l) {{
        int mid = l + (r + l) / 2;
        if (arr[mid] == x)
            return mid;
        if (arr[mid] > x)
            return binarySearch(arr, l, mid - 1, x);
        return binarySearch(arr, mid + 1, r, x);
    }}
    return -1;
}}

// Fixed Function
int binarySearch(int arr[], int l, int r, int x)
{{
    if (r >= l) {{
        int mid = l + (r - l) / 2;
        if (arr[mid] == x)
            return mid;
        if (arr[mid] > x)
            return binarySearch(arr, l, mid - 1, x);
        return binarySearch(arr, mid + 1, r, x);
    }}
    return -1;
}}

// Provide a fix for the buggy function

// Buggy Function
{example_bug}

// Fixed Function
{example_fix}

// Provide a fix for the buggy function

// Buggy Function
{bug}

// Fixed Function
"""

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

class CodeDataset(ABC):
    def __init__(self) -> None:
        self.problems = None
        
    @abc.abstractmethod
    def __len__():
        raise NotImplementedError
    
    @abc.abstractmethod
    def __getitem__(self, i):
        raise NotImplementedError
    

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

#export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
#export PATH=$PATH:$JAVA_HOME/bin
class Defects4jDataset(CodeDataset):
    def __init__(self, repair_data_path, loc_folder, defect4j_path, java_home=None, local_cache_path="/tmp/"):
        # Reference: Automated Program Repair in the Era of Large Pre-trained Language Models
        self.repair_data_path = repair_data_path
        self.local_cache_path = local_cache_path
        self.loc_folder = loc_folder
        self.defect4j_path = defect4j_path
        self.java_home = java_home
        self.env = os.environ.copy()
        self.env["PATH"] = defect4j_path + os.pathsep + self.env["PATH"]
        self.cache = dict()
        if java_home is not None:
            assert "bin" not in java_home
            self.env["PATH"] = os.path.join(java_home, "bin") + os.pathsep + self.env["PATH"]
            if "JAVA_HOME" not in self.env:
                self.env["JAVA_HOME"] = java_home
        
        with open(repair_data_path, "r") as f:
            self.problems = json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(f.read())
        self.clean_result = Defects4jDataset.clean_parse_d4j(self.problems)
        self.index = list(self.problems.keys())
        
        self.metadata = dict()
        for bug_id in self.index:
            project = bug_id.split("-")[0]
            bug = bug_id.split("-")[1]
            self.metadata[bug_id] = {
                "project": project,
                "bug": bug,
                "start": self.problems[bug_id]["start"],
                "end": self.problems[bug_id]["end"],
            }
            
    def __len__(self):
        return len(self.problems)
    
    def __getitem__(self, i):
        return self.problems[self.index[i]]

    def check_result(self, generate_code, problem_id: int, completion_id=1, output_error_case=False):
        bug_id = self.index[problem_id]
        project = self.metadata[bug_id]["project"]
        bug = self.metadata[bug_id]["bug"]
        start = self.metadata[bug_id]["start"]
        end = self.metadata[bug_id]["end"]
        
        tmp_bug_id = "defect4j_test_" + bug_id
        tmp_path = os.path.join(self.local_cache_path, tmp_bug_id)
        subprocess.run('rm -rf ' + str(tmp_path), shell=True)
        subprocess.run("defects4j checkout -p %s -v %s -w %s" % (project, bug + 'b', str(tmp_path)), shell=True, env=self.env)
        #testmethods = os.popen('defects4j export -w %s -p tests.trigger' % str(tmp_path)).readlines()
        #source_dir = os.popen("defects4j export -p dir.src.classes -w " + str(tmp_path)).readlines()[-1].strip()
        
        cmd1 = 'defects4j export -w %s -p tests.trigger' % str(tmp_path)
        process = subprocess.Popen(cmd1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env)
        output, error = process.communicate()
        
        if process.returncode != 0:
            # Non-zero return code indicates failure
            error = error.decode().strip()
            logger.error(f"Error: Command failed with return code {process.returncode}")
            logger.error(f"Error message: {error}")
            raise Exception(f"Error: Command {cmd1} failed with return code {process.returncode}")
            return None, error
        
        testmethods = output.decode().splitlines()

        cmd2 = "defects4j export -p dir.src.classes -w " + str(tmp_path)
        process = subprocess.Popen(cmd2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env)
        output, error = process.communicate()
        if process.returncode != 0:
            # Non-zero return code indicates failure
            error = error.decode().strip()
            logger.error(f"Error: Command failed with return code {process.returncode}")
            logger.error(f"Error message: {error}")
            raise Exception(f"Error: Command {cmd2} failed with return code {process.returncode}")
        source_dir = output.decode().splitlines()[-1].strip()
        
        with open(os.path.join(self.loc_folder, f"{bug_id}.buggy.lines"), "r") as f:
            locs = f.read()
        loc = set([x.split("#")[0] for x in locs.splitlines()])  # should only be one
        loc = loc.pop()

        # This is for single-hunk repair
        try:
            with open(os.path.join(tmp_path, source_dir, loc), 'r') as f:
                source = f.readlines()
        except:
            with open(os.path.join(tmp_path, source_dir, loc), 'r', encoding='ISO-8859-1') as f:
                source = f.readlines()
        source = "\n".join(source[:start - 1]) + generate_code + "\n".join(source[end:])
        
        try:
            with open(os.path.join(tmp_path, source_dir, loc), 'w') as f:
                f.write(source)
        except:
            with open(os.path.join(tmp_path, source_dir, loc), 'w', encoding='ISO-8859-1') as f:
                f.write(source)
        
        compile_fail, timed_out, bugg, entire_bugg, syntax_error = self.run_d4j_test(source, testmethods, tmp_bug_id, tmp_path)
        
        if compile_fail:
            result = "compile_fail"
        elif timed_out:
            result = "timed_out"
        elif bugg:
            result = "buggy"
        elif syntax_error:
            result = "syntax_error"
        else:
            result = "correct"
        
        subprocess.run('rm -rf ' + str(tmp_path), shell=True)
        return result
        
        
        
    @staticmethod
    def clean_parse_d4j(data):
        cleaned_result = {}
        for k, v in data.items():
            lines = v['buggy'].splitlines()
            leading_white_space = len(lines[0]) - len(lines[0].lstrip())
            cleaned_result[k + ".java"] = {"buggy": "\n".join([line[leading_white_space:] for line in lines])}
            lines = v['fix'].splitlines()
            leading_white_space = len(lines[0]) - len(lines[0].lstrip())
            cleaned_result[k + ".java"]["fix"] = "\n".join([line[leading_white_space:] for line in lines])
        return cleaned_result

    def run_d4j_test(self, source, testmethods, bug_id, tmp_path):
        bugg = False
        compile_fail = False
        timed_out = False
        entire_bugg = False
        error_string = ""

        try:
            tokens = javalang.tokenizer.tokenize(source)
            parser = javalang.parser.Parser(tokens)
            parser.parse()
        except:
            #print("Syntax Error")
            return compile_fail, timed_out, bugg, entire_bugg, True

        
        for t in testmethods:
            # print(t.strip())
            cmd = 'defects4j test -w %s/ -t %s' % (str(tmp_path), t.strip())
            Returncode = ""
            error_file = open("stderr.txt", "wb")
            child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=error_file, bufsize=-1,
                                    start_new_session=True, env=self.env)
            while_begin = time.time()
            while True:
                Flag = child.poll()
                if Flag == 0:
                    Returncode = child.stdout.readlines()  # child.stdout.read()
                    self.cache[bug_id] = {"error_info": b"".join(Returncode).decode('utf-8')}
                    #print(b"".join(Returncode).decode('utf-8'))
                    error_file.close()
                    break
                elif Flag != 0 and Flag is not None:
                    compile_fail = True
                    error_file.close()
                    with open("stderr.txt", "rb") as f:
                        r = f.readlines()
                    for line in r:
                        if re.search(':\serror:\s', line.decode('utf-8')):
                            error_string = line.decode('utf-8')
                            break
                    #print(error_string)
                    self.cache[bug_id] = {"error_info": error_string}
                    break
                elif time.time() - while_begin > 15:
                    error_file.close()
                    os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                    timed_out = True
                    break
                else:
                    time.sleep(0.01)
            log = Returncode
            if len(log) > 0 and log[-1].decode('utf-8') == "Failing tests: 0\n":
                continue
            else:
                bugg = True
                break

        # Then we check if it passes all the tests, include the previously okay tests
        if not bugg:
            #print('So you pass the basic tests, Check if it passes all the test, include the previously passing tests')
            cmd = 'defects4j test -w %s/' % (str(tmp_path))
            Returncode = ""
            child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1,
                                    start_new_session=True, env=self.env)
            while_begin = time.time()
            while True:
                Flag = child.poll()
                if Flag == 0:
                    Returncode = child.stdout.readlines()  # child.stdout.read()
                    break
                elif Flag != 0 and Flag is not None:
                    bugg = True
                    break
                elif time.time() - while_begin > 180:
                    os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                    bugg = True
                    break
                else:
                    time.sleep(0.01)
            log = Returncode
            if len(log) > 0 and log[-1].decode('utf-8') == "Failing tests: 0\n":
                #print('success')
                pass
            else:
                entire_bugg = True

        return compile_fail, timed_out, bugg, entire_bugg, False
    
    

def extract_code_block(text, select_idx=None):
    code_blocks = []
    start = 0

    while True:
        # Find the start of a code block
        start_idx = text.find("```", start)
        if start_idx == -1:
            break

        # Find the end of the code block, starting after the found start
        end_idx = text.find("```", start_idx + 3)
        if end_idx == -1:
            break

        # Extract the content between the backticks
        # Skip any text on the same line as the opening backticks
        code_start = text.find("\n", start_idx + 3) + 1  # Find the start of the next line after the opening ```
        #print(code_start)
        #print(end_idx)
        code_block = text[code_start:end_idx].strip()
        
        code_blocks.append(code_block)

        # Update the start position to search for the next code block
        start = end_idx

    if select_idx is None:
        return code_blocks  # Return all matches if no specific index is requested

    # Return specific match if select_idx is provided and within range
    if 0 <= select_idx < len(code_blocks):
        return code_blocks[select_idx]
    else:
        return None  # Return None if select_idx is out of bounds


'''    
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
'''

def load_shelve(path):
    with shelve.open(path) as db:
        loaded_data = dict(db)
    return loaded_data
