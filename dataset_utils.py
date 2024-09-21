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

import utils

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
    def __init__(self, 
                 repair_data_path, 
                 loc_folder, 
                 defect4j_path, 
                 java_home=None, 
                 local_cache_path="/tmp/", 
                 only_same=True,
                 used_prompt=JAVA_LONG_VARY_PROMPT):
        # Reference: Automated Program Repair in the Era of Large Pre-trained Language Models
        self.repair_data_path = repair_data_path
        self.local_cache_path = local_cache_path
        self.loc_folder = loc_folder
        self.defect4j_path = defect4j_path
        self.java_home = java_home
        self.only_same = only_same
        self.used_prompt = used_prompt
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
        self.clean_dataset = Defects4jDataset.clean_parse_d4j(self.problems)
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

    def get_prompt(self, bug_id):
        # file_name: 'Chart-1.java'
        file_name = bug_id + ".java"
        example_bug, example_fix = Defects4jDataset.pick_smallest_example_fix(self.clean_dataset, file_name, only_same=self.only_same)
        prompt = self.used_prompt.format(example_bug=example_bug, example_fix=example_fix, bug=self.problems[bug_id]['buggy'])
        return prompt
        
        
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

    # picking an example fix pairs from a project
    @staticmethod
    def pick_smallest_example_fix(bugs, current_bug, only_same=False):
        def _get_relevant_bugs(bugs, current_bug, only_same):
            potential_pairs = []
            project = current_bug.split("-")[0]
            for file_name, bug in bugs.items():
                if file_name == current_bug:
                    continue
                if file_name.startswith(project + "-") and only_same:
                    potential_pairs.append((len(bug['buggy']) + len(bug['fix']), file_name))
                elif not only_same:
                    potential_pairs.append((len(bug['buggy']) + len(bug['fix']), file_name))
            # sort from smallest to largest
            potential_pairs.sort(key=lambda x: x[0])
            return potential_pairs
        
        potential_pairs = _get_relevant_bugs(bugs, current_bug, only_same)
        if len(potential_pairs) == 0:
            # No other bugs in the same project
            potential_pairs = _get_relevant_bugs(bugs, current_bug, only_same=False)
        return bugs[potential_pairs[0][1]]['buggy'], bugs[potential_pairs[0][1]]['fix']
    
    

def extract_code_block(text, select_idx=None):
    """
    Extract code block enclosed by ``` ```.
    Return the code block and its start and end positions in the text.
    """
    code_blocks = []
    code_blocks_info = []
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
        

        # Update the start position to search for the next code block
        start = end_idx
        if len(code_block) == 0:
            continue
        
        code_blocks.append(code_block)
        code_blocks_info.append([code_start, end_idx])

    if select_idx is None:
        return code_blocks, code_blocks_info  # Return all matches if no specific index is requested

    # Return specific match if select_idx is provided and within range
    if 0 <= select_idx < len(code_blocks):
        return code_blocks[select_idx], code_blocks_info[select_idx]
    else:
        return None, None  # Return None if select_idx is out of bounds


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

# === Utility functions for finding positions of substrings in strings for code translation dataset. ===
def find_all_substring_positions(original_string, sub_string):
    """
    Given original_string and sub_string, find all occurrences of the sub_string in original_string
    ---
    Args:
        original_string: string
        substring: string
    Output:
        List[Tuple]: list of start and end position tuples. 
    """
    positions = []
    start_pos = 0
    
    # Loop through the string to find all occurrences
    while True:
        start_pos = original_string.find(sub_string, start_pos)
        
        # If no more occurrences are found, break the loop
        if start_pos == -1:
            break
        
        # Calculate the end position
        end_pos = start_pos + len(sub_string) - 1
        
        # Append the start and end positions as a tuple to the list
        positions.append((start_pos, end_pos))
        
        # Move the start_pos to the next character after the current match to avoid infinite loop
        start_pos += 1
    
    return positions

def find_code_block_positions(original_string, sub_string):
    """
    Given original_string and sub_string, find all occurrences of the sub_string in original_string
    ---
    Args:
        original_string: string
        sub_string: string
    Output:
        List[Tuple]: list of start and end Line Number tuples. 
    """
    # Split the original and sub string into lines
    original_lines = original_string.splitlines()
    sub_lines = sub_string.splitlines()
    
    # List to store all matching positions
    matches = []
    
    # Iterate over the original string lines
    for i in range(len(original_lines)):
        # Check if the current line matches the first line of the sub_string
        if original_lines[i].strip() == sub_lines[0].strip():
            # Assume it's a match, start checking the following lines
            match_found = True
            for j in range(1, len(sub_lines)):
                # If we go out of bounds of the original lines or a line doesn't match, set match_found to False
                if i + j >= len(original_lines) or original_lines[i + j].strip() != sub_lines[j].strip():
                    match_found = False
                    break
            
            # If all lines matched, calculate the start and end positions in the original string
            if match_found:
                start_pos = i
                end_pos = i + len(sub_lines)
                matches.append((start_pos, end_pos))
    
    # Return the list of matched positions as tuples (start_line, end_line)
    return matches

def find_buggy_positions(original_code: str, raw_buggy_code: str, logging_idx=-1):
    """
    Collect all buggy positions according to raw_buggy_code
    ---
    Args:
        original_code: str
        raw_buggy_code: str. Returned response from OPENAI.
        logging_idx: str. Used for reporting error and debugging
    Output:
        List[Tuple]: list of start and end Line Number tuples.
    """
    # The last one is the fully runnable code
    # The second-last one is the text description
    code_block, code_block_info = extract_code_block(raw_buggy_code, None)
    buggy_code_list = code_block[:-2]
    result = list()
    if len(buggy_code_list) == 0:
        return result
    for buggy_code in buggy_code_list:
        positions = find_code_block_positions(original_code, buggy_code)
        if len(positions) == 0:
            logger.error(f"No matching position find for {logging_idx}")
            logger.error(f"buggy_code: {buggy_code}")
            logger.error(f"Original code: {original_code}")
        result += positions
    return result

def get_candidate_tokens(data, key, tokenizer, lang):
    """
    Get candidate tokens per line.
    Use is_line_only_punctuators_pygments to filter out lines that only contain punctuators.
    ---
    Args:
        data: dict. data_index -> collected data mapping
        key: str. data_index
        tokenizer: TokenizerFast. The tokenizer used in model inference.
        lang: str. The language of the code block.
    Output:
        candidate_tokens: List[List]: List of token idx. Each list corresponds to a line in the code block.
    """
    code_blocks, code_blocks_info = extract_code_block(data[key]['str_output'])
    line_char_mapping = utils.get_line_to_char_index_mapping(data[key]['str_output'])

    target_code_block_info = code_blocks_info[-1] # The last code enclosed by ``` ```.
    tokenized_info = tokenizer(data[key]['str_output'], return_offsets_mapping=True, add_special_tokens=False)

    line_num = len(line_char_mapping)
    split_response = data[key]['str_output'].splitlines()
    # Find the start position of the code block
    for idx in range(line_num):
        if line_char_mapping[idx][-1] <= target_code_block_info[0]:
            continue
        else:
            start_line = idx
            break

    # Find the end position of the code block
    for idx in range(start_line, line_num):
        if line_char_mapping[idx][0] < target_code_block_info[1]:
            continue
        else:
            end_line = idx
            break

    # Collect candidate tokens per line.
    # Ignore lines that only contain punctuators such as ;}
    candidate_tokens = []
    for idx in range(start_line, end_line):
        if utils.is_line_only_punctuators_pygments(split_response[idx], lang):
            continue
        tokens = utils.get_token_indices(tokenized_info["offset_mapping"], line_char_mapping[idx][0], line_char_mapping[idx][1])
        candidate_tokens.append(tokens)
    return candidate_tokens


def load_shelve(path):
    with shelve.open(path) as db:
        loaded_data = dict(db)
    return loaded_data

def load_shelve_and_resume(dir_path):
    """
    Looping through the files under the dir_path. If there is more than
    one shelve file, raise an Exception. Otherwise, record the already
    saved count using len(loaded_data) and return the next index
    to resume.
    """
    import os
    import shelve
    # Get list of all shelve files in the directory
    shelve_files = [f for f in os.listdir(dir_path) if f.endswith('.db') or f.endswith('.dat')]

    # If there's more than one shelve file, raise an exception
    if len(shelve_files) > 1:
        raise Exception("More than one shelve file found in the directory.")
    
    # If no shelve file is found, return 0 as the starting index
    if not shelve_files:
        return 0, None
    
    # Load the shelve file
    shelve_file = os.path.splitext(shelve_files[0])[0]
    path = os.path.join(dir_path, shelve_file)
    with shelve.open(path) as db:
        loaded_data = list(db.keys())
    
    # Return the next index to resume
    return len(loaded_data), path