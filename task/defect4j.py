import logger
import json
import subprocess
import os
import javalang
import collections
import time
import signal
import re


from .dataset_utils import CodeDataset


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