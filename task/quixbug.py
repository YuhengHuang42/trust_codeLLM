# Refenrece: Automated Program Repair in the Era of Large Pre-trained Language Models
import glob
from difflib import unified_diff
import os
import subprocess

from .dataset_utils import CodeDataset
VARY_BASE_PROMPT = """# Provide a fix for the buggy function

# Buggy Function
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return fibonacci(n-1) - fibonacci(n-2)

# Fixed Function
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Provide a fix for the buggy function

# Buggy Function
{example_bug}

# Fixed Function
{example_fix}

# Provide a fix for the buggy function

# Buggy Function
{bug}

# Fixed Function
"""

def get_unified_diff(source, mutant):
    output = ""
    for line in unified_diff(source.split('\n'), mutant.split('\n'), lineterm=''):
        output += line + "\n"
    return output

# picking an example fix pairs from a project
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

class QuixbugDatasetPy(CodeDataset):
    def __init__(self, 
                 repair_data_path,
                 local_cache_path,
                 quixbug_system_path, 
                 used_prompt=VARY_BASE_PROMPT):
        # Reference: Automated Program Repair in the Era of Large Pre-trained Language Models
        self.repair_data_path = repair_data_path
        self.used_prompt = used_prompt
        self.local_cache_path = local_cache_path
        self.quixbug_system_path = quixbug_system_path
        self.env = os.environ.copy()
        self.clean_dataset = QuixbugDatasetPy.parse_python(self.repair_data_path)
        
        self.index = list(sorted(self.clean_dataset.keys()))
        os.makedirs(self.local_cache_path, exist_ok=True)
    
    def __getitem__(self, i):
        return self.clean_dataset[self.index[i]]

    def __len__(self):
        return len(self.clean_dataset)
    
    @staticmethod
    def parse_python(folder):
        ret = {}
        parent = os.path.join(folder, "QuixBugs/Python/fix/*.py")
        for file in glob.glob(parent):
            filename = os.path.basename(file)
            if "_test" in file or "node.py" in file:
                continue
            with open(file, "r") as f:
                x = f.read().strip()
            with open(os.path.join(folder, "QuixBugs/Python/buggy/{}".format(filename)), "r") as f:
                y = f.read().strip()
            ret[filename] = {'fix': x, 'buggy': y, 'diff': get_unified_diff(x, y)}
            diff_lines = get_unified_diff(y, x).splitlines()
            line_no = -1
            for line in diff_lines:
                if "@@" in line:
                    rline = line.split(" ")[1][1:]
                    line_no = int(rline.split(",")[0].split("-")[0])
                elif line_no == -1:
                    continue
                if line.startswith("-"):
                    ret[filename]['line_no'] = line_no - 2
                    ret[filename]['line_content'] = line[1:]
                    ret[filename]['type'] = 'modify'
                    ret[filename]['prefix'] = "\n".join(ret[filename]['buggy'].splitlines()[:ret[filename]['line_no']])
                    ret[filename]['suffix'] = "\n".join(ret[filename]['buggy'].splitlines()[ret[filename]['line_no'] + 1:])
                    break
                elif line.startswith("+"):
                    ret[filename]['line_no'] = line_no - 2
                    ret[filename]['line_content'] = line[1:]
                    ret[filename]['type'] = 'add'
                    ret[filename]['prefix'] = "\n".join(ret[filename]['buggy'].splitlines()[:ret[filename]['line_no']])
                    ret[filename]['suffix'] = "\n".join(ret[filename]['buggy'].splitlines()[ret[filename]['line_no']:])
                    break
                line_no += 1

            #print(y.splitlines()[ret[filename]['line_no']])

        assert (len(ret) == 40)  # should only be 40 buggy/fix pairs
        return ret

    def get_prompt(self, idx):
        # file_name: 'Chart-1.java'
        file_name = self.index[idx]
        example_bug, example_fix = pick_smallest_example_fix(self.clean_dataset, file_name)
        prompt = self.used_prompt.format(example_bug=example_bug, example_fix=example_fix, bug=self.clean_dataset[file_name]['buggy'])
        return prompt, self.clean_dataset[file_name]['buggy']
    
    def get_buggy_code(self, idx):
        return self.clean_dataset[self.index[idx]]['buggy']

    def get_fix_code(self, idx):
        return self.clean_dataset[self.index[idx]]['fix']
    
    def check_result(self, generated_code, idx):
        file_name = self.index[idx]
        tmp_path = os.path.join(self.local_cache_path, file_name)
        subprocess.run('rm -rf ' + str(tmp_path), shell=True)
        
        diff = get_unified_diff(self.clean_dataset[file_name]['buggy'], generated_code)
        if diff == "":
            result = "buggy"
        
        with open(tmp_path, 'w') as f:
            f.write(generated_code)
            
        bug = file_name.split(".py")[0]
        exit_code = subprocess.run("cd {}; python python_tester.py --bug {} --file {} --add_pf"
                                    .format(self.quixbug_system_path, bug, tmp_path), shell=True,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if exit_code.returncode == 0:
            result = "correct"
        else:
            result = "buggy"
        
        return result

class QuixbugDatasetJa(CodeDataset):
    def __init__(self, 
                 repair_data_path,
                 local_cache_path,
                 quixbug_system_path, 
                 used_prompt=VARY_BASE_PROMPT):
        # Reference: Automated Program Repair in the Era of Large Pre-trained Language Models
        self.repair_data_path = repair_data_path
        self.used_prompt = used_prompt
        self.local_cache_path = local_cache_path
        self.quixbug_system_path = quixbug_system_path
        self.env = os.environ.copy()
        self.clean_dataset = QuixbugDatasetJa.parse_java(self.repair_data_path)
        
        self.index = list(sorted(self.clean_dataset.keys()))
        os.makedirs(self.local_cache_path, exist_ok=True)
    
    def __getitem__(self, i):
        return self.clean_dataset[self.index[i]]

    def __len__(self):
        return len(self.clean_dataset)
    
    @staticmethod
    def parse_java(folder):
        ret = {}
        parent = os.path.join(folder, "QuixBugs/Java/fix/*.java")
        for file in sorted(glob.glob(parent)):
            filename = os.path.basename(file)
            with open(file, "r") as f:
                x = f.read().strip()
            with open(os.path.join(folder, "QuixBugs/Java/buggy/{}".format(filename)), "r") as f:
                y = f.read().strip()
            ret[filename] = {'fix': x, 'buggy': y, 'diff': get_unified_diff(y, x)}
            diff_lines = get_unified_diff(y, x).splitlines()
            remove, gap, add, single_hunk = False, False, False, True
            line_no = 0
            for line in diff_lines[2:]:
                if "@@" in line:
                    rline = line.split(" ")[1][1:]
                    line_no = int(rline.split(",")[0].split("-")[0])
                elif line.startswith("-"):
                    if not remove:
                        start_line_no = line_no
                    if gap:
                        single_hunk = False
                    remove = True
                    end_line_no = line_no
                elif line.startswith("+"):
                    if not remove and not add:
                        start_line_no = line_no
                    if not add:
                        end_line_no = line_no
                    add = True
                    if gap:
                        single_hunk = False
                else:
                    if remove:
                        gap = True

                if not single_hunk:
                    break
                line_no += 1

            #if not single_hunk:
                #print("Not single hunk bug")
            if single_hunk:
                #print("Single hunk bug")
                ret[filename]['prefix'] = "\n".join(ret[filename]['buggy'].splitlines()[:start_line_no - 2])
                ret[filename]['suffix'] = "\n".join(ret[filename]['buggy'].splitlines()[end_line_no - 2:])

        assert (len(ret) == 40)  # should only be 40 buggy/fix pairs
        return ret

    def get_prompt(self, idx):
        # file_name: 'Chart-1.java'
        file_name = self.index[idx]
        example_bug, example_fix = pick_smallest_example_fix(self.clean_dataset, file_name)
        prompt = self.used_prompt.format(example_bug=example_bug, example_fix=example_fix, bug=self.clean_dataset[file_name]['buggy'])
        return prompt, self.clean_dataset[file_name]['buggy']
    
    def get_buggy_code(self, idx):
        return self.clean_dataset[self.index[idx]]['buggy']
    
    def check_result(self, generated_code, idx):
        file_name = self.index[idx]
        tmp_path = os.path.join(self.local_cache_path, file_name)
        subprocess.run('rm -rf ' + str(tmp_path), shell=True)
        
        diff = get_unified_diff(self.clean_dataset[file_name]['buggy'], generated_code)
        if diff == "":
            result = "buggy"
        
        with open(tmp_path, 'w') as f:
            f.write(generated_code)
            
        bug = file_name.split(".java")[0]
        exit_code = subprocess.run("cd {}; python java_tester.py --bug {} --file {} --add_pf"
                                    .format(self.quixbug_system_path, bug, tmp_path), shell=True)
                                    #stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if exit_code.returncode == 0:
            result = "correct"
        else:
            result = "buggy"
        
        return result