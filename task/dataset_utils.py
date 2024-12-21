from datasets import load_dataset
#from codetlingua.tools.utils import check_correctness, get_problem, untrusted_check
import abc
from loguru import logger
from abc import ABC

import utility.utils as utils


class CodeDataset(ABC):
    def __init__(self) -> None:
        self.problems = None
        
    @abc.abstractmethod
    def __len__():
        raise NotImplementedError
    
    @abc.abstractmethod
    def __getitem__(self, i):
        raise NotImplementedError
 
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

def find_code_block_positions(original_string, sub_string, filter_language=None):
    """
    Given original_string and sub_string, find all occurrences of the sub_string in original_string
    ---
    Args:
        original_string: string
        sub_string: string
    Output:
        List[List]: list of matched Line number. 
    """
    # Split the original and sub string into lines
    if filter_language is not None:
        original_lines = utils.normalize_code(original_string, filter_language)
        sub_lines = utils.normalize_code(sub_string, filter_language)
    else:
        original_lines = original_string.splitlines()
        sub_lines = sub_string.splitlines()
    if filter_language is None:
        cleaned_sub_lines = sub_lines
    else:
        cleaned_sub_lines = []
        for line in sub_lines:
            if utils.is_line_only_punctuators_pygments(line, filter_language):
                continue
            cleaned_sub_lines.append(line)
    
    # List to store all matching positions
    matches = []
    if len(cleaned_sub_lines) == 0:
        return matches
    
    # Iterate over the original string lines
    for i in range(len(original_lines)):
        # Check if the current line matches the first line of the sub_string
        recorded_line_number = []
        if original_lines[i].strip() == cleaned_sub_lines[0].strip():
            # Assume it's a match, start checking the following lines
            match_found = True
            start_index = i
            j = 1
            recorded_line_number.append(i)
            while(j < len(cleaned_sub_lines)):
                #for j in range(1, len(cleaned_sub_lines)):
                # If we go out of bounds of the original lines, set match_found to False
                if start_index + j >= len(original_lines):
                    match_found = False
                    break
                elif original_lines[start_index + j].strip() != cleaned_sub_lines[j].strip():
                    # We allow partial match. As long as the order is correct.
                    #break
                    start_index += 1
                    continue
                else:
                    recorded_line_number.append(start_index + j)
                    j += 1
            
            # If all lines matched, calculate the start and end positions in the original string
            if match_found:
                #start_pos = i
                #end_pos = i + len(cleaned_sub_lines)
                #matches.append((start_pos, end_pos))
                matches.append(recorded_line_number)
    
    # Return the list of matched positions as tuples (start_line, end_line)
    return matches

def find_buggy_positions(original_code: str, raw_buggy_code: str, logging_idx=-1, filter_language=None, without_filter=False):
    """
    Collect all buggy positions according to raw_buggy_code
    ---
    Args:
        original_code: str
        raw_buggy_code: str. Returned response from OPENAI.
        logging_idx: str. Used for reporting error and debugging
        filter_language: str. Whether to filter out lines that only contain punctuators or comments. Set to target language to enable filtering.
    Output:
        List[Tuple]: list of start and end Line Number tuples.
    """
    # The last one is the fully runnable code
    # The second-last one is the text description
    if without_filter is False:
        code_block, code_block_info = utils.extract_code_block(raw_buggy_code, None)
        buggy_code_list = code_block[:-2] # Skip the Indicator and runnable code.
    else:
        buggy_code_list = [raw_buggy_code]
        
    #buggy_code_list = code_block[:-2] # Skip the Indicator and runnable code.
    result = list()
    if len(buggy_code_list) == 0:
        return result
    for buggy_code in buggy_code_list:
        if len(buggy_code.strip()) == 0:
            continue
        positions = find_code_block_positions(original_code, buggy_code, filter_language) #  List[List]: list of matched Line number. 
        if len(positions) == 0:
            logger.error(f"No matching position find for {logging_idx}")
            logger.error(f"buggy_code: {buggy_code}")
            logger.error(f"Original code: {original_code}")
        result += positions
    return result


def get_candidate_tokens(data, key, tokenizer, lang, code_blocks_info=None):
    """
    Get candidate tokens per line.
    Use is_line_only_punctuators_pygments to filter out lines that only contain punctuators.
    ---
    Args:
        data: dict. data_index -> collected data mapping
        key: str. data_index
        tokenizer: TokenizerFast. The tokenizer used in model inference.
        lang: str. The language of the code block.
        code_blocks_info: List[List]. List of code block start and end char positions. Set to [[0, len(data[key]["str_output"])]] to skip extraction.
    Output:
        candidate_tokens: List[List]: List of token idx. Each list corresponds to a line in the code block.
    """
    if code_blocks_info is None:
        code_blocks, code_blocks_info = utils.extract_code_block(data[key]['str_output'])
    if len(code_blocks_info) == 0:
        return None
    target_code_block_info = code_blocks_info[-1] # The last code enclosed by ``` ```.
    #tokenized_info = tokenizer(data[key]['str_output'], loggru=True, add_special_tokens=False)
    #tokenized_info_start_idx = utils.remove_redundant_tuples(tokenized_info["offset_mapping"])
    #offset_mapping = tokenized_info["offset_mapping"][tokenized_info_start_idx:]
    tokenized_info = utils.ModelOutputCleaner.clean_first_tokenization(data[key]['str_output'], tokenizer)

    line_char_mapping = utils.get_line_to_char_index_mapping(data[key]['str_output'])
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
    end_line = line_num
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