from . import utils

TRANSLATION_OPENAI_PROMPT = "Here is a code translation result from {} to {}. But the {} code is wrong. Please help me identify which line(s) of code are wrong and give me a correct version. You could use multiple blocks (enclosed by ``` ```) if it is multi-chunk bug. Your answer should follow the format: \n\
Erroneous code: ``` THE ORIGINAL INCORRECT CODE LINE(s)```\n\
Corrected version: ``` NEW CODE ```. The NEW CODE should be the full version that is runnable."

TRANSLATION_EXAMPLE_PROMPT = '''\n\nExample Response Format:\n\n\
The erroneous code block(s): 
```
int N = sc<s> of.nextInt();
```

```
cum_remainders[i + 1] = (cum_remainders[i] + A[i]) % M;
```

```
for (int count : remainder_counts) {
    combinations += count * (count - 1) / 2;
}
```\n\n\
The corrected version: 
```
import java.util.*;
public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int N = scanner.nextInt();
        int M = scanner.nextInt();
        long[] A = new long[N];
        for (int i = 0; i < N; i++) {
            A[i] = scanner.nextLong();
        }
        long[] cum_remainders = new long[N + 1];
        int[] remainder_counts = new int[M];
        remainder_counts[0] = 1;  // Initialize for remainder 0
        for (int i = 0; i < N; i++) {
            cum_remainders[i + 1] = (cum_remainders[i] + A[i]) % M;
            if (cum_remainders[i + 1] < 0) {
                cum_remainders[i + 1] += M;
            }
            remainder_counts[(int) cum_remainders[i + 1]]++;
        }
        long combinations = 0;
        for (int count : remainder_counts) {
            if (count > 1) {
                combinations += (long) count * (count - 1) / 2;
            }
        }
        System.out.println(combinations);
    }
}
```\n\n'''

GENERATION_SYSTEM_PROMPT = """Below is the code generated according to the instructions in the comment. The comment is ground truth and the code is inconsistent to the instructions. Do not modify comment. Please help me identify which line(s) of code are wrong. You could use multiple blocks (enclosed by ``` ```) if it is a multi-chunk bug. Finally, please present a full runnable corrected version based on the bugs you find."""

GENERATION_EXAMPLE_PROMPT = '''\n\nExample Response Format:\n\n\
The erroneous code block(s): 
    
```
even = [i for i in l if i % 2 == 0]
odd = [i for i in l if i % 2!= 0]
return sorted(even) + sorted(odd)
```


```
even = 1
odd = 2
```

The corrected version: 
```
from typing import List
def sort_even(l: list):
    even = [l[i] for i in range(len(l)) if i % 2 == 0]
    even.sort()
    return [even[i // 2] if i % 2 == 0 else l[i] for i in range(len(l))]
```\n\n'''

EDITTING_SYSTEM_PROMPT = """You are given a code editing problem where the provided solution does not meet the requirements outlined in the instructions. Your task is to:
1. Identify the specific line(s) of code that contain errors. Clearly indicate each erroneous part using multiple code blocks if there are several separate issues (enclosed within triple backticks (```).
2. Provide a fully corrected version of the code, ensuring it is runnable and satisfies all given requirements.
"""

EDITTING_EXAMPLE_PROMPT = '''\n\nExample Response Format:\n\n\
Instruction: Change data structure of a and c from List to Dict.
Original Code:
```
a = list()
b = 1
c = list()
```

The wrong edited code:
```
a = set()
b = 1
c = set()
```

Your answer:
The erroneous code block(s): 
```
a = set()
```

```
c = set()
```

The corrected version: 
```
a = dict()
b = 1
c = dict()
```
'''



def get_translation_error_prompt(source_lang, 
                                 target_lang,
                                 original_code,
                                 gt_code,
                                 generated_code,
                                 code_result,
                                 trans_openai_prompt=TRANSLATION_OPENAI_PROMPT, 
                                 example_prompt=TRANSLATION_EXAMPLE_PROMPT):
    openai_prompt = trans_openai_prompt.format(source_lang, target_lang, target_lang)
    openai_system_input = openai_prompt + example_prompt
    openai_user_input = "Now here is the problem:\n\n" +\
    "Original code:\n\n" + original_code +\
    "\n\nReference correct code:\n\n```\n{}\n```".format(gt_code) +\
    "\n\n**Translated code to be corrected**:\n\n" + generated_code +\
    "\n\nThe Error in the translated code: {}".format(code_result) +\
    "\n\nYour answer:\n"
    message = {"system": openai_system_input, "user": openai_user_input}
    return message

def get_generation_error_prompt(generated_code,
                                error_test_case=None,
                                openai_prompt=GENERATION_SYSTEM_PROMPT,
                                example_prompt=GENERATION_EXAMPLE_PROMPT
                                ):
    openai_system_input = openai_prompt + example_prompt
    openai_user_input = "The code is:\n\n" +\
    "```\n{}\n```\n\n".format(generated_code)
    if error_test_case is not None:
        openai_user_input += "The test cases that failed are: \n{}\n\n".format("".join(error_test_case))
    openai_user_input += "Your answer:\n"
    message = {"system": openai_system_input, "user": openai_user_input}
    return message

def get_editting_error_prompt(
                            original_code,
                            generated_code,
                            instruction,
                            gt_code,
                            openai_prompt=EDITTING_SYSTEM_PROMPT,
                            example_prompt=EDITTING_EXAMPLE_PROMPT):
    openai_system_input = openai_prompt + example_prompt
    openai_user_input = "Now here is the problem\n\n" +\
    "Instruction: {}\n\n".format(instruction) +\
    "Original Code: \n```{}\n```\n\n".format(original_code) +\
    "Reference correct code: \n```\n{}\n```\n\n".format(gt_code) +\
    "The wrong edited code: \n```{}\n```\n\n".format(generated_code) +\
    "\n\nYour answer:\n"
    message = {"system": openai_system_input, "user": openai_user_input}
    return message

def generate_jsonl_for_openai(request_id_list, 
                              message_list, 
                              output_path,
                              max_tokens=None, 
                              model_type="gpt-4o-mini", 
                              url="/v1/chat/completions"):
    """
    Prepare input batch data for OPENAI API
    Args:
        request_id_list: used to index the message_list.
        message_list: list of messages to be sent to the API
        output_path: output file path
        max_tokens: maximum tokens to generate
        model_type: model type of OPENAI API
        url: API endpoint
    """
    assert len(request_id_list) == len(message_list)
    data = []
    requestid_to_message = dict(zip(request_id_list, message_list))
    for idx, item in enumerate(request_id_list):
        request_id = f"request-{item}"
        message = message_list[idx]
        body = {"model": model_type, 
         "messages": [{"role": "system", "content": message["system"]},
                      {"role": "user", "content": message["user"]}],
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        per_request = {
            "custom_id": request_id,
            "method": "POST",
            "url": url,
            "body": body
        }
        data.append(per_request)
    
    utils.write_jsonl(output_path, data)
    
    return data, requestid_to_message

def submit_batch_request_openai(client, 
                                input_file_path, 
                                url="/v1/chat/completions", 
                                completion_window="24h", 
                                description="code analysis"):
    """
    Submit the batch task to OPENAI API
    Args:
        client: OPENAI API client (client = openai.OpenAI(api_key=api_key))
        input_file_path: input file path
        url: API endpoint
        completion_window: completion window
        description: description of the submitted task
    """
    batch_input_file = client.files.create(
        file=open(input_file_path, "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    batch_submit_info = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint=url,
        completion_window=completion_window,
        metadata={
        "description": description
        }
    )
    batch_submit_info_id = batch_submit_info.id
    batch_result_info = client.batches.retrieve(batch_submit_info_id)
    
    return (batch_submit_info, batch_result_info)