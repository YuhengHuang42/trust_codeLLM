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
    "\n\nReference correct code:\n\n```{}```".format(gt_code) +\
    "\n\n**Translated code to be corrected**:\n\n" + generated_code +\
    "\n\nThe Error in the translated code: {}".format(code_result) +\
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