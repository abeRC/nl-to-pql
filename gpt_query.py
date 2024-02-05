import openai
import pandas as pd
import os, time, sys
from multiprocessing import Process,Manager

from collections.abc import Sequence, Mapping
from typing import Callable

RETRIES = 3
TIMEOUT = 60 # seconds
WAIT_TIME = 50 # seconds
# rate limit is 3 requests/min

def construct_fewshot_messages (
    prompt : str, instruction_prompt : str, 
    train_queries: Sequence[str], train_answers : Sequence[str]
    ) -> Sequence[Mapping[str,str]]:

    # start messages list and add instruction
    messages = []
    messages.append({"role": "system", "content": instruction_prompt})

    # loop over train queries/answers
    for query, answer in zip(train_queries, train_answers):
        messages.append({"role": "user", "content": query})
        messages.append({"role": "assistant", "content": answer})
        
    # don't forget the prompt
    messages.append({"role": "user", "content": prompt})

    print(messages)
    return messages

def attempt_completion (client, messages, multiproc_return_dict):
    # attempt to get completion from openai
    
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        top_p=0.1,
        seed=42
    )
    # use proxy dict to return result
    multiproc_return_dict["result"] = completion

def gpt_query_individual (
    prompt : str, instruction_prompt : str, df_current_train : pd.DataFrame
    ) -> str:

    # check openai key
    client = openai.OpenAI()
    if client.api_key is None:
        print("No API key found!")
        sys.exit(1)

    # construct fewshot
    train_queries = df_current_train["English"]
    train_answers = df_current_train["PQL (reference)"]
    messages = construct_fewshot_messages(prompt, instruction_prompt, train_queries, train_answers)
    
    # setup for multiprocessing stuff
    retry_number = 0
    completion = None
    manager = Manager()
    multiproc_return_dict = manager.dict()
    attempt_completion_with_args = lambda : attempt_completion(client, messages, multiproc_return_dict)

    # generate completion with timeout and retries
    # based on https://stackoverflow.com/a/14924210/10736421
    print("\n############### FETCHING COMPLETION ... ##############")
    print(f"Prompt is: '{prompt}'")
    while retry_number < RETRIES and completion is None:
        print(f"\n############### ATTEMPT {retry_number} ... ##############")
        # create a new process, start it, and block for TIMEOUT seconds or until process finishes
        proc = Process(target=attempt_completion_with_args)
        proc.start()
        proc.join(TIMEOUT)

        # after TIMEOUT seconds, ...
        if proc.is_alive():
            # function didn't finish and must die
            print(">:) KILL KILL KILL KILL >:)")
            proc.terminate()
        else:
            # function finished properly
            print("success!")
            completion = multiproc_return_dict["result"]

            # wait a bit so we don't get rate limited
            time.sleep(WAIT_TIME)

        retry_number += 1 # don't forget to increase counter

    # important debug info
    print(f"fingerprint: {completion.system_fingerprint}")
    print(f"usage: {completion.usage}")
    print(f"finish reason: {completion.choices[0].finish_reason}")

    # result
    print(f"\nquestion: {prompt}")
    print("response:\n")
    print(completion.choices[0].message.content)

    # dump all completion data
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/log-{time.perf_counter_ns()}.txt", "w") as f:
        f.write(completion.model_dump_json(indent=2))
    
    # return completion result
    prompt_result = completion.choices[0].message.content
    return prompt_result

def main():
    type_list = [1, 1]
    train_prompt_list = [
        "For each case, count the number of events handled by the role 'PRE_APPROVER'.",
        "For each case, count the number of events handled by the role 'DIRECTOR'."
        ]
    train_pql_list = [
"""FILTER "log"."org:role" = 'PRE_APPROVER';
TABLE (
"log_CASES"."Case ID"  AS "Case ID",
COUNT("log"."org:role")  AS "Count"
);""",
"""FILTER "log"."org:role" = 'DIRECTOR';
TABLE (
"log_CASES"."Case ID"  AS "Case ID",
COUNT("log"."org:role")  AS "Count"
);"""
    ]
    prompt = "For each case, count the number of events handled by the role 'EMPLOYEE'."
    instruction_prompt = '''Given a request in natural language, construct the corresponding Celonis PQL query. The event log table will always be named "log", and the columns in it are as follows: "Case ID", "concept:name" (the activity column), "time:timestamp", "org:resource", "org:role", "id" (internal code). The case table is named "log_CASES" and contains one column named "Case ID".'''

    df = pd.DataFrame({"Type": type_list, "English": train_prompt_list, "PQL (reference)": train_pql_list})
    prompt_result = gpt_query_individual(prompt, instruction_prompt, df)

if __name__ == "__main__":
    main()