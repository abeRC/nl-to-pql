import pandas as pd
import numpy as np
from gpt_query import gpt_query_individual

# OPENAI_API_KEY environment variable should be set to an appropriate value

queries_csv_filename = "Consultas NL - PQL - Página1.csv"
results_csv_filename = "Resultados individuais.csv"

test_indices = [6,
                5, 11,
                0, 2, 8, 
] # 1f 2m 3d
initial_train_indices = [3,
                        12,
                        1,
] # 1f 1m 1d

def get_train_config (df_current_train_types : pd.Series):
    all_counts = df_current_train_types.value_counts()
    return f"E:{all_counts[1]} M:{all_counts[2]} H:{all_counts[3]}"

def gpt_query_df_test (df_current_train, df_test, instruction_prompt : str, train_config : str) -> pd.DataFrame:
    generated_pql = [] 
    for _, row in df_test.iterrows():
        prompt = row["English"]
        #prompt_result = row["PQL (reference)"] # DBG
        prompt_result = gpt_query_individual(prompt, instruction_prompt, df_current_train)

        generated_pql.append(prompt_result)
        # NOTE: can't/shouldn't use df.append

    results_for_test_with_train = pd.DataFrame({
        "Configuration": train_config,
        "Query ID": df_test["Query ID"], 
        "PQL (reference)": df_test["PQL (reference)"], 
        "PQL (generated)": generated_pql
    })
    return results_for_test_with_train

# separate out instruction prompt
df_with_instruction = pd.read_csv(queries_csv_filename)
instruction_prompt = df_with_instruction.iloc[0,2]

# select df without instruction prompt and cast to int after removing NA in column
df = df_with_instruction.iloc[1:,:].astype({'Type': np.int32})

# select test df and train df
test_indices_bool = np.bincount(test_indices, minlength=df.shape[0]).astype(bool) # turn indices into one-hot
df_test = df.iloc[test_indices_bool, :]
train_indices_bool = ~test_indices_bool
df_full_train = df.iloc[train_indices_bool, :]

# prepare initial_train and remaining_train for loop
current_train_indices_bool = np.bincount(initial_train_indices, minlength=df.shape[0]).astype(bool)
remaining_train_indices_bool = ~current_train_indices_bool*train_indices_bool # not initial but still in train
df_current_train : pd.DataFrame = df.iloc[current_train_indices_bool, :]
df_remaining_train : pd.DataFrame = df.iloc[remaining_train_indices_bool, :]

# start all_results table 
all_results_list = [] 

# generate PQL for all tests using the first training configuration
print(f"current query ids: {list(df_current_train.iloc[:,0])}\n")
train_config = get_train_config(df_current_train["Type"])
print(f"current train config: {train_config}")
results = gpt_query_df_test(df_current_train, df_test, instruction_prompt, train_config)
all_results_list.append(results)

# for each training configuration, generate PQL for all tests
df_remaining_train = df_remaining_train.sort_values('Type') # sort by type ("difficulty")
for _, row in df_remaining_train.iterrows():

    # update df_current_train
    row_index = row["Query ID"]
    print("type:", row["Type"])
    current_train_indices_bool[row_index] = True
    df_current_train = df.iloc[current_train_indices_bool, :]

    # generate PQL for all tests using this new training configuration
    print(f"current query ids: {list(df_current_train.iloc[:,0])}\n")
    train_config = get_train_config(df_current_train["Type"])
    print(f"current train config: {train_config}")
    results = gpt_query_df_test(df_current_train, df_test, instruction_prompt, train_config)
    all_results_list.append(results)
    

# compile all results into final table
print("compiling results into final table...")
all_results = pd.concat(all_results_list)
print(all_results.shape)
all_results.to_csv(results_csv_filename, sep=",", index=False)
