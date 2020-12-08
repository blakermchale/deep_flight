import pandas as pd
import json
from glob import glob


def create_df(name, section=True):
    """
    Create dataframe. Removes extra values if there are further
    """
    if section:
        save_eps = 50
        name_parts = name.split("_")
        start_eps = int(name_parts[-2])
        end_eps = int(name_parts[-1])
        real_end_eps = (end_eps//save_eps) * save_eps
    with open(glob(f"{name}/openaigym.episode_batch*")[0]) as f:
        gym_dict = json.load(f)
    gym_dict.pop("episode_types")
    initial_reset_timestamp = gym_dict.pop("initial_reset_timestamp")

    df = pd.DataFrame(gym_dict)
    df["timestamps"] = df["timestamps"] - initial_reset_timestamp
    if section:
        if start_eps == 0:
            df = df[:(real_end_eps+1)]
        else:
            df = df[:real_end_eps]
    return df

# Create json from recorded training results
df_lst = []
# file_lst = [
#     "training_results_final_0_356", "training_results_final_351_406", 
#     "training_results_final_401_660"]
file_lst = ["training_results_test_0_657"]
for name in file_lst:
    if name == file_lst[0]:
        df_lst.append(create_df(name))
    else:
        df_lst.append(create_df(name))
        df_lst[-1]["timestamps"] += df_lst[-2]["timestamps"].iloc[-1]
df = pd.concat(df_lst).reset_index(drop=True)

df["steps"] = df["episode_lengths"].cumsum()
df.to_json("training_results_final.json")