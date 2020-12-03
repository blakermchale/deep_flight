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
        print(f"Real end episode: {real_end_eps}, Start: {start_eps}, End: {end_eps}")
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
df1 = create_df("training_results_final_0_356")
df2 = create_df("training_results_final_351_406")
df2["timestamps"] += df1["timestamps"].iloc[-1]
df3 = create_df("training_results_final_401_660")
df3["timestamps"] += df2["timestamps"].iloc[-1]
df = pd.concat([df1, df2, df3]).reset_index(drop=True)

df["steps"] = df["episode_lengths"].cumsum()
df.to_json("training_results_final.json")