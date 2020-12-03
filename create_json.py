import pandas as pd
import json
from glob import glob


def create_df(name):
    with open(glob(f"{name}/openaigym.episode_batch*")[0]) as f:
        gym_dict = json.load(f)
    gym_dict.pop("episode_types")
    initial_reset_timestamp = gym_dict.pop("initial_reset_timestamp")

    df = pd.DataFrame(gym_dict)
    df["timestamps"] = df["timestamps"] - initial_reset_timestamp
    return df

# Create json from recorded training results
df1 = create_df("training_results_final_0_356")[:351]
df2 = create_df("training_results_final_351_406")[:50]
df2["timestamps"] += df1["timestamps"].iloc[-1]
df3 = create_df("training_results_401_660")[:250]
df3["timestamps"] += df2["timestamps"].iloc[-1]
df = pd.concat([df1, df2, df3]).reset_index(drop=True)

df["steps"] = df["episode_lengths"].cumsum()
df.to_json("training_results_final.json")