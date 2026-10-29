import numpy as np
import pandas as pd
Base = ["ping", "kills", "assists","deaths", "mvps", "hs_percent", "points"]
Maps = ["Dust II", "Mirage", "Inferno"]
Map_cols = [f"map_{m}" for m in Maps + ["Other"]]
ENGINEERED = ["kd", "kills_minus_deaths", "assists_per_death",
              "points_per_kill", "log_ping", "hs_kills"]
Features = Base + ENGINEERED + Map_cols
def make_features(df):
    out = df[Base].copy()
    deaths =  out["deaths"].clip(lower=1)
    kills = out["kills"].clip(lower=1)

    out["kd"] = out["kills"] / deaths
    out["kills_minus_deaths"] = out["kills"] - out["deaths"]
    out["assists_per_death"] = out["assists"] / deaths
    out["points_per_kill"] = out["points"] / kills
    out["log_ping"] = np.log1p(out["ping"])
    out["hs_kills"] = out["hs_percent"] * out["kills"] / 100
    grouped = df["map"].where(df["map"].isin(Maps),"Other")
    for m in Maps + ["Other"]:
        out[f"map_{m}"] = (grouped == m).astype(int)
    return out[Features]
def load_data(path="csgo.csv"):
    d = pd.read_csv(path)
    d = d[d["result"] != "Tie"].copy()
    d["win"] = (d["result"] == "Win").astype(int)
    d["date_dt"] = pd.to_datetime(dict(year=d["year"].astype(int),
                                       month=d["month"].astype(int),
                                       day=d["day"].astype(int)))
    return d.reset_index(drop=True)