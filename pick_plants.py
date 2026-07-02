import pandas as pd

df = pd.read_csv("data/powerplants.csv")
india_coal = df[(df["country"] == "IND") & (df["primary_fuel"] == "Coal")]
top5 = india_coal.sort_values("capacity_mw", ascending=False).head(5)

cols = ["name", "capacity_mw", "latitude", "longitude"]
top5[cols].to_csv("data/top5_plants.csv", index=False)
print(top5[cols].to_string(index=False))
print("\nSaved data/top5_plants.csv")