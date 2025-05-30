import pandas as pd

df = pd.read_csv("H3_labeled_with_host.csv")

class0 = df[df["binding_score"] == 0].sample(n=400, random_state=42)
class1 = df[df["binding_score"] == 1].sample(n=400, random_state=42)
class2 = df[df["binding_score"] == 2].sample(n=400, random_state=42)  

balanced_df = pd.concat([class0, class1, class2]).sample(frac=1, random_state=42)

balanced_df.to_csv("H3_balanced_training_set_v3.csv", index=False)
print("✅ Sample distribution：")
print(balanced_df["binding_score"].value_counts())
