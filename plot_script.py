import pandas as pd
import matplotlib.pyplot as plt

# Script for generating tables and plots for the report


df = pd.read_csv("Data/general_stats.csv", sep=",", header=0)

df.to_latex("Figures/general_stats.tex", index=False)

df_disfluencies = pd.read_csv("Disfluency/disfluency_stats.csv", sep=",", header=0)
print(df_disfluencies)


# drop cols not needed for table and save as latex table
df_disfluency_stats = df_disfluencies.drop(
    ["fp", "up", "art", "sub", "rep", "cd"], axis=1)
df_disfluency_stats.to_latex("Figures/disfluency_stats.tex", index=False)

# plot df type frequencies as stacked bar plot
# drop columns that are not needed for the plot
df_disfluencies.drop(["total_df", "mean_per_word",
                     "mean_per_utterance"], axis=1, inplace=True)
df_disfluencies.set_index("Participant", inplace=True)
df_disfluencies.plot(kind='bar', stacked=True,
                     colormap="BuPu", edgecolor="black", rot=0)
plt.xlabel("Participant")
plt.ylabel("Number of disfluencies")
plt.savefig("Figures/disfluency_freq.png", bbox_inches="tight")
plt.show()

# Compute relative frequency of fp and save as latex table
df_disfluencies["fp_rel"] = df_disfluencies["fp"] / df_disfluencies["total_df"]
df_disfluencies["fp_rel"] = df_disfluencies["fp_rel"].round(2)

# Only keep Participant and fp_rel
df_disfluencies = df_disfluencies[["Participant", "fp_rel"]]
df_disfluencies.set_index("Participant", inplace=True)
# transpose df
df_disfluencies = df_disfluencies.T
df_disfluencies.to_latex("Figures/fp_rel.tex", index=True)

# Generate start disfluency table and save as latex table
df_start = pd.read_csv("Disfluency/start_df_freq.csv", sep=",", header=0)
df_start.to_latex("Figures/start_df_freq.tex", index=False)
