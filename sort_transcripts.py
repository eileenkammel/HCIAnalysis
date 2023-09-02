import pandas as pd
import os

# Couldn't find a way to export transcripts
# sorted by time stamps directly from Elan
# This script sorts the transcripts by time stamps
# and overwrites the unsorted files
# Result: Utterances are chronologically ordered in the csv files


path_base = "/Users/eileen/HCIAnalysis/Transcripts"

for file in os.listdir(path_base):
    # check if file is a csv file, just in case
    if file.endswith(".csv"):
        path = os.path.join(path_base, file)
        # read transcript from file , add header
        df = pd.read_csv(path, sep=",", header=None, names=[
                         "speaker", "start", "end", "duration", "text"])
        # sort df by column "start"
        df.sort_values(by="start", inplace=True)
        # reset index
        df.reset_index(drop=True, inplace=True)
        # convert time stamps to pd timedelta objects
        df["start"] = pd.to_timedelta(df["start"])
        df["end"] = pd.to_timedelta(df["end"])
        df["duration"] = pd.to_timedelta(df["duration"])
        # save sorted df to file (overwrite unsorted file)
        df.to_csv(path, index=False)
