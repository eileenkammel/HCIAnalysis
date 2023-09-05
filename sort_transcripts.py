import pandas as pd
import os

# Couldn't find a way to export transcripts
# sorted by time stamps directly from Elan
# This script sorts the transcripts by time stamps
# and overwrites the unsorted files
# Result: Utterances are chronologically ordered in the csv files


path_base = "/Users/eileen/HCIAnalysis/Transcripts"


def sort_transcripts(path_base):
    # iterate over all files in path
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
            # save sorted df to file (overwrite unsorted file)
            df.to_csv(path, index=False)

# Concatenate all transcripts into one file, add a column for participant number
# and save to file
# Result: One csv file with all transcripts, sorted by time stamps
# and with a column for participant number

# create empty df


def concat_all_transcripts(path_base):
    df_all = pd.DataFrame(columns=["Participant No", "speaker", "start",
                                    "end", "duration", "text"])
    # iterate over all files in path
    for participant in range(2, 8):
        path = os.path.join(path_base, "p" +
                            str(participant)+"_transcript.csv")
        df = pd.read_csv(path, sep=",", header=0)
        # extract participant number from file name
        # add column with participant number
        df["Participant No"] = participant
        # concat df to df_all
        df_all = pd.concat([df_all, df])

    # save df_all to file
    df_all.to_csv("Transcripts/all_transcripts.csv", index=False)

concat_all_transcripts(path_base)