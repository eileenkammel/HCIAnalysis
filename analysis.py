import pandas as pd
import nltk
import os


def count_participant_utterances(transcript_df):
    participant_utterances = transcript_df["speaker"].value_counts()[
        "Participant"]
    return participant_utterances


def count_furhat_utterances(transcript_df):
    furhat_utterances = transcript_df["speaker"].value_counts()["Furhat"]
    return furhat_utterances


def count_disfluencies(transcript_df):
    disfluencies = transcript_df["speaker"].value_counts()["DF"]
    return disfluencies


def get_disfluency_distribution(transcript_df):
    all_disfluencies = transcript_df.loc[transcript_df["speaker"] == "DF", "text"].tolist(
    )
    disfluencies_histo = dict(nltk.FreqDist(all_disfluencies))

    return disfluencies_histo


def split_df_histo(disfluencies_histo):
    fp = disfluencies_histo["FP"] if "FP" in disfluencies_histo else 0
    sub = disfluencies_histo["SUB"] if "SUB" in disfluencies_histo else 0
    up = disfluencies_histo["UP"] if "UP" in disfluencies_histo else 0
    rep = disfluencies_histo["REP"] if "REP" in disfluencies_histo else 0
    art = disfluencies_histo["ART"] if "ART" in disfluencies_histo else 0
    cd = disfluencies_histo["CD"] if "CD" in disfluencies_histo else 0
    return fp, up, rep, sub, art, cd


def count_words(transcript_df):
    participant_words = transcript_df.loc[transcript_df["speaker"] ==
                                          "Participant", "text"].str.split().apply(lambda x: len(x)).sum()
    return participant_words


def get_mean_words_per_utterance(word_count, utterance_count):
    return round((word_count / utterance_count), 3)


def get_mean_dfs_per_utterance(df_count, utterance_count):
    return round((df_count / utterance_count), 3)


def get_mean_dfs_per_word(df_count, word_count):
    return round((df_count / word_count), 3)


def get_mean_time_btw_turns(transcript_df):
    mean_turn_change_time = transcript_df.loc[transcript_df["speaker"]
                                              == "Silence", "duration"].mean()
    return mean_turn_change_time


def get_dialogue_length(transcript_df):
    # substract start time because dialogues do not
    # start at 0:00:00
    return transcript_df["end"].max() - transcript_df["start"].min()


def total_talk_duration(transcript_df, participant_no):
    participant_talk = transcript_df.loc[transcript_df["speaker"]
                                         == "Participant", "duration"].sum()
    furhat_talk = transcript_df.loc[transcript_df["speaker"]
                                    == "Furhat", "duration"].sum()
    return {
        "participant_no": participant_no,
        "participant_talk": participant_talk,
        "furhat_talk": furhat_talk
    }


def get_overlap(transcript_df):
    overlap = transcript_df["speaker"].value_counts()["Overlap"]
    return overlap


def add_mean_row(transcript_df):
    mean_row = transcript_df.mean()
    mean_row["Participant"] = "Mean"
    transcript_df = pd.concat([transcript_df, pd.DataFrame([mean_row])])
    return transcript_df


# Wrapper function for all general stats
def analyze_general(transcript_df, participant_no):
    participant_no = participant_no
    dia_len = get_dialogue_length(transcript_df)
    utterances_p = count_participant_utterances(transcript_df)
    utterances_f = count_furhat_utterances(transcript_df)
    words = count_words(transcript_df)
    mean_words_per_utterance = get_mean_words_per_utterance(
        words, utterances_p)
    disfluencies = count_disfluencies(transcript_df)
    mean_sil_btw_turns = get_mean_time_btw_turns(transcript_df)
    overlap = get_overlap(transcript_df)

    return {
        "Participant": participant_no,
        "Lenght": dia_len,
        "Utts P ": utterances_p,
        "Utts F": utterances_f,
        "Overlap": overlap,
        "Words": words,
        "Mean Words per Utt": mean_words_per_utterance,
        "DF": disfluencies,
        "Mean SIL between Turns": mean_sil_btw_turns
    }


# Get general dialogue stats. One output file for all participants
def get_general_stats(path_to_trancripts):
    gen_stats = pd.DataFrame(columns=["Participant", "Lenght", "Utts P ",
                                      "Utts F", "Overlap", "Words", "Mean Words per Utt", "DF", "Mean SIL between Turns"])
    for participant in range(2, 8):
        path = os.path.join(path_to_trancripts, "p" +
                            str(participant)+"_transcript.csv")
        transcript_df = pd.read_csv(path, sep=",", header=0)
        transcript_df["start"] = pd.to_timedelta(transcript_df["start"])
        transcript_df["end"] = pd.to_timedelta(transcript_df["end"])
        transcript_df["duration"] = pd.to_timedelta(transcript_df["duration"])
        row = analyze_general(transcript_df, participant)
        gen_stats = pd.concat([gen_stats, pd.DataFrame([row])])
    gen_stats.sort_values(by="Participant", inplace=True)
    gen_stats.reset_index(drop=True, inplace=True)
    gen_stats = add_mean_row(gen_stats)
    gen_stats.to_csv("general_stats.csv", index=False)


# Wrapper function for all disfluency stats
def analyze_disfluencies(transcript_df, participant_no):
    participant_no = participant_no
    words = count_words(transcript_df)
    utterances = count_participant_utterances(transcript_df)
    total_df = count_disfluencies(transcript_df)
    mean_per_word = get_mean_dfs_per_utterance(total_df, words)
    mean_per_utterance = get_mean_dfs_per_word(total_df, utterances)
    utt_histo = get_disfluency_distribution(transcript_df)
    fp, up, rep, sub, art, cd = split_df_histo(utt_histo)
    return {
        "Participant": participant_no,
        "total_df": total_df,
        "mean_per_word": mean_per_word,
        "mean_per_utterance": mean_per_utterance,
        "fp": fp,
        "up": up,
        "rep": rep,
        "sub": sub,
        "art": art,
        "cd": cd
    }


# Get general disfluency stats. One output file for all participants
def get_disfluency_stats(path_to_trancripts):
    df_stats = pd.DataFrame(columns=["Participant", "total_df", "mean_per_word",
                                     "mean_per_utterance", "fp", "up", "rep", "sub", "art", "cd"])
    for participant in range(2, 8):
        path = os.path.join(path_to_trancripts, "p" +
                            str(participant)+"_transcript.csv")
        transcript_df = pd.read_csv(path, sep=",", header=0)
        row = analyze_disfluencies(transcript_df, participant)
        df_stats = pd.concat([df_stats, pd.DataFrame([row])])
    df_stats.sort_values(by="Participant", inplace=True)
    df_stats.reset_index(drop=True, inplace=True)
    df_stats.to_csv("disfluency_stats.csv", index=False)


# While annotating, I noticed that a lot of filled pauses
# occur right at the beginning of utterances. Times of the transcriptions
# and the DF labels on the DF tier were carfully alinged for that matter.
# DF is filtered for that alignment.
def find_start_disfluencies(transcript_df):
    speaker_df = transcript_df[(transcript_df["speaker"] == "Participant") | (
        transcript_df["speaker"] == "DF")]
    # find start values that occur more than once
    start_df = speaker_df[speaker_df.duplicated(["start"], keep=False)]
    other_df = speaker_df.drop_duplicates(["start"], keep=False)
    return start_df, other_df


# Write start disfluencies to one file, all other utterances to another file.
# Two files per participant.
def get_start_disfluencies(path_to_trancripts):
    for participant in range(2, 8):
        path = os.path.join(path_to_trancripts, "p" +
                            str(participant)+"_transcript.csv")
        transcript_df = pd.read_csv(path, sep=",", header=0)
        start_df, other_df = find_start_disfluencies(transcript_df)
        out_path = "Disfluency/"
        start_out = os.path.join(
            out_path, "p" + str(participant)+"_start_df.csv")
        other_out = os.path.join(
            out_path, "p" + str(participant)+"_other_df.csv")
        start_df.to_csv(start_out, index=False)
        other_df.to_csv(other_out, index=False)


# Get absolut frequency of start disfluencies
def get_freq_of_start_disfluencies(path_to_transcripts):
    df = pd.DataFrame(columns=["Participant", "FP", "UP", "REP", "SUB", "ART", "CD"])
    for participant in range(2, 8):
        path = os.path.join(path_to_transcripts, "p" +
                            str(participant)+"_start_df.csv")
        transcript_df = pd.read_csv(path, sep=",", header=0)

        try:
            fp_val = transcript_df["text"].value_counts()["FP"]
        except KeyError:
            fp_val = 0

        try:
            up_val = transcript_df["text"].value_counts()["UP"]
        except KeyError:
            up_val = 0

        try:
            rep_val = transcript_df["text"].value_counts()["REP"]
        except KeyError:
            rep_val = 0

        try:
            sub_val = transcript_df["text"].value_counts()["SUB"]
        except KeyError:
            sub_val = 0

        try:
            art_val = transcript_df["text"].value_counts()["ART"]
        except KeyError:
            art_val = 0

        try:
            cd_val = transcript_df["text"].value_counts()["CD"]
        except KeyError:
            cd_val = 0

        row = {"Participant": participant, "FP": fp_val, "UP": up_val, "REP": rep_val, "SUB": sub_val, "ART": art_val, "CD": cd_val}
        df = pd.concat([df, pd.DataFrame([row])])
    df.sort_values(by="Participant", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv("Disfluency/start_df_freq.csv", index=False)

if __name__ == "__main__":

    # Calls are kept separate in order to keep
    # different analysis scripts separate


    # Get general dialogue stats. One output file for all participants
    # get_general_stats("Transcripts")


    # Get general disfluency stats. One output file for all participants
    # get_disfluency_stats("Transcripts")


    # Split DF into two dfs: one with start disfluencies and one with
    # all other utterances. Save to two files per participant.
    # get_start_disfluencies("Transcripts")


    # Get frequency of start disfluencies
    #get_freq_of_start_disfluencies("Disfluency")
