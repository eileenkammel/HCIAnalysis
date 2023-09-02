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
    # substract start time because dialogues does not
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


def analyze_general(transcript_df, participant_no):
    participant_no = participant_no
    dia_len = get_dialogue_length(transcript_df)
    utterances_p = count_participant_utterances(transcript_df)
    utterances_f = count_furhat_utterances(transcript_df)
    utterances = str(utterances_p) + "/" + str(utterances_f)
    words = count_words(transcript_df)
    mean_words_per_utterance = get_mean_words_per_utterance(
        words, utterances_p)
    disfluencies = count_disfluencies(transcript_df)
    mean_sil_btw_turns = get_mean_time_btw_turns(transcript_df)

    return {
        "participant_no": participant_no,
        "dialogue lenght": dia_len,
        "utterances (p/f)": utterances,
        "words": words,
        "mean words per utterance (p)": mean_words_per_utterance,
        "disfluencies": disfluencies,
        "mean silence between turns": mean_sil_btw_turns
    }


def general_stats(path_to_trancripts):
    gen_stats = pd.DataFrame(columns=["participant_no", "dialogue lenght", "utterances (p/f)",
                             "words", "mean words per utterance (p)", "disfluencies", "mean silence between turns"])
    for participant in range(2, 8):
        path = os.path.join(path_to_trancripts, "p" +
                            str(participant)+"_transcript.csv")
        transcript_df = pd.read_csv(path, sep=",", header=0)
        transcript_df["start"] = pd.to_timedelta(transcript_df["start"])
        transcript_df["end"] = pd.to_timedelta(transcript_df["end"])
        transcript_df["duration"] = pd.to_timedelta(transcript_df["duration"])
        row = analyze_general(transcript_df, participant)
        gen_stats = pd.concat([gen_stats, pd.DataFrame([row])])
    gen_stats.sort_values(by="participant_no", inplace=True)
    gen_stats.reset_index(drop=True, inplace=True)
    gen_stats.to_csv("general_stats.csv", index=False)


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
        "participant_no": participant_no,
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


def disfluency_stats(path_to_trancripts):
    df_stats = pd.DataFrame(columns=["participant_no", "total_df", "mean_per_word",
                             "mean_per_utterance", "fp", "up", "rep", "sub", "art", "cd"])
    for participant in range(2, 8):
        path = os.path.join(path_to_trancripts, "p" +
                            str(participant)+"_transcript.csv")
        transcript_df = pd.read_csv(path, sep=",", header=0)
        row = analyze_disfluencies(transcript_df, participant)
        df_stats = pd.concat([df_stats, pd.DataFrame([row])])
    df_stats.sort_values(by="participant_no", inplace=True)
    df_stats.reset_index(drop=True, inplace=True)
    df_stats.to_csv("disfluency_stats.csv", index=False)


def analize_all(path_to_trancripts):
    results_df = pd.DataFrame(columns=["participant_no", "words", "utterances", "disfluencies",
                              "disfluencies_histo", "mean_words_per_utterance", "mean_dfs_per_utterance", "mean_dfs_per_word"])
    total_talk = pd.DataFrame(
        columns=["participant_no", "participant_talk", "furhat_talk"])
    for participant in range(2, 8):

        path = os.path.join(path_to_trancripts, "p" +
                            str(participant)+"_transcript.csv")
        transcript_df = pd.read_csv(path, sep=",", header=0)
        transcript_df["start"] = pd.to_timedelta(transcript_df["start"])
        transcript_df["end"] = pd.to_timedelta(transcript_df["end"])
        transcript_df["duration"] = pd.to_timedelta(transcript_df["duration"])
        row = analyze_general(transcript_df, participant)
        results_df = pd.concat([results_df, pd.DataFrame([row])])
        row = total_talk_duration(transcript_df, participant)
        total_talk = pd.concat(
            [total_talk, pd.DataFrame([row])])
    results_df.sort_values(by="participant_no", inplace=True)
    results_df.reset_index(drop=True, inplace=True)
    results_df.to_csv("results.csv", index=False)
    total_talk.sort_values(by="participant_no", inplace=True)
    total_talk.reset_index(drop=True, inplace=True)
    total_talk.to_csv("total_talk_duration.csv", index=False)


# analize_all("/Users/eileen/HCIAnalysis/Transcripts")
# general_stats("/Users/eileen/HCIAnalysis/Transcripts")
disfluency_stats("/Users/eileen/HCIAnalysis/Transcripts")
