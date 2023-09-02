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


def analyze_transcript(transcript_df, participant_no):
    participant_no = participant_no
    words = count_words(transcript_df)
    utterances = count_participant_utterances(transcript_df)
    disfluencies = count_disfluencies(transcript_df)
    disfluencies_histo = get_disfluency_distribution(transcript_df)
    mean_words_per_utterance = get_mean_words_per_utterance(
        words, utterances)
    mean_dfs_per_utterance = get_mean_dfs_per_utterance(
        disfluencies, utterances)
    mean_dfs_per_word = get_mean_dfs_per_word(disfluencies, words)

    return {
        "participant_no": participant_no,
        "words": words,
        "utterances": utterances,
        "disfluencies": disfluencies,
        "disfluencies_histo": disfluencies_histo,
        "mean_words_per_utterance": mean_words_per_utterance,
        "mean_dfs_per_utterance": mean_dfs_per_utterance,
        "mean_dfs_per_word": mean_dfs_per_word
    }


def analize_all(path_to_trancripts):
    results_df = pd.DataFrame(columns=["participant_no", "words", "utterances", "disfluencies",
                              "disfluencies_histo", "mean_words_per_utterance", "mean_dfs_per_utterance", "mean_dfs_per_word"])
    total_talk_duration = pd.DataFrame(
        columns=["participant_no", "participant_talk", "furhat_talk"])
    for participant in range(2, 8):

        path = os.path.join(path_to_trancripts, "p" +
                            str(participant)+"_transcript.csv")
        transcript_df = pd.read_csv(path, sep=",", header=0)
        transcript_df["start"] = pd.to_timedelta(transcript_df["start"])
        transcript_df["end"] = pd.to_timedelta(transcript_df["end"])
        transcript_df["duration"] = pd.to_timedelta(transcript_df["duration"])
        row = analyze_transcript(transcript_df, participant)
        results_df = results_df.append(row, ignore_index=True)
        row = total_talk_duration(transcript_df, participant)
        total_talk_duration = total_talk_duration.append(
            row, ignore_index=True)
    results_df.sort_values(by="participant_no", inplace=True)
    results_df.reset_index(drop=True, inplace=True)
    results_df.to_csv("results.csv", index=False)
    total_talk_duration.sort_values(by="participant_no", inplace=True)
    total_talk_duration.reset_index(drop=True, inplace=True)
    total_talk_duration.to_csv("total_talk_duration.csv", index=False)


analize_all("/Users/eileen/HCIAnalysis/Transcripts")
