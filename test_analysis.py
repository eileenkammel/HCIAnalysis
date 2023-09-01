import pytest
import pandas as pd
from analysis import *


@pytest.fixture(scope="module")
def test_df():
    return pd.read_csv("Transcripts/p2_transcript.csv", sep=",", header=0)


class TestAnalysis:

    def test_count_words(self, test_df):
        assert count_words(test_df) == 202

    def test_count_participant_utterances(self, test_df):
        assert count_participant_utterances(test_df) == 55

    def test_count_furhat_utterances(self, test_df):
        assert count_furhat_utterances(test_df) == 43

    def test_count_disfluencies(self, test_df):
        assert count_disfluencies(test_df) == 14

    def test_get_disfluency_distribution(self, test_df):
        assert get_disfluency_distribution(
            test_df) == {'FP': 8, 'SUB': 2, 'UP': 3, 'REP': 1}

    def test_get_mean_words_per_utterance(self, test_df):
        words = count_words(test_df)
        utterances = count_participant_utterances(test_df)
        assert get_mean_words_per_utterance(words, utterances) == 3.673

    def test_get_mean_dfs_per_utterance(self, test_df):
        disfluencies = count_disfluencies(test_df)
        utterances = count_participant_utterances(test_df)
        assert get_mean_dfs_per_utterance(disfluencies, utterances) == 0.255

    def test_get_mean_dfs_per_word(self, test_df):
        disfluencies = count_disfluencies(test_df)
        words = count_words(test_df)
        assert get_mean_dfs_per_word(disfluencies, words) == 0.069
