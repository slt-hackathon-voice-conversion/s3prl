




from dataset import TorgoDataset
from dataset import VCC2020Dataset
from generate_directory_list import  generate_directory_uaspeech
from generate_directory_list import check_transcripts
from rename_datasets import spkr_transcript_matching
import pandas as pd


def test_torgo_data():
    torgo = TorgoDataset("test", "MC01", 1, 2)

    file_paths = torgo.get_all_file_paths()

    print(file_paths)
    print(len(file_paths))


def test_torgo_transcript():
    check_transcripts("./data/torgo/*/*0*/Session*/prompts/*.txt")
def test_many_to_one():
    """
    Test Torgo many to one function
    Returns:

    """

    df = pd.read_csv("transcripts.csv")

    train, test = spkr_transcript_matching("FC03", df, 2, "M01")
    print("\n")
    print(train.shape[0])
    print(test.shape[0])


def test_UAspeech_transcript_generator():
    generate_directory_uaspeech("./data/UASpeech/audio/original/*/*.wav", "./data/UASpeech/doc/speaker_wordlist.xls")


def test_many_to_one_UASpeech():
    df = pd.read_csv("UAspeech_transcripts.csv")
    train, test = spkr_transcript_matching("CM10", df, 2)



def main():
    generate_directory_uaspeech("./data/UASpeech/audio/original/*/*.wav", "./data/UASpeech/doc/speaker_wordlist.xls")


if __name__ == "__main__":
    main()


