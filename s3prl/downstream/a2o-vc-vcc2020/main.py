




from dataset import TorgoDataset
from dataset import VCC2020Dataset


def test_vcc_data():
    # FBANK is not an int
    vcc_data = VCC2020Dataset("train", "TEF2",
                              "/Users/ianyip/Documents/Coding/s3prl/s3prl/downstream/a2o-vc-vcc2020/data/vcc2020",
                              "/Users/ianyip/Documents/Coding/s3prl/s3prl/downstream/a2o-vc-vcc2020/data/lists",
                              1)
    x = vcc_data[0]

    print(x)
def test_torgo_data():
    torgo = TorgoDataset("test", "MC01", 1, 2)

    file_paths = torgo.get_all_file_paths()

    print(file_paths)
    print(len(file_paths))

def main():

    print("hello world")





if __name__ == "__main__":
    main()


