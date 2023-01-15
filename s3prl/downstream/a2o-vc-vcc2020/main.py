




from dataset import TorgoDataset



def main():

    torgo = TorgoDataset("train", "MC01", 1, 2)

    file_paths = torgo.get_all_file_paths()

    print(file_paths)
    print(len(file_paths))


if __name__ == "__main__":
    main()


