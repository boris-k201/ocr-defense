import pickle

class Imdb_Dataloader:
    def __init__(self, file_path):
        """_summary_

        Args:
            file_path (_type_): _description_
        """
        self.file_path = file_path

    def load_data(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        with open(self.file_path, "rb") as f:
            dataset = pickle.load(f)
        
        return dataset
