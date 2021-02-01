class Config():

    def __init__(self):
        self.data_folder = '../data/'   # Relative to the src folder
        self.device = 'cpu'
        self.max_len = 10
        self.batch_size = 128
        self.epochs = 200
        self.learning_rate = 0.02
        self.log_folder = '../out/'
        self.runs = 1


config = Config()
