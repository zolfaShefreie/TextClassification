import os
import pandas as pd

TRAIN_PATH = './Dataset/Train'
TEST_PATH = './Dataset/Test'


def get_df(path):
    list_row = list()
    list_of_file_dir = os.listdir(path)
    for each in list_of_file_dir:
        full_path = path + '/' + each
        if os.path.isdir(full_path):
            list_of_entry = os.listdir(full_path)
            for entry in list_of_entry:
                file_path = full_path + '/' + entry
                if not os.path.isdir(file_path):
                    file = open(file_path, encoding='utf8')
                    list_row.append({'category': each, 'text': file.read()})
                    file.close()
    return pd.DataFrame(list_row)


def preprocess_text(text):
    pass

