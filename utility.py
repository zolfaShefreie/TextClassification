import os
import pandas as pd
import parsivar
import codecs
# from sets import Set
from hazm import Normalizer

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
                    text = preprocess_text(file.read())
                    print('hehe')
                    list_row.append({'category': each, 'text': text})
                    file.close()
    return pd.DataFrame(list_row)


def preprocess_text(text):
    text = parsivar.Normalizer().normalize(text)
    tokenizer = parsivar.Tokenizer()
    stop_words = get_stop_words()
    tokens = tokenizer.tokenize_words(text)
    new_text = str()
    for each in tokens:
        new_word = parsivar.FindStems().convert_to_stem(each)
        if '#' in new_word:
            new_word = new_word.split('#')[0]
        if new_word in stop_words:
            continue
        new_text += ' ' + new_word
    return new_text


def get_stop_words():
    nmz = Normalizer()
    stops = "\n".join(
        sorted(
            list(
                set(
                    [
                        nmz.normalize(w) for w in codecs.open('./persian', encoding='utf-8').read().split('\n') if w
                    ]
                )
            )
        )
    )
    return stops

print(get_df(TRAIN_PATH))
