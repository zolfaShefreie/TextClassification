import os
import pandas as pd
import parsivar
import re
import hazm

TRAIN_PATH = './Dataset/Train'
TEST_PATH = './Dataset/Test'


def get_df(path, clean_data=True):
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
                    text = file.read()
                    if clean_data:
                        text = preprocess_text(text)
                    list_row.append({'category': each, 'text': text})
                    file.close()
    return pd.DataFrame(list_row)


def preprocess_text(text):
    text = parsivar.Normalizer().normalize(text)
    tokenizer = parsivar.Tokenizer()
    stop_words = get_stop_words()
    chars = get_characters()
    chars = list(chars)
    chars += [';', ',', '&', '?', '.', '%']
    for each in chars:
        text = text.replace(each, ' ')
    tokens = tokenizer.tokenize_words(text)
    stemmer = parsivar.FindStems()
    new_text = str()
    lemmatizer = hazm.Lemmatizer()
    for each in tokens:
        new_word = stemmer.convert_to_stem(each)
        if '&' in new_word:
            new_word = new_word.split('&')[0]
        new_word = lemmatizer.lemmatize(new_word)
        if '#' in new_word:
            new_word = new_word.split('#')[0]
        if new_word in stop_words:
            continue
        if new_word in [';', ',', '&', '?', 'amp', 'nbsp', '.', 'o']:
            continue
        if re.match(r'\d+', new_word):
            continue
        new_text += ' ' + new_word
    new_text = new_text.replace('‌', '')
    return new_text


def get_characters():
    f = open('./chars', encoding='utf-8')
    chars = f.read()
    return set([w for w in chars.split('\n') if w])


def get_stop_words():
    nmz = parsivar.Normalizer()
    f = open('./persian', encoding='utf-8')
    words = f.read()
    # words = codecs.open('./persian', encoding='utf-8').read()
    return set([nmz.normalize(w) for w in words.split('\n') if w])


if __name__ == "__main__":
    pass


