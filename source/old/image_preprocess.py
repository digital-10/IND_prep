# import os
import sys
from PIL import Image
# import numpy as np
import pandas as pd
# from keras.preprocessing.image import ImageDataGenerator, load_img
# from keras.utils import to_categorical
# from sklearn.model_selection import train_test_split

def read_config(file_name):
    import yaml
    try:
        with open(file_name, "r", encoding="utf-8") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            train_file = cfg['train_file']
            test_file = cfg['test_file']
            train_read_path = cfg['train_read_path']
            train_save_path = cfg['train_save_path']
            test_read_path = cfg['test_read_path']
            test_save_path = cfg['test_save_path']
            img_size = cfg['size']
            return train_file, test_file, train_read_path, train_save_path, test_read_path, test_save_path, img_size
    except Exception as e:
        print(e)
        return ''


def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


def make_resize(image, want_size=255):
    return image.resize((want_size, want_size))


if __name__ == '__main__':
    config_file_name = sys.argv[1]
    print('config_file_name', config_file_name)

    train_file, test_file, train_read_path, train_save_path, test_read_path, test_save_path, img_size = read_config(config_file_name)
    train_df = pd.read_csv(train_file)
    for index, row in train_df.iterrows():
        img = Image.open(rf'{train_read_path}\{row[0]}')
        new_image = make_square(img)
        new_image = make_resize(new_image, img_size)
        if row[0].split('.')[-1] == 'jpg':
            row[0] = row[0].replace('jpg', 'png')
        new_image.save(rf'{train_save_path}\{row[0]}')

    test_df = pd.read_csv(test_file)
    for index, row in test_df.iterrows():
        img = Image.open(rf'{test_read_path}\{row[0]}')
        new_image = make_square(img)
        new_image = make_resize(new_image, img_size)
        if row[0].split('.')[-1] == 'jpg':
            row[0] = row[0].replace('jpg', 'png')
        new_image.save(rf'{test_save_path}\{row[0]}')
