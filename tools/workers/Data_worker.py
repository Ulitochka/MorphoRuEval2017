# -*- coding: utf-8 -*-
#
# Скрипт ориентирован на использование в Python 3.*
#
# Скрипт содержит в себе класс DataPreparator для загрузки корпуса, сборка статистика, извлечения признаков.

import os
import pickle
from pickle import dumps
import pprint
import distutils.dir_util

from tools.workers.Stat_worker import StatWorker
from tools.workers.Binarizator import Binarizator
from tools.workers.Config_worker import Config
from tools.workers.Feature_extractor import FeatureExtractor
from tools.workers.Dimensionality_reductor import SVDReductor
from tools.parsers.Parser import MorphoParser
import csv


class DataPreparator:
    """
    Класс для загрузки данных, парсинга, извлечения признаков.
    """

    def __init__(self):

        translate_config_options = {
            "True": True,
            "False": False
        }

        self.config = Config()
        self.data_type = self.config.get("FORMAT")
        self.corpora_name = self.config.get("CORPORA")
        self.all_data = translate_config_options.get(self.config.get("X_Y"))
        self.spec_labels = self.config.get("POS_X")

        label_options_from_config = self.config.get("LABEL")
        data_type = self.config.get("FORMAT")
        binarization = translate_config_options.get(self.config.get("BINARY"))
        dim_reduction = translate_config_options.get(self.config.get("DIM_RED"))
        data_mark = 'type_%s_bin_%s_dimred_%s' % (data_type, binarization, dim_reduction)

        file_path = os.path.split(os.path.abspath(__file__))[0]
        self.data_path = os.path.abspath(
            file_path + '/../../data/' + self.corpora_name + '/')

        labels = None
        if label_options_from_config != 'False':
            labels = [label_options_from_config]
        else:
            labels = self.load_list_grammar_cat()

        print('\nData preparation')
        pprint.pprint(self.config.get('FEATURES_OPTIONS'))
        print('Labels:', labels)
        print('Corpora:', self.corpora_name)

        self.morpho_parser = MorphoParser()

        for label_ in labels:

            self.data_sets_x = {
                'x_train': None,
                'x_test': None,
                'x_dev': None
            }

            self.data_sets_y = {
                'y_train': None,
                'y_test': None,
                'y_dev': None
            }

            self.form_sets(label_, self.config)
            self.data_path = self.data_path + '/%s/' % label_ + data_mark
            features_names = None

            if data_type == 'dict':
                if self.all_data is True:
                    features_names = [el for el in self.data_sets_x['x_train'][0][0]]
            else:
                if data_type == 'list':
                    if self.all_data is True:
                        # массив с именами признаков;
                        features_names = list(self.data_sets_x['x_train'].columns)

            if binarization:
                binarizator = Binarizator()
                self.data_sets_x['x_train'], self.data_sets_x['x_test'], self.data_sets_x['x_dev'] = binarizator.transform(
                    self.data_sets_x['x_train'], self.data_sets_x['x_test'], self.data_sets_x['x_dev'])

            if dim_reduction:
                svd_reductor = SVDReductor(svd_options=self.config.get("SVD"), mode='TruncatedSVD')
                self.data_sets_x['x_train'], self.data_sets_x['x_test'], self.data_sets_x['x_dev'] = svd_reductor.dimension_reduction(
                    self.data_sets_x['x_train'], self.data_sets_x['x_test'], self.data_sets_x['x_dev'])

            for x_data_sets in self.data_sets_x:
                if data_type == 'dict' and self.all_data is True:
                    self.save_binary_serialize(self.data_sets_x[x_data_sets], x_data_sets)
                else:
                    if self.all_data is True:
                        self.save_binary(self.data_sets_x[x_data_sets], x_data_sets)

            for y_data_sets in self.data_sets_y:
                self.save_binary(self.data_sets_y[y_data_sets], y_data_sets)

            if features_names:
                self.save_in_csv_format(features_names, 'features', ['features'])
            self.data_path = os.path.abspath(
                file_path + '/../../data/'+ self.corpora_name + '/')

    def save_in_csv_format(self, data, file_name, columns_names):
        """
        Сохранить данные в формате csv.
        :param columns_names:
        :param data:
        :param file_name:
        :return:
        """

        file_name = self.data_path + '/%s.csv' % file_name
        with open(file_name, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',',  quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(columns_names)
            for el in data:
                writer.writerow([el])

    def save_binary(self, data, file_name):
        """
        Сохранение данных в бинарном формате.
        :param data:
        :param file_name:
        :return:
        """

        distutils.dir_util.mkpath(self.data_path)
        with open(self.data_path + '/%s.pkl' % file_name, 'wb') as file:
            pickle.dump(data, file)

    def save_binary_serialize(self, data, file_name):
        """
        Сохранить в бинарном формате без загрузки всех данных в оперативную память.
        :param data:
        :param file_name:
        :return:
        """

        bin_records = list()
        distutils.dir_util.mkpath(self.data_path)
        file = open(self.data_path + '/%s.pkl' % file_name, "wb")
        for elements in data:
            pickled_elt_str = dumps(elements)
            x = file.write(pickled_elt_str)
            bin_records.append(x)
        file.close()
        self.save_binary(bin_records, 'bin_records_for_%s' % file_name)

    def get_label_position(self, label):
        """
        Индексы элементов для парсинга.
        :param label:
        :return:
        """

        if 'POS' in label:
            return self.config.get("LABELS_OPTIONS").get(self.corpora_name).get(label)
        else:
            return self.config.get("LABELS_OPTIONS").get(self.corpora_name).get("GRAMMAR_CAT")

    def form_sets(self, label, features):
        """
        Извлеченеи признаков.
        :param label:
        :param features:
        :return:
        """

        file_to_parse = ['train', 'test', 'dev']
        feature_extractor = FeatureExtractor(config=self.config, data_type=self.data_type, spec_labels=self.spec_labels)
        labels_format = self.get_label_position(label)
        data_sets = {file: self.morpho_parser.parse(self.open_text(file), labels_format, label) for file in file_to_parse}

        # вывод статистики;
        for data_set in data_sets:
            print('\n', data_set)
            stat = StatWorker(features=self.config.get('FEATURES_OPTIONS'))
            # костыль убрать;
            stat.count_elements(data=data_sets[data_set])
            stat.create_report()

        # выделение признаков, формирование множеств;
        if self.all_data is True:
            print('\n', '*' * 50, 'Label:', label, '*' * 50)
            print('feature_extractor starts ...')
            self.data_sets_x['x_train'] = feature_extractor.sentence2features(data_sets['train'])
            self.data_sets_y['y_train'] = feature_extractor.sentence2labels(data_sets['train'])
            print('x_train, y_train done.')
            self.data_sets_x['x_dev'] = feature_extractor.sentence2features(data_sets['dev'])
            self.data_sets_y['y_dev'] = feature_extractor.sentence2labels(data_sets['dev'])
            print('x_dev, y_dev done.')
            self.data_sets_x['x_test'] = feature_extractor.sentence2features(data_sets['test'])
            self.data_sets_y['y_test'] = feature_extractor.sentence2labels(data_sets['test'])
            print('x_test, y_test done.')
        else:
            # работа только с метками классов, т.к. признаки будут одинаковыми для всех классификаторов;
            print('\n', '*' * 50, 'Label:', label, '*' * 50)
            print('feature_extractor starts ...')
            self.data_sets_y['y_train'] = feature_extractor.sentence2labels(data_sets['train'])
            print('y_train done.')
            self.data_sets_y['y_dev'] = feature_extractor.sentence2labels(data_sets['dev'])
            print('y_dev done.')
            self.data_sets_y['y_test'] = feature_extractor.sentence2labels(data_sets['test'])
            print('y_test done.')
        print('feature_extractor done.')

    def open_text(self, corpora_name=None):
        """
        Загрузка текстового файла.
        :return:
        """

        corpora_file = self.data_path + '/' + corpora_name
        print('Loading file:', corpora_file)
        with open(corpora_file, 'r', encoding='utf-8') as train_text:
            train_text = train_text.read()
            sentences = train_text.split('\n\n')
            return sentences

    def load_list_grammar_cat(self):
        """
        Загрузка списка грамматических категорий.
        :return:
        """

        file = self.data_path + '/grammar_catigories.txt'
        with open(file, 'r', encoding='utf-8') as text:
            text = text.read()
            sub_tags = text.split('\n')
            return [el for el in sub_tags if el != '']

if __name__ == '__main__':
    DataPreparator()
