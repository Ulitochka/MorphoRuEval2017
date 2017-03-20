# -*- coding: utf-8 -*-
#
# Скрипт ориентирован на использование в Python 3.*
#
# Скрипт содержит в себе класс ExperimentManager.

import os
import pickle
from pickle import loads
import pprint
import distutils.dir_util

import numpy
import lasagne

from tools.workers.Config_worker import Config
from tools.algorithms.CRF import CRF
from tools.algorithms.MLP_theano import MLPT
from tools.algorithms.LSTM_keras import LongShortTermMemory
from tools.algorithms.Logistic_regression import LogRegression


class ModelWorker:
    def __init__(self):

        translate_config_options = {
            "True": True,
            "False": False,
            'CRF': CRF,
            'LOGREG': LogRegression,
            'MLPT': MLPT,
            'LSTM': LongShortTermMemory
        }

        self.config = Config()
        self.data_type = self.config.get("FORMAT")
        self.save_sent_borders = False
        self.classifier = self.config.get("ALGORITHM")
        self.corpora_name = self.config.get("CORPORA")

        data_type = self.config.get("FORMAT")
        label_options_from_config = self.config.get("LABEL")
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
            labels = self.load_list_of_sub_tags()

        print('Labels:', labels)
        print('Model:', self.classifier)
        print('Corpora:', self.corpora_name)

        for label_ in labels:
            print('\n', '*' * 50, 'Label:', label_, '*' * 50)

            self.data_sets_x = [
                ['x_train', None],
                ['x_test', None],
                ['x_dev', None]
            ]

            self.data_sets_y = [
                ['y_train', None],
                ['y_test', None],
                ['y_dev', None]
            ]

            self.find_files(label_, data_mark)

            for index in range(0, len(self.data_sets_x)):
                if data_type == 'dict':
                    self.data_sets_x[index][1] = self.load_data(self.data_sets_x[index][1])
                else:
                    self.data_sets_x[index][1] = self.load_binary(self.data_sets_x[index][1])

                self.data_sets_y[index][1] = self.load_binary(self.data_sets_y[index][1])

                self.data_sets_x[index][1], self.data_sets_y[index][1] = self.validate(self.data_sets_x[index][1],
                                                                                         self.data_sets_y[index][1])

            self.algorithm = translate_config_options[self.classifier](
                model_options=self.config.get("ALGORITHM_OPTIONS").get(self.classifier),
                label=label_)
            self.experiment(label=label_, search_param=False)

    def validate(self, x_data, y_data):
        """
        Валидация корпуса.
        :param x_data:
        :param y_data:
        :return:
        """

        # todo костыль для RNCgold
        print('\n', 'Possible non valid patterns.')
        non_valid_objects_index = list()
        for i in range(0, len(x_data)):
            if len(x_data[i]) != len(y_data[i]):
                print([el.get('word.lower()') for el in (x_data[i])], y_data[i])
                non_valid_objects_index.append(i)
        print('Count non valid patterns:', len(non_valid_objects_index))

        x_data = [el for index, el in enumerate(x_data) if index not in non_valid_objects_index]
        y_data = [el for index, el in enumerate(y_data) if index not in non_valid_objects_index]
        print('Count elements in train, test after filter:', len(x_data), len(y_data), '\n')
        return x_data, y_data

    def load_binary(self, file):
        """
        Загрузка бинарного файла.
        :param file:
        :return:
        """

        with open(file, 'rb') as f:
            print('Load:', file)
            return pickle.load(f)

    def load_binary_serializing_data(self, file_name):
        """
        Загрузка бинарного файла частями.
        :param file_name:
        :param bin_records:
        :return:
        """

        bin_records = self.load_binary(
            '/'.join(file_name.split('/')[:-1]) + '/bin_records_for_%s' % file_name.split('/')[-1:][0])
        file = open(file_name, 'rb')
        for binary_index in bin_records:
            yield loads(file.read(binary_index))

    def find_files(self, label, data_mark):
        """
        Каждому типу сета найти соответ. файл с данными.
        :param label:
        :return:
        """

        for folders in os.listdir(self.data_path):
            if folders == label:
                label_folder = self.data_path + '/' + folders

                for folders_ in os.listdir(label_folder):
                    if data_mark in folders_:
                        data_folder = label_folder + '/' + data_mark

                        for files in os.listdir(data_folder):
                            for index in range(0, len(self.data_sets_x)):

                                if self.data_sets_x[index][0] in files and 'bin_records_for' not in files:
                                    self.data_sets_x[index][1] = data_folder + '/' + files

                                if self.data_sets_y[index][0] in files and 'bin_records_for' not in files:
                                    self.data_sets_y[index][1] = data_folder + '/' + files

    def load_data(self, file_name):
        """
        Загрузка данных.
        :param file_name:
        :return:
        """

        data = list()
        binary_record = '/'.join(file_name.split('/')[:-1]) + '/bin_records_for_%s' % file_name.split('/')[-1:][0]
        bin_records = self.load_binary(binary_record)
        file = open(file_name, 'rb')
        for binary_index in bin_records:
            data.append(loads(file.read(binary_index)))
        print('Load: %s' % file_name)
        return data

    def load_list_of_sub_tags(self):
        """
        Загрузка списка субтегов.
        :return:
        """

        file = self.data_path + '/grammar_catigories.txt'
        with open(file, 'r', encoding='utf-8') as text:
            text = text.read()
            sub_tags = text.split('\n')
            return [el for el in sub_tags if el != '']

    def save_binary(self, label, data):
        """
        Сохранить данные в бинарном формате.
        :param file_name:
        :param data:
        :return:
        """

        directory = os.path.abspath(os.path.split(os.path.abspath(__file__))[0] + '/../../models/%s/%s' %
                                    (self.corpora_name, self.algorithm.algorithm_name))
        distutils.dir_util.mkpath(directory)
        file_path = directory + '/' + label + '.pkl'
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    def save_binary_theano(self, label):
        """
        Сохранить данные в бинарном формате.
        :param file_name:
        :param data:
        :return:
        """

        directory = os.path.abspath(os.path.split(os.path.abspath(__file__))[0] + '/../../models/%s/%s' %
                                    (self.corpora_name, self.algorithm.algorithm_name))
        distutils.dir_util.mkpath(directory)
        file_path = directory + '/' + label
        numpy.savez(file_path, *lasagne.layers.get_all_param_values(self.algorithm.network))

    def experiment(self, label="POS", search_param=None):
        """
        Обучения классификатора.
        Сохранение модели.
        Прогон на тестовом множестве.
        :return:
        """

        if search_param:
            self.algorithm.hyperparameter_optimization(self.data_sets_x[0][1],
                                                       self.data_sets_y[0][1])
            self.save_binary(label, self.algorithm.model)
        else:
            self.algorithm.train(self.data_sets_x[0][1],
                                 self.data_sets_y[0][1],
                                 self.data_sets_x[2][1],
                                 self.data_sets_y[2][1])

            if self.classifier != 'MLPT':
                self.save_binary(label, self.algorithm.model)
                y_predicted, labels_without_o = self.algorithm.prediction(self.data_sets_x[1][1], model=self.algorithm.model)
                self.algorithm.evaluation(self.data_sets_y[1][1], y_predicted, labels_without_o, model=self.algorithm.model)
            else:
                self.algorithm.evaluation(self.data_sets_x[1][1], self.data_sets_y[1][1])
                self.save_binary_theano(label)

if __name__ == '__main__':
    ModelWorker()
