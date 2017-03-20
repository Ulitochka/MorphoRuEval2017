# -*- coding: utf-8 -*-
#
# Скрипт ориентирован на использование в Python 3.*
#
# Скрипт содержит в себе класс Binarizator, содержащий методы для бинаризации признаков.

import os

from tabulate import tabulate
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy


class Binarizator:

    def split_categorical_numeral_per_column_name(self, data, data_name):
        """
        Разделение признаков на категориальные и остальные.
        :param data:
        :return:
        """

        print('\nSplitting categorical features starts ...')
        numeral_features_column_names = list(data.select_dtypes(include=['number']).columns)
        numeral_features = data.loc[:, data.columns.isin(numeral_features_column_names)]
        print('%s with only numeral features shape:' % data_name, numeral_features.shape)
        # print(numeral_features)

        non_numeral_features_column_names = list(data.select_dtypes(include=['object', 'bool']).columns)
        non_numeral_features = data.loc[:, data.columns.isin(non_numeral_features_column_names)]
        print('%s with only non_numeral features shape:' % data_name, non_numeral_features.shape)
        # print(non_numeral_features)

        print('Splitting categorical features done.')
        return non_numeral_features, numeral_features

    def transform(self, train_x, test_x, dev_x):
        """
        Бинаризация данных.
        :return:
        """

        print('\nBinarization starts...')

        data_cat_train, data_num_train = self.split_categorical_numeral_per_column_name(train_x, 'train_x')
        data_cat_test, data_num_test = self.split_categorical_numeral_per_column_name(test_x, 'test_x')
        data_cat_dev, data_num_dev = self.split_categorical_numeral_per_column_name(dev_x, 'dev_x')

        # Feature_means
        feature_means = self.extract_cat_features_means(data_cat_train, data_cat_test, data_cat_dev)

        # One hor encoding
        binary_train_data = self.binarized(data_cat_train, feature_means, sparse_matrix=True)
        binary_test_data = self.binarized(data_cat_test, feature_means, sparse_matrix=True)
        binary_dev_data = self.binarized(data_cat_dev, feature_means, sparse_matrix=True)

        # Merge
        merged_x_test = self.concatenate_num_cat(data_num_test, binary_test_data, sparse_matrix=True)
        merged_x_train = self.concatenate_num_cat(data_num_train, binary_train_data, sparse_matrix=True)
        merged_x_dev = self.concatenate_num_cat(data_num_dev, binary_dev_data, sparse_matrix=True)

        print(tabulate([
            [
                merged_x_test.shape,
                merged_x_train.shape,
                merged_x_dev.shape
            ]],
            headers=[
                'merged_test',
                'merged_train',
                'merged_dev']))

        return merged_x_train, merged_x_test, merged_x_dev

    @staticmethod
    def split_categorical_numeral_per_index(data, index_data_cat, index_data_num):
        """
        Разделение числовых и категориальных признаков.
        :param data:
        :param index_data:
        :return:
        """

        return data.iloc[:, index_data_cat], data.iloc[:, index_data_num]

    def extract_cat_features_means(self, data_train_cat=None, data_test_cat=None, data_dev_cat=None):
        """
        Извлеченеи категориальных признаков.
        :param data_train_cat:
        :param data_test_cat:
        :param data_dev_cat:
        :return:
        """

        # Список переменных;
        features_means = list()
        # делаем выборку только категориальных признаков;
        for i in range(0, data_train_cat.shape[1]):
            # в каждом столбце выбираем уникальные значения;
            train = data_train_cat.iloc[:, i].unique()
            test = data_test_cat.iloc[:, i].unique()
            dev = data_dev_cat.iloc[:, i].unique()
            # объединяем значения из test, train, dev;
            features_means.append(sorted(list(set(train) | set(test) | set(dev))))
        print('\nCount all features:', len([sub_el for el in features_means for sub_el in el]), '\n')
        return features_means

    @staticmethod
    def binarized(data_cat, features_means, sparse_matrix=True):
        """
        Бинаризация.
        :param data_cat:
        :param features_means:
        :return:
        """

        # One hot encode all categorical attributes
        cats = list()
        conc_crs_matrix = None

        for i in range(0, data_cat.shape[1]):
            # Для кажого признака (колонки) уникальных значений из test и train проводим следующие операции:
            label_encoder = LabelEncoder()
            # ~ инициализация label_энкодера массивами с уникальными значениями каждого признака;
            label_encoder.fit(features_means[i])
            # transform from categorical to numeral;
            feature = label_encoder.transform(data_cat.iloc[:, i])  # передаем соответ. колонку из данных;
            feature = feature.reshape(data_cat.shape[0], 1)  # размер: количество объектов и 1 столбец;
            one_hot_encoder = OneHotEncoder(sparse=sparse_matrix,
                                            dtype="float32",
                                            n_values=len(features_means[i]),
                                            handle_unknown='error')
            # мы объединили значения из test, train. Поэтому one hot vector будет больше;
            feature = one_hot_encoder.fit_transform(feature)
            cats.append(feature)

            if sparse_matrix:
                if conc_crs_matrix is None:
                    conc_crs_matrix = sparse.csr_matrix.copy(feature)
                else:
                    conc_crs_matrix = sparse.hstack((conc_crs_matrix, feature), format='csr')

        if conc_crs_matrix is not None:
            return conc_crs_matrix
        else:
            # Make a 2D array from a list of 1D arrays
            return numpy.column_stack(cats)

    @staticmethod
    def concatenate_num_cat(data_num, data_cat_enc, sparse_matrix=False):
        """
        Concatenate encoded attributes with continuous attributes.
        :param data_num:
        :param data_cat_enc:
        :return:
        """

        if sparse_matrix:
            data_num = sparse.csr_matrix(data_num, dtype="float32")
            return sparse.hstack((data_cat_enc, data_num), format='csr')
        else:
            return numpy.concatenate((data_cat_enc, data_num), axis=1)
