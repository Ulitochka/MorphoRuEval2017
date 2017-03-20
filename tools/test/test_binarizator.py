import unittest
import os

import pandas

from tools.workers.Binarizator import Binarizator


class TestBinarizator(unittest.TestCase):
    def setUp(self):
        self.path_to_test_file = os.path.split(os.path.abspath(__file__))[0] + '/testing_binarization.csv'
        self.test_binarizator = Binarizator()

    @staticmethod
    def load_csv(file_name=None, drop_nan=None):
        """
        Загрузка данных в стандартном формате:
                имена признаков - колонки;
                строки - значения признаков у объектов;
        :return:
        """

        dataset = pandas.read_csv(file_name, delimiter=',', quotechar='|')
        print('%s shape:' % file_name, dataset.shape)
        if drop_nan:
            dataset = dataset.dropna()
            print('%s shape after drop nan:' % file_name, dataset.shape)
        return dataset

    def test_split_categorical_numeral(self):
        true_result = [(6, 8), (6, 1)]
        non_numeral_features, numeral_features = self.test_binarizator.split_categorical_numeral_per_column_name(
            self.load_csv(file_name=self.path_to_test_file), 'test_data')
        self.assertEqual([non_numeral_features.shape, numeral_features.shape], true_result)

    def test_extract_cat_features_means(self):
        true_features_means = [[False, True], [False, True], [False, True], [False, True], [False, True],
                               ['False', 'А', 'Б', 'В', 'Г', 'Д'], ['False', 'АА', 'ВВ', 'ГГ', 'ДД'],
                               ['False', 'ААА', 'БББ', 'ВВВ', 'ГГГ', 'ДДД']]
        non_numeral_features, numeral_features = self.test_binarizator.split_categorical_numeral_per_column_name(
            self.load_csv(file_name=self.path_to_test_file), 'test_data')
        fact_feature_means = self.test_binarizator.extract_cat_features_means(
            data_train_cat=non_numeral_features, data_dev_cat=non_numeral_features, data_test_cat=non_numeral_features)
        self.assertEqual(fact_feature_means, true_features_means)

    def test_binarized(self):

        true_result = [
            [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
             0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
             1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]

        non_numeral_features, numeral_features = self.test_binarizator.split_categorical_numeral_per_column_name(
            self.load_csv(file_name=self.path_to_test_file), 'test_data')
        fact_feature_means = self.test_binarizator.extract_cat_features_means(
            data_train_cat=non_numeral_features, data_dev_cat=non_numeral_features, data_test_cat=non_numeral_features)
        fact_result = self.test_binarizator.binarized(non_numeral_features, fact_feature_means, sparse_matrix=False)
        # print(non_numeral_features) # for compare
        self.assertEqual([list(el) for el in list(fact_result)], true_result)

    def test_concatenate_num_cat_sparse_false(self):
        non_numeral_features, numeral_features = self.test_binarizator.split_categorical_numeral_per_column_name(
            self.load_csv(file_name=self.path_to_test_file), 'test_data')
        fact_feature_means = self.test_binarizator.extract_cat_features_means(
            data_train_cat=non_numeral_features, data_dev_cat=non_numeral_features, data_test_cat=non_numeral_features)
        bin = self.test_binarizator.binarized(non_numeral_features, fact_feature_means, sparse_matrix=False)
        merged = self.test_binarizator.concatenate_num_cat(bin, numeral_features)
        self.assertEqual(numeral_features.shape[1] + bin.shape[1], merged.shape[1])

    def test_concatenate_num_cat_sparse_true(self):
        non_numeral_features, numeral_features = self.test_binarizator.split_categorical_numeral_per_column_name(
            self.load_csv(file_name=self.path_to_test_file), 'test_data')
        fact_feature_means = self.test_binarizator.extract_cat_features_means(
            data_train_cat=non_numeral_features, data_dev_cat=non_numeral_features, data_test_cat=non_numeral_features)
        bin = self.test_binarizator.binarized(non_numeral_features, fact_feature_means, sparse_matrix=True)
        merged = self.test_binarizator.concatenate_num_cat(bin, numeral_features, sparse_matrix=True)
        self.assertEqual(numeral_features.shape[1] + bin.shape[1], merged.shape[1])

if __name__ == '__main__':
    unittest.main(verbosity=2)