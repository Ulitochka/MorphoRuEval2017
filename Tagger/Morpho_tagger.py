# -*- coding: utf-8 -*-
#
# Скрипт ориентирован на использование в Python 3.*
#
# Скрипт содержит в себе класс CRFMorphoTagger для морфологической разметки токенов.

import os
import glob
import distutils.dir_util
import pickle
from pickle import loads

from tools.workers.Config_worker import Config
from tools.algorithms.CRF import CRF
from tools.algorithms.Logistic_regression import LogRegression
from tools.workers.Feature_extractor import FeatureExtractor

import numpy


class MorphoTagger:
    """
    Класс для морфологического теггера.
    """
    def __init__(self, config=None, model_name=None, text=None, eval_type=None):

        self.decoder = {
                "True": True,
                "False": False,
                'CRF': CRF,
                'LOGREG': LogRegression
        }

        self.algorithms = {
            'crf_': CRF,
            'logreg_': LogRegression
        }

        self.file_path = os.path.split(os.path.abspath(__file__))[0]
        if config:
            self.config = config
        else:
            file_name = os.path.abspath(self.file_path + '/config.json')
            self.config = Config(file_name=file_name)

        self.corpora_name = self.config.get("CORPORA")
        self.data_path = None
        self.data_mark = None

        self.eval_type = eval_type
        self.text = text

        self.model_name = model_name
        self.model = None
        self.load_models(models_type=model_name)

        self.data_sets_x = None
        self.data_sets_y = None
        self.spec_words = self.load_spec_tokens()
        self.spec_words_labels = {'ADP', 'CONJ', 'PART', 'H', 'INTJ'}

    def load_data(self):
        """
        Загрузка данных.
        :return:
        """

        if self.text is True:

            feature_extractor = FeatureExtractor(features=self.config.get('FEATURES_OPTIONS'),
                                                 data_type='dict',
                                                 spec_labels=[])
            self.data_path = os.path.abspath(self.file_path + '/../data/text/' + self.config.get("TEXT_NAME") + '.txt')
            print('Text:', self.config.get("TEXT_NAME"))
            text = self.load_text_to_tagger()
            print('Count sent:', len(text))
            text_with_features = feature_extractor.sentence2features(text)
            print('Features done.')
            return text_with_features, None

        else:

            self.data_path = os.path.abspath(self.file_path + '/../data/' + self.corpora_name + '/')

            data_type = self.config.get("FORMAT")
            binarization = self.decoder.get(self.config.get("BINARY"))
            dim_reduction = self.decoder.get(self.config.get("DIM_RED"))
            self.data_mark = 'type_%s_bin_%s_dimred_%s' % (data_type, binarization, dim_reduction)

            data_sets_x = [['x_test', None]]
            data_sets_y = [['y_test', None]]

            self.find_files('POS&GRAMMAR', self.data_mark)
            for index in range(0, len(data_sets_x)):
                if self.model_name == 'crf_':
                    data_sets_x[index][1] = self.load_evaluate_data(data_sets_x[index][1])
                else:
                    data_sets_x[index][1] = self.load_binary(data_sets_x[index][1])

                data_sets_y[index][1] = self.load_binary(data_sets_y[index][1])

            return data_sets_x, data_sets_y

    def load_text_to_tagger(self):
        """
        Загрузка текста для разметки.
        :return:
        """

        data = list()
        with open(self.data_path, 'r', encoding='utf-8') as train_text:
            train_text = train_text.read()
            sentences = train_text.split('\n\n')
            for sentence in sentences:
                if not sentence.strip():
                    continue
                all_words_in_sentence = list()
                for tokens in sentence.rstrip().split('\n'):
                    tokens = tokens.split('\t')
                    all_words_in_sentence.append([tokens[-1], None])
                data.append(all_words_in_sentence)
            return data

    def load_spec_tokens(self):
        """
        Загрузка списков служебных слов.
        :return:
        """

        spec_words = list()
        spec_tokens_data_path = os.path.abspath(self.file_path + '/../data/' + self.corpora_name + '/spec_words/')

        for files in os.listdir(spec_tokens_data_path):
            with open(spec_tokens_data_path + '/' + files, 'r', encoding='utf-8') as text:
                spec_words.extend([tokens.lower() for tokens in text.read().split('\n')])
        return set(spec_words)

    def load_binary(self, file):
        """
        Загрузка бинарного файла.
        :param file:
        :return:
        """

        with open(file, 'rb') as f:
            return pickle.load(f)

    def find_files(self, label, data_mark):
        """
        Каждому типу сета найти соответ. файл с данными.
        :param label:
        :return:
        """

        for folders in os.listdir(self.data_path):
            if label in folders:
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

    def load_evaluate_data(self, file_name):
        """
        Загрузка данных.
        :param file_name:
        :return:
        """

        data = []
        binary_record = '/'.join(file_name.split('/')[:-1]) + '/bin_records_for_%s' % file_name.split('/')[-1:][0]
        bin_records = self.load_binary(binary_record)
        file = open(file_name, 'rb')
        for binary_index in bin_records:
            data.append(loads(file.read(binary_index)))
        print('Load: %s' % file_name)
        file.close()
        return data

    def load_models(self, models_type=None):
        """
        Загрузка файлов моделей.
        :return:
        """

        models_files = dict()
        directory = os.path.abspath(os.path.split(os.path.abspath(__file__))[0] + '/../models/%s/%s' % (self.corpora_name , self.model_name,))
        os.chdir(directory)
        list_files = glob.glob('*.*pkl')

        pattern = directory
        for files_ in list_files:
            string = pattern + "/" + files_
            print('Load model:', string)
            model_from_file = self.load_binary(string)
            model = self.algorithms.get(self.model_name)(model_options=None, model=model_from_file)
            models_files[files_.replace('.pkl', '')] = model
        self.model = models_files

    def merge_predictions(self, data):
        """
        Соединить результаты различных классификаторов.
        :param data:
        :return:
        """

        grammar_classes_positions = self.config.get('GRAMMAR_CLASSES_POSITIONS')
        # генерируем пустой массив предложений; каждому токену соотв. массив grammar_classes_positions;
        empty_data = [[[[] for i in range(len(grammar_classes_positions))] for i in range(len(el))]
                      for el in data["POS"]]
        # заполняем пустой массив;
        for el in data:
            for index_s, sent in enumerate(data[el]):
                for index_t, tokens in enumerate(sent):
                    if tokens != 'O':
                        if self.eval_type:
                            if el != 'POS':
                                empty_data[index_s][index_t][grammar_classes_positions[el]].append(el + '=' + tokens)
                            else:
                                empty_data[index_s][index_t][grammar_classes_positions[el]].append(tokens)
                        else:
                            empty_data[index_s][index_t][grammar_classes_positions[el]].append(tokens)

        # перевод массива grammar_classes_positions в строки;
        string_format_labels = [['_'.join([labels[0] for labels in tokens if labels])
                                 for tokens in sent] for sent in empty_data]
        return string_format_labels

    def marker(self, x_data):
        """
        Морфологическая размека токенов.
        :param data:
        :param model:
        :return:
        """

        return {el: self.model[el].prediction(x_data, model=self.model[el].model)[0] for el in self.model}

    def evaluate_tagger(self, predictions):
        """
        Оценка таггера.
        :param y_test:
        :param predictions:
        :param labels:
        :return:
        """

        labels = sorted(set([labels for sent in self.data_sets_y[0][1] for labels in sent]))
        if self.model_name == 'crf_':
            # у любого ключа / экземпляра класса CRF_model есть такой метод;
            self.model['POS'].evaluation(self.data_sets_y [0][1], predictions, labels)
        else:
            predictions = numpy.array(predictions)
            y_test = numpy.array(self.data_sets_y [0][1])
            self.model['POS'].evaluation(y_test.reshape(y_test.shape[0], 1),
                                         predictions.reshape(predictions.shape[0], 1),
                                         labels)

    def revert_token_reg(self, word_lower, word_upper, word_title):
        """
        Восстанавливается исходный регистр токена.
        :param word_lower:
        :param word_upper:
        :param word_title:
        :return:
        """

        if word_upper == 'True':
            return word_lower.upper()
        elif word_title == 'True':
            return word_lower[0].upper() + ''.join(word_lower[1:])
        else:
            return word_lower

    def out_in_ud_format(self, predicted_data, x_data):
        """
        Выгрузка в специальном формате.
        "Индекс<TAB>слово[<TAB>лемма]<TAB>часть_речи<TAB>полная_метка"
        is_upper(token) Токен в верхнем регистре?
        is_title(token) Начинается ли строка с заглавной буквы

        :param predited_data:
        :return:
        """

        tokens_with_labels = list()
        for sent_index, sent_features in enumerate(x_data):
            for token_index, token_features in enumerate(sent_features):

                id = str(token_index+1)
                token = self.revert_token_reg(token_features.get('word.lower()'),
                                              token_features.get('word.is_upper()'),
                                              token_features.get('word.is_title()'))
                pos = predicted_data[sent_index][token_index].split('_')[0]
                morpho_tags = '|'.join(predicted_data[sent_index][token_index].split('_')[1:])
                if morpho_tags == '':
                    morpho_tags = '_'

                tokens_with_labels.append('\t'.join([id, token, pos, morpho_tags]))
            tokens_with_labels.append('')

        data_path_for_result = os.path.abspath(self.data_path + '/../../results/' + self.corpora_name + '/')
        distutils.dir_util.mkpath(data_path_for_result)

        if self.text is True:
            file_name = data_path_for_result + '/' + self.model_name + self.config.get("TEXT_NAME") + '_' + self.corpora_name + '.txt'
        else:
            file_name = data_path_for_result + '/' + self.model_name + '_' + self.corpora_name + '.txt'

        f = open(file_name, "w", encoding='utf-8')
        for el in tokens_with_labels:
            f.write(el + '\n')
        f.close()

if __name__ == '__main__':
    tagger = MorphoTagger(model_name='crf_', text=True, eval_type=True)
    x_data, y_data = tagger.load_data()
    predictions = tagger.marker(x_data)
    predictions = tagger.merge_predictions(predictions)
    # tagger.evaluate_tagger(predictions)
    tagger.out_in_ud_format(predictions, x_data)
