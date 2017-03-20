# -*- coding: utf-8 -*-
#
# Скрипт ориентирован на использование в Python 3.*
#
# Скрипт содержит в себе класс FeatureExtractor, содержащий методы для извлечения различных признаков и формирования
# вектора признаков.

import os
import copy
import pickle
import distutils.dir_util
from inspect import isfunction
from collections import OrderedDict

import numpy
import pandas

from tools.algorithms.CRF import CRF


class FeatureExtractor:
    def __init__(self, config=None, data_type=None, spec_labels=None):
        """
        data_type: list - pandas data frame для бинаризации и сокращения размерности;
                   dict - для crf-sklearn;
        :param features:
        :param data_type:
        """

        self.config = config
        self.features = self.config.get('FEATURES_OPTIONS')

        self.data_type = data_type
        self.spec_labels = spec_labels

        self.context_right = self.features.get("CONTEXT_RIGHT")
        self.context_left = self.features.get("CONTEXT_LEFT")

        self.standard_features = [['bias', 1.0]]
        self.features_for_context_tokens = list()
        self.features_name_init()

        if self.features.get("POS_TAGGER"):
            self.pos_tagger = CRF(model_options=None, model=self.load_pos_tagger())

    def features_name_init(self):
        """
        Инициализация имен признаков.
        :return:
        """

        if self.features.get("SUFFIX_OPTIONS"):
            self.standard_features.extend([['word[%s]' % el, el] for el in self.features.get("SUFFIX_OPTIONS") if el != 0])
        if self.features.get("PREFIX_OPTIONS"):
            self.standard_features.extend([['word[%s]' % el, el] for el in self.features.get("PREFIX_OPTIONS") if el != 0])
        if self.features.get("ISUPPER"):
            self.standard_features.extend([['word.is_upper()', self.is_upper]])
        if self.features.get("ISTITLE"):
            self.standard_features.extend([['word.is_title()', self.is_title]])
        if self.features.get("ISDIGIT"):
            self.standard_features.extend([['word.is_digit()', self.is_digit]])
        if self.features.get("LOWER_TOKEN"):
            self.standard_features.extend([['word.lower()', self.transform_to_lower_format]])

        if self.features.get("SUFFIX_OPTIONS_CONTEXT"):
            self.features_for_context_tokens.extend([['word[%s]' % el, el] for el in self.features.get("SUFFIX_OPTIONS_CONTEXT") if el != 0])
        if self.features.get("PREFIX_OPTIONS_CONTEXT"):
            self.features_for_context_tokens.extend([['word[%s]' % el, el] for el in self.features.get("PREFIX_OPTIONS_CONTEXT") if el != 0])
        if self.features.get("ISUPPER_CONTEXT"):
            self.features_for_context_tokens.extend([['word.is_upper()', self.is_upper]])
        if self.features.get("ISTITLE_CONTEXT"):
            self.features_for_context_tokens.extend([['word.is_title()', self.is_title]])
        if self.features.get("ISDIGIT_CONTEXT"):
            self.features_for_context_tokens.extend([['word.is_digit()', self.is_digit]])
        if self.features.get("LOWER_TOKEN_CONTEXT"):
            self.features_for_context_tokens.extend([['word.lower()', self.transform_to_lower_format]])

    def load_pos_tagger(self):
        """
        Загрузка бинарного файла POS-теггера.
        :return:
        """

        directory = os.path.abspath(os.path.split(os.path.abspath(__file__))[0] + '/../../models/%s/%s' %
                                    (self.config.get("CORPORA"), 'crf_'))
        distutils.dir_util.mkpath(directory)
        file_path = directory + '/' + 'POS' + '.pkl'
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def is_digit(token):
        """
        Токен - число?
        :param token:
        :return:
        """

        return token.isdigit()

    @staticmethod
    def is_title(token):
        """
        Начинается ли токен с заглавной буквы.
        :param token:
        :return:
        """

        return token.istitle()

    @staticmethod
    def transform_to_lower_format(token):
        """
        Перевести в нижний регистр.
        :param token:
        :return:
        """

        return token.lower()

    @staticmethod
    def is_upper(token):
        """
        Токен в верхнем регистре?
        :param token:
        :return:
        """

        return token.isupper()

    def pos_label(self, token):
        """
        Добавление частиречной разметки.
        'word.pos_tag()'
        :param token:
        :return:
        """

        return self.pos_tagger.model.predict([[token]])[0][0]

    @staticmethod
    def is_bound(m):
        return hasattr(m, '__self__')

    def generate_feature_dict_per_tokens(self, word, grammem, context_index=None):
        """
        Генерация словаря с признаками для токена
        :param word:
        :param grammem:
        :param context_index:
        :return:
        """

        features_per_word = None
        if context_index:
            features_per_word = copy.deepcopy(self.features_for_context_tokens)
        else:
            features_per_word = copy.deepcopy(self.standard_features)

        for elements in features_per_word:
            if context_index:
                elements[0] = elements[0].replace('word', '%s:word' % context_index)
            if word:
                if elements[0] != 'bias':
                    if grammem != 'PUNCT':

                        # @staticmethod
                        if isfunction(elements[1]):
                            elements[1] = str(elements[1](word))

                        else:
                            # prefix, suffix;
                            if len(word) < abs(elements[1]):
                                elements[1] = 'False'
                            else:
                                if elements[1] < 0:
                                    elements[1] = word.lower()[elements[1]:]
                                else:
                                    elements[1] = word.lower()[:elements[1]]

                    # для пунктуации все 0;
                    else:
                        elements[1] = 'False'
            else:
                if elements[0] != 'bias':
                    elements[1] = 'False'

        return features_per_word

    def word2features(self, sentence, index):
        """
        Создание вектора признаков для токена.
        :param sentence:
        :param index:
        :return:
        """

        features_per_word = None
        word = sentence[index][0]

        features_per_word_x = self.generate_feature_dict_per_tokens(word, sentence[index][0])
        features_per_word = features_per_word_x

        # признаки для контекста;
        maximum_context = {i for i in range(self.context_left * -1, self.context_right + 1) if i}
        real_context = [i for i in range(max(0, index - self.context_left),
                                         min(index + self.context_right + 1, len(sentence)))
                        if i != index]
        fake_context = (maximum_context.difference({i-index for i in real_context}))

        for i in real_context:
            position_in_context_window = i - index
            if position_in_context_window > 0:
                position_in_context_window = '+' + str(position_in_context_window)
                features_per_word.extend(self.generate_feature_dict_per_tokens(
                    sentence[i][0], sentence, position_in_context_window))
            else:
                features_per_word.extend(self.generate_feature_dict_per_tokens(
                    sentence[i][0], sentence, position_in_context_window))

        for i_ in fake_context:
            if i_ > 0:
                features_per_word.extend(self.generate_feature_dict_per_tokens(None, None, '+' + str(i_)))
            else:
                features_per_word.extend(self.generate_feature_dict_per_tokens(None, None, str(i_)))

        # символы начала и конца предложения;
        if index == 0:
            features_per_word.append(['BOS', 'True'])
        else:
            features_per_word.append(['BOS', 'False'])

        if index == len(sentence)-1:
            features_per_word.append(['EOS', 'True'])
        else:
            features_per_word.append(['EOS', 'False'])

        # добавление POS;
        if self.features.get("POS_TAGGER"):
            features_per_word.append(['word.pos_tag()', self.pos_label(OrderedDict(features_per_word))])

        return OrderedDict(features_per_word)

    def sentence2features(self, data):
        """
        Преобразование каждого токена предложения в вектор признаков.
        :param data:
        :return:
        """

        if self.data_type == 'dict':
            return [[self.word2features(sentence, i) for i, el in enumerate(sentence)] for sentence in data]
        else:
            return pandas.DataFrame([self.word2features(sentence, i) for sentence in data for i, el in enumerate(sentence)])

    def sentence2labels(self, data):
        """
        Замещение каждого токена в предложении его классом.
        :param data:
        :return:
        """

        if self.data_type == 'dict':
            return [[el[-1] if el[-1] not in self.spec_labels else 'X' for el in sentence ] for sentence in data]
        else:
            return pandas.DataFrame([(el[-1]) for sentence in data for i, el in enumerate(sentence)], columns=('label',))
