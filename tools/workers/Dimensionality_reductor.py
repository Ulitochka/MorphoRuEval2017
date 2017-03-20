# -*- coding: utf-8 -*-
#
# Скрипт ориентирован на использование в Python 3.*
#
# Скрипт содержит в себе класс SVDReductor.

import pickle
import os
import csv
from collections import OrderedDict

from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
import numpy


class SVDReductor:
    def __init__(self, svd_options=None, mode='svds'):
        self.n_components = svd_options.get('N_COMPONENTS')
        self.mode = mode
        algorithm = svd_options.get('ALGORITHM')
        tol = svd_options.get('TOL')
        n_iter = svd_options.get('N_ITER')

        self.svd = TruncatedSVD(n_components=self.n_components,
                                algorithm=algorithm,
                                tol=tol,
                                n_iter=n_iter)

    def dimension_reduction(self, train, test, dev):
        """
        Сокращение размерности признаков.
        :param data:
        :param library:
        :return:
        """

        print('\nDim reduction starts ...')
        data = [train, test, dev]
        data_red = list()
        for index, sets in enumerate(data):
            if self.mode == 'TruncatedSVD':
                if index == 0:
                    self.svd.fit(sets)
                    sets = self.svd.transform(sets)
                    print('reduced shape:', sets.shape)
                    data_red.append(sets)
                else:
                    sets = self.svd.transform(sets)
                    print('reduced shape:', sets.shape)
                    data_red.append(sets)
            else:
                u, s, vt = svds(sets, k=self.n_components)
                print('svds done.')
                s_diagonal_matrix = numpy.zeros((s.shape[0], s.shape[0]))
                for i in range(s.shape[0]):
                    s_diagonal_matrix[i, i] = s[i]
                print('s_diagonal_matrix done.')
                sets = numpy.dot(numpy.dot(u, s_diagonal_matrix), vt)
                print('lower rank shape:', sets.shape)
        print('Dim reduction done.')
        return data_red[0], data_red[1], data_red[2]
