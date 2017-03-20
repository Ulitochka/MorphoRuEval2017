# -*- coding: utf-8 -*-
#
# Скрипт ориентирован на использование в Python 3.*
#
# Скрипт содержит в себе класс для работы с CRF.

from collections import Counter

import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.grid_search import RandomizedSearchCV
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import cross_val_score
from tabulate import tabulate


class CRF:
    """
    Класс для работы с алгоритмом CRF.

    """
    def __init__(self, model_options=None, model=None, label=None):
        self.algorithm_name = 'crf_'
        translate_config_options = {
            "True": True,
            "False": False
        }

        if model:
            self.model = model
        else:
            self.model = sklearn_crfsuite.CRF(
                algorithm=model_options.get('OPTIMIZATION_ALGORITHM'),
                c1=model_options.get('C1'),
                c2=model_options.get('C2'),
                max_iterations=model_options.get('MAX_ITERATIONS'),
                all_possible_transitions=translate_config_options.get(model_options.get('ALL_POSSIBLE_TRANSITIONS')),
                verbose=True
            )

    def evaluation(self, y_test, y_predicted, labels, model=None):
        """
        Оценка модели.
        :param y_test:
        :param y_predicted:
        :param labels:
        :return:
        """

        result = {
            'f1_score': metrics.flat_f1_score(y_test,
                                              y_predicted,
                                              average='weighted',
                                              labels=labels),
            'precision': metrics.flat_precision_score(y_test,
                                                      y_predicted,
                                                      average='weighted',
                                                      labels=labels),
            'accuracy': metrics.flat_accuracy_score(y_test,
                                                    y_predicted)
        }

        # базовые метрики;
        print('Metrics briefly.')
        headers_means = [(el, result[el]) for el in result]
        print(tabulate([[el[1] for el in headers_means]], headers=[el[0] for el in headers_means]))
        print('\n')

        # отчет по каждому классу: precision, recall, f1-score, support;
        sorted_labels = sorted(labels)

        print('Count true labels:', len(set([sub_el for el in y_test for sub_el in el])))
        print('Count fact labels:', len(set([sub_el for el in y_predicted for sub_el in el])))

        print(metrics.flat_classification_report(y_test,
                                                 y_predicted,
                                                 labels=sorted_labels,
                                                 digits=3))

    def prediction(self, x_test, delete_labels=('O'), model=None):
        """
        Предсказание.
        :param x_test:
        :param delete_labels:
        :param model:
        :return:
        """

        assert (model is not None), 'No model!'
        labels = [el for el in list(model.classes_) if el not in delete_labels]
        y_predicted = model.predict(x_test)
        return y_predicted, labels

    def train(self, x_train, y_train, x_dev=None, y_dev=None, cross_valid=None):
        """
        Обучение.
        :param x_train:
        :param y_train:
        :param x_dev:
        :param y_dev:
        :param cross_valid:
        :param model:
        :return:
        """

        if cross_valid:
            self.model.fit(x_train, y_train)
            labels = sorted(set([sub_el for el in y_train for sub_el in el]))
            f1_scorer = make_scorer(sklearn_crfsuite.metrics.flat_f1_score,
                                    average='weighted',
                                    labels=labels)
            scores = cross_val_score(self.model, x_train, y_train, cv=3, scoring=f1_scorer)
            print('Scores (f1):', scores)
        else:
            self.model.fit(x_train, y_train)

    def state_features(self, model):
        """
        Анализ признаков.
        :param model:
        :return:
        """

        print("Top positive:")
        for (attr, label), weight in Counter(model.state_features_).most_common(100):
            print("%0.6f %-8s %s" % (weight, label, attr))

        print("Top negative:")
        for (attr, label), weight in Counter(model.state_features_).most_common()[-100:]:
            print("%0.6f %-8s %s" % (weight, label, attr))

    def hyperparameter_optimization(self, x_train, y_train, x_test=None, y_test=None, labels=None):
        """
        Подбор оптимальных параметров.
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :param labels:
        :return:
        """

        model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=300,
            all_possible_transitions=True
        )

        params_space = {
            'c1': scipy.stats.expon(scale=0.01),
            'c2': scipy.stats.expon(scale=0.01),
        }

        f1_scorer = make_scorer(sklearn_crfsuite.metrics.flat_f1_score,
                                average='weighted',
                                labels=labels)

        rs = RandomizedSearchCV(model, params_space,
                                cv=3,
                                verbose=1,
                                n_jobs=1,
                                n_iter=10,
                                scoring=f1_scorer)

        rs.fit(x_train, y_train)
        print('\n')
        print('best params:', rs.best_params_)
        print('best CV score:', rs.best_score_)

        best_model = rs.best_estimator_

        if y_test:
            y_pred = best_model.predict(x_test)
            print(metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=3))
        else:
            self.model = best_model
