import pandas as pd
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt


class MulticriteriaDecision:
    """

    """
    def __init__(self, excel_table):
        """

        :param excel_table: шлях до Excel таблиці з даними
        """
        self._data = pd.read_excel(excel_table)
        self._feature_names = self._data['Features'].to_list()
        self._condition_map = self._data['Condition'].to_list()
        self._priority_map = self._data['Priority'].to_list()
        self._option_names = []
        self._normalized_features = defaultdict()

        self._data_matrix = self._get_matrix_view()
        self._get_normalized_features()

    def make_decision(self):
        """

        :return:
        """
        integral = np.zeros(self._data_matrix.shape[1])
        weights = defaultdict()

        # Вираховуємо вагові коефіцієнти
        for i in range(len(self._priority_map)):
            weights[f'feature_{i}_weight'] = self._priority_map[i] / sum(self._priority_map)

        # Вираховуємо інтегральні оцінки
        for i in range(self._data_matrix.shape[1]):
            integral[i] = 0
            for j in range(self._data_matrix.shape[0]):
                weight_coefficient = weights[f'feature_{j}_weight']
                normalized_feature = self._normalized_features[f'norm_feature_{j}'][i]

                integral[i] += weight_coefficient * (1 - normalized_feature) ** (-1)

        # Приймаємо рішення по найнижчому інтегралу
        opt = 0
        min_value = 1000
        for i in range(self._data_matrix.shape[1]):
            if min_value > integral[i]:
                min_value = integral[i]
                opt = i
        print('| Назва опції | Інтегрована оцінка |')
        for option, score in zip(self._option_names, integral):
            print(f'| {option} | {score} |')
        print('Оптимальний варіант:', self._option_names[opt])

    def show_plot(self):
        """

        :return:
        """
        for i, (_, values) in enumerate(self._normalized_features.items()):
            plt.plot(values, label=self._feature_names[i])

        plt.title('Normalized values of all features')
        plt.xlabel('Options')
        plt.ylabel('Normalized Values')
        plt.legend()
        plt.grid(True)
        plt.show()

    def _get_normalized_features(self):
        """

        :return:
        """
        features = defaultdict()
        sum_features = defaultdict()
        norm_features = defaultdict()

        # Отримуємо масиви рядків та їх суми
        for i in range(self._data_matrix.shape[0]):
            # отримуємо рядки
            feature_array = self._get_row_array(i)
            features[f'feature_{i}'] = feature_array

            # отримуємо суми рядків
            if self._condition_map[i] == 'max':
                feature_array = 1 / feature_array

            sum_features[f'sum_feature_{i}'] = sum(feature_array)

        # Отримуємо нормалізовані масиви
        for i in range(self._data_matrix.shape[0]):
            norm_features[f'norm_feature_{i}'] = np.zeros(features[f'feature_{i}'].shape)
            for j in range(self._data_matrix.shape[1]):
                feature_cell = features[f'feature_{i}'][j]

                if self._condition_map[i] == 'max':
                    feature_cell = 1 / feature_cell

                norm_features[f'norm_feature_{i}'][j] = feature_cell / sum_features[f'sum_feature_{i}']

        self._normalized_features = norm_features

    def _get_matrix_view(self):
        """

        :return:
        """
        sample_data = self._data.drop(columns=['Condition', 'Features', 'Priority'])
        data_matrix = np.zeros(sample_data.shape)
        self._option_names = sample_data.columns.tolist()

        for i in range(sample_data.shape[1]):
            data_matrix[:, i] = sample_data.iloc[:, i]

        return data_matrix

    def _get_row_array(self, row_number):
        """

        :return:
        """
        row_array = np.zeros(self._data_matrix.shape[1])

        for i in range(self._data_matrix.shape[1]):
            row_array[i] = self._data_matrix[row_number, i]

        return row_array
