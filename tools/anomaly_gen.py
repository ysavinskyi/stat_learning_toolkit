import random


class AnomalyGenerator:
    """
    Містить логіку генерації випадкових аномалій в переданій вибірці
    """
    def __init__(self, dist):
        """
        Ініціалізує клас
        :param dist: продукт класу Distribution (tuple)
        """
        self._distribution = dist[0]
        self._distribution_type = dist[1]
        self.anomaly_indexes = None

    def create_anomalies(self, number, scale):
        """
        Створює випадкові аномалії за обраною кількістю та множником значення виміру у вибірці
        :param number: кількість аномалій
        :param scale: множик що буде застосований до значення на яке припадає створення аномалії
        """
        anomalies_scale = scale
        anomaly_indexes = random.sample(range(0, len(self._distribution)), number)
        self.anomaly_indexes = anomaly_indexes

        self._distribution[anomaly_indexes] *= anomalies_scale

    def get_distribution(self):
        """
        Повертає вибірку до якої застосовані аномальні виміри
        :return: продукт класу Distribution (tuple)
        """
        return self._distribution, self._distribution_type
