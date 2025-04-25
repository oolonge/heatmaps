#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import sys
import argparse

# Функция для вычисления косинусного сходства
def calculate_cosine_similarity(vector1, vector2):
    if np.all(np.abs(vector1) < 1e-10) or np.all(np.abs(vector2) < 1e-10):
        return 0.0
    
    dot_product = np.dot(vector1, vector2)
    norm1 = np.sqrt(np.sum(vector1**2))
    norm2 = np.sqrt(np.sum(vector2**2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    similarity = max(0.0, min(1.0, (similarity + 1) / 2))
    
    return similarity

# Функция для вычисления манхэттенского сходства
def calculate_manhattan_similarity(vector1, vector2):
    n = len(vector1)
    if n == 0:
        return 1.0
    
    v1_norm = np.array(vector1) / (np.max(np.abs(vector1)) if np.max(np.abs(vector1)) > 0 else 1)
    v2_norm = np.array(vector2) / (np.max(np.abs(vector2)) if np.max(np.abs(vector2)) > 0 else 1)
    
    distance = np.sum(np.abs(v1_norm - v2_norm))
    max_possible_distance = n * 2
    similarity = 1.0 - (distance / max_possible_distance)
    
    return max(0.0, min(1.0, similarity))

# Функция для вычисления евклидова сходства
def calculate_euclidean_similarity(vector1, vector2):
    n = len(vector1)
    if n == 0:
        return 1.0
    
    v1_norm = np.array(vector1) / (np.max(np.abs(vector1)) if np.max(np.abs(vector1)) > 0 else 1)
    v2_norm = np.array(vector2) / (np.max(np.abs(vector2)) if np.max(np.abs(vector2)) > 0 else 1)
    
    squared_distance = np.sum((v1_norm - v2_norm) ** 2)
    distance = np.sqrt(squared_distance)
    
    max_possible_distance = np.sqrt(n * 4)
    similarity = 1.0 - (distance / max_possible_distance)
    
    return max(0.0, min(1.0, similarity))

# Класс для создания тепловой карты близости косметических продуктов
class CosmeticsProximityMap:
    def __init__(self):
        self.cosmetics_data = None
        self.cosmetics_names = []
        self.features = {'числовые': {}, 'категориальные': {}}
        self.distance_method = 1  # 1 - косинусная, 2 - манхэттенская, 3 - евклидова
        self.similarity_matrices = {}
        
    def load_all_data(self):
        # Загружаем данные о косметических продуктах
        try:
            self.cosmetics_data = pd.read_csv('data/cosmetics.csv', sep=';')
            self.cosmetics_names = self.cosmetics_data['Название'].tolist()
        except Exception as e:
            print(f"Ошибка при загрузке данных о косметических продуктах: {e}")
            return False
        
        # Загружаем матрицу сходства стран
        try:
            self.similarity_matrices['производитель'] = pd.read_csv('data/countries_similarity.csv', sep=';', index_col=0)
        except Exception as e:
            print(f"Ошибка при загрузке матрицы сходства стран: {e}")
            return False
        
        # Загружаем матрицу сходства типов
        try:
            self.similarity_matrices['тип'] = pd.read_csv('data/types_similarity.csv', sep=';', index_col=0)
        except Exception as e:
            print(f"Ошибка при загрузке матрицы сходства типов: {e}")
            return False
        
        # Загружаем матрицу сходства областей
        try:
            self.similarity_matrices['область'] = pd.read_csv('data/areas_similarity.csv', sep=';', index_col=0)
        except Exception as e:
            print(f"Ошибка при загрузке матрицы сходства областей: {e}")
            return False
        
        # Загружаем матрицу сходства действий
        try:
            self.similarity_matrices['действие'] = pd.read_csv('data/actions_similarity.csv', sep=';', index_col=0)
        except Exception as e:
            print(f"Ошибка при загрузке матрицы сходства действий: {e}")
            return False
        
        # Загружаем матрицу сходства годов
        try:
            years_sim = pd.read_csv('data/years_similarity.csv', sep=';', index_col=0)
            # Преобразуем индексы в строки для правильного сравнения
            years_sim.index = years_sim.index.astype(str)
            years_sim.columns = years_sim.columns.astype(str)
            self.similarity_matrices['год презентации'] = years_sim
        except Exception as e:
            print(f"Ошибка при загрузке матрицы сходства годов: {e}")
            return False
        
        # Загружаем файл конфигурации с весами признаков
        try:
            config_data = pd.read_csv('data/config.csv', sep=';')
            for _, row in config_data.iterrows():
                if row['Тип'] == 'числовой':
                    self.features['числовые'][row['Признак']] = float(row['Вес'])
                elif row['Тип'] == 'категориальный':
                    self.features['категориальные'][row['Признак']] = float(row['Вес'])
        except Exception as e:
            print(f"Ошибка при загрузке файла конфигурации: {e}")
            return False
        
        return True
    
    def preprocess_data(self):
        numeric_columns = list(self.features['числовые'].keys())
        if numeric_columns:
            numeric_data = self.cosmetics_data[numeric_columns].copy()
            for col in numeric_columns:
                numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
            
            scaler = MinMaxScaler()
            self.cosmetics_data[numeric_columns] = scaler.fit_transform(numeric_data)
        
        return True
    
    def set_distance_method(self, method):
        if method in [1, 2, 3]:
            self.distance_method = method
            return True
        return False
    
    def calculate_similarity(self, product1, product2):
        # Извлекаем значения числовых признаков
        numeric_features1 = []
        numeric_features2 = []
        
        for feature, weight in self.features['числовые'].items():
            if feature in product1 and feature in product2:
                try:
                    val1 = float(product1[feature]) * weight
                    val2 = float(product2[feature]) * weight
                    numeric_features1.append(val1)
                    numeric_features2.append(val2)
                except (ValueError, TypeError):
                    pass  # Пропускаем признаки, которые не могут быть преобразованы в числа
        
        # Вычисляем сходство для числовых признаков
        if numeric_features1 and numeric_features2:
            numeric_vector1 = np.array(numeric_features1)
            numeric_vector2 = np.array(numeric_features2)
            
            if self.distance_method == 1:
                numeric_similarity = calculate_cosine_similarity(numeric_vector1, numeric_vector2)
            elif self.distance_method == 2:
                numeric_similarity = calculate_manhattan_similarity(numeric_vector1, numeric_vector2)
            else:
                numeric_similarity = calculate_euclidean_similarity(numeric_vector1, numeric_vector2)
        else:
            numeric_similarity = 0.0
        
        # Вычисляем сходство для категориальных признаков
        categorical_similarities = []
        
        for feature, weight in self.features['категориальные'].items():
            if feature in self.similarity_matrices and feature in product1 and feature in product2:
                val1 = str(product1[feature])
                val2 = str(product2[feature])
                
                # Проверяем, есть ли значения в матрице сходства
                if val1 in self.similarity_matrices[feature].index and val2 in self.similarity_matrices[feature].columns:
                    try:
                        similarity = float(self.similarity_matrices[feature].loc[val1, val2]) * weight
                        categorical_similarities.append(similarity)
                    except (ValueError, TypeError):
                        pass  # Пропускаем признаки, где сходство не может быть вычислено
        
        # Рассчитываем общее сходство
        if not categorical_similarities and numeric_features1 and numeric_features2:
            return numeric_similarity
        elif categorical_similarities and not (numeric_features1 and numeric_features2):
            return sum(categorical_similarities) / len(categorical_similarities)
        elif categorical_similarities and numeric_features1 and numeric_features2:
            categorical_similarity = sum(categorical_similarities) / len(categorical_similarities)
            # Вычисляем взвешенное среднее числовых и категориальных сходств
            overall_similarity = (numeric_similarity + categorical_similarity) / 2
            return overall_similarity
        else:
            return 0.0
    
    def calculate_proximity_matrix(self):
        self.preprocess_data()
        
        n_products = len(self.cosmetics_data)
        proximity_matrix = np.ones((n_products, n_products))
        
        # Для каждой пары косметических продуктов
        for i in range(n_products):
            for j in range(i + 1, n_products):
                product1 = self.cosmetics_data.iloc[i]
                product2 = self.cosmetics_data.iloc[j]
                
                similarity = self.calculate_similarity(product1, product2)
                
                proximity_matrix[i, j] = similarity
                proximity_matrix[j, i] = similarity
        
        return proximity_matrix
    
    def create_heatmap(self, save_file=False):
        proximity_matrix = self.calculate_proximity_matrix()
        
        plt.figure(figsize=(10, 7))
        sns.set(font_scale=0.6)  # Уменьшаем размер шрифта для вмещения длинных названий
        
        method_names = {1: "косинусного", 2: "манхэттенского", 3: "евклидова"}
        
        ax = sns.heatmap(
            proximity_matrix,
            annot=True,
            cmap="YlGnBu",
            xticklabels=self.cosmetics_names,
            yticklabels=self.cosmetics_names,
            vmin=0,
            vmax=1,
            linewidths=0.5,
            fmt=".2f"
        )
        
        plt.title(f"Тепловая карта {method_names[self.distance_method]} сходства косметических продуктов", fontsize=16)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_file:
            method_names_file = {1: "cosine", 2: "manhattan", 3: "euclidean"}
            output_file = f"cosmetics_heatmap_{method_names_file[self.distance_method]}.png"
            plt.savefig(output_file)
            print(f"Тепловая карта сохранена в файле {output_file}")
        else:
            plt.show()
        
        return proximity_matrix

def main():
    parser = argparse.ArgumentParser(description='Программа для создания тепловой карты близости косметических продуктов')
    parser.add_argument('-a', '--all', action='store_true', help='Создать тепловые карты для всех методов расчета расстояния')
    args = parser.parse_args()
    
    proximity_map = CosmeticsProximityMap()
    
    if not proximity_map.load_all_data():
        print("Ошибка при загрузке данных. Проверьте наличие всех необходимых файлов в директории 'data'.")
        sys.exit(1)
    
    if args.all:
        for method in [1, 2, 3]:
            proximity_map.set_distance_method(method)
            proximity_map.create_heatmap(save_file=True)
        print("Все тепловые карты успешно созданы!")
    else:
        while True:
            try:
                method = int(input("Выберите формулу расчета расстояния: Косинусная - 1, Манхэттенская - 2, Евклидова - 3: "))
                if proximity_map.set_distance_method(method):
                    break
                print("Ошибка: Введите число от 1 до 3.")
            except ValueError:
                print("Ошибка: Введите число от 1 до 3.")
        
        proximity_map.create_heatmap(save_file=False)

if __name__ == "__main__":
    main()
