#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler

# Default distance method (3 = Euclidean)
DEFAULT_DISTANCE_METHOD = 3

class RestaurantRecommender:
    def __init__(self):
        self.restaurant_data = None
        self.restaurant_names = []
        self.features = {'числовые': {}, 'категориальные': {}}
        self.similarity_matrices = {}
        self.proximity_matrix = None
        self.distance_method = DEFAULT_DISTANCE_METHOD  # Euclidean

    def load_all_data(self):
        # Load restaurant data
        try:
            self.restaurant_data = pd.read_csv('data/restaurants.csv', sep=';')
            self.restaurant_names = self.restaurant_data['Название'].tolist()
        except Exception as e:
            print(f"Ошибка при загрузке данных о ресторанах: {e}")
            return False

        # Process numeric columns
        try:
            # Convert "Средний чек, Р" to numeric format
            self.restaurant_data['Средний чек, Р'] = self.restaurant_data['Средний чек, Р'].apply(
                lambda x: float(re.search(r'(\d+)', str(x)).group(1)) if re.search(r'(\d+)', str(x)) else 0
            )
        except Exception as e:
            print(f"Ошибка при обработке числовых данных: {e}")
            return False

        # Load cuisine similarity matrix
        try:
            self.similarity_matrices['Кухня'] = pd.read_csv('data/cuisines_similarity.csv', sep=';', index_col=0)
        except Exception as e:
            print(f"Ошибка при загрузке матрицы сходства кухонь: {e}")
            return False
            
        # Load district similarity matrix
        try:
            self.similarity_matrices['Район'] = pd.read_csv('data/districts_similarity.csv', sep=';', index_col=0)
        except Exception as e:
            print(f"Ошибка при загрузке матрицы сходства районов: {e}")
            return False

        # Load feature weights
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
        
        # Calculate proximity matrix
        self.proximity_matrix = self.calculate_proximity_matrix()
        
        return True

    def preprocess_data(self):
        numeric_columns = list(self.features['числовые'].keys())
        if numeric_columns:
            numeric_data = self.restaurant_data[numeric_columns].copy()
            for col in numeric_columns:
                try:
                    numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
                except Exception as e:
                    print(f"Ошибка при преобразовании столбца {col}: {e}")
            
            # Normalize numeric data
            scaler = MinMaxScaler()
            self.restaurant_data[numeric_columns] = scaler.fit_transform(numeric_data)
        
        return True

    def calculate_euclidean_similarity(self, vector1, vector2):
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

    def calculate_similarity(self, restaurant1, restaurant2):
        # Extract numeric features
        numeric_features1 = []
        numeric_features2 = []
        
        for feature, weight in self.features['числовые'].items():
            if feature in restaurant1 and feature in restaurant2:
                try:
                    val1 = float(restaurant1[feature]) * weight
                    val2 = float(restaurant2[feature]) * weight
                    numeric_features1.append(val1)
                    numeric_features2.append(val2)
                except (ValueError, TypeError):
                    pass  # Skip features that can't be converted to numbers
        
        # Calculate similarity for numeric features
        if numeric_features1 and numeric_features2:
            numeric_vector1 = np.array(numeric_features1)
            numeric_vector2 = np.array(numeric_features2)
            
            numeric_similarity = self.calculate_euclidean_similarity(numeric_vector1, numeric_vector2)
        else:
            numeric_similarity = 0.0
        
        # Calculate similarity for categorical features
        categorical_similarities = []
        
        for feature, weight in self.features['категориальные'].items():
            if feature in self.similarity_matrices and feature in restaurant1 and feature in restaurant2:
                # Special handling for "Кухня" column that may contain multiple values
                if feature == 'Кухня':
                    cuisines1 = [c.strip() for c in str(restaurant1[feature]).split(',')]
                    cuisines2 = [c.strip() for c in str(restaurant2[feature]).split(',')]
                    
                    # Calculate average similarity between all cuisine pairs
                    cuisine_similarities = []
                    for c1 in cuisines1:
                        for c2 in cuisines2:
                            if c1 in self.similarity_matrices[feature].index and c2 in self.similarity_matrices[feature].columns:
                                similarity = float(self.similarity_matrices[feature].loc[c1, c2])
                                cuisine_similarities.append(similarity)
                    
                    if cuisine_similarities:
                        avg_cuisine_similarity = sum(cuisine_similarities) / len(cuisine_similarities)
                        categorical_similarities.append(avg_cuisine_similarity * weight)
                else:
                    val1 = restaurant1[feature]
                    val2 = restaurant2[feature]
                    
                    if val1 in self.similarity_matrices[feature].index and val2 in self.similarity_matrices[feature].columns:
                        try:
                            similarity = float(self.similarity_matrices[feature].loc[val1, val2]) * weight
                            categorical_similarities.append(similarity)
                        except (ValueError, TypeError):
                            pass  # Skip features where similarity can't be computed
        
        # Calculate overall similarity
        if not categorical_similarities and numeric_features1 and numeric_features2:
            return numeric_similarity
        elif categorical_similarities and not (numeric_features1 and numeric_features2):
            return sum(categorical_similarities) / len(categorical_similarities)
        elif categorical_similarities and numeric_features1 and numeric_features2:
            categorical_similarity = sum(categorical_similarities) / len(categorical_similarities)
            overall_similarity = (numeric_similarity + categorical_similarity) / 2
            return overall_similarity
        else:
            return 0.0

    def calculate_proximity_matrix(self):
        self.preprocess_data()
        
        n_restaurants = len(self.restaurant_data)
        proximity_matrix = np.ones((n_restaurants, n_restaurants))
        
        # Calculate similarity for each pair of restaurants
        for i in range(n_restaurants):
            for j in range(i + 1, n_restaurants):
                restaurant1 = self.restaurant_data.iloc[i]
                restaurant2 = self.restaurant_data.iloc[j]
                
                similarity = self.calculate_similarity(restaurant1, restaurant2)
                
                proximity_matrix[i, j] = similarity
                proximity_matrix[j, i] = similarity
        
        return proximity_matrix

    def get_recommendations(self, liked_indices, disliked_indices, top_n=None):
        """
        Get restaurant recommendations based on liked and disliked restaurants.
        
        Args:
            liked_indices: List of indices of liked restaurants
            disliked_indices: List of indices of disliked restaurants
            top_n: Number of recommendations to return
            
        Returns:
            List of dictionaries with restaurant info and combined similarity score
        """
        if top_n is None:
            top_n = len(self.restaurant_data)
            
        n_restaurants = len(self.restaurant_data)
        combined_scores = np.zeros(n_restaurants)
        
        # Add similarity scores for liked restaurants
        for idx in liked_indices:
            for i in range(n_restaurants):
                combined_scores[i] += self.proximity_matrix[idx, i]
        
        # Subtract similarity scores for disliked restaurants
        for idx in disliked_indices:
            for i in range(n_restaurants):
                combined_scores[i] -= self.proximity_matrix[idx, i]
        
        # Normalize the scores by the total number of restaurants used as reference
        total_reference_restaurants = len(liked_indices) + len(disliked_indices)
        if total_reference_restaurants > 0:
            combined_scores = combined_scores / total_reference_restaurants
        
        # Create a list of (index, score) tuples, excluding liked/disliked restaurants
        excluded_indices = liked_indices + disliked_indices
        recommendations = [(i, combined_scores[i]) for i in range(n_restaurants) if i not in excluded_indices]
        
        # Sort by score in descending order
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to list of dictionaries with restaurant info
        result = []
        for idx, score in recommendations[:top_n]:
            restaurant_info = self.restaurant_data.iloc[idx].to_dict()
            restaurant_info['similarity_score'] = score
            result.append(restaurant_info)
        
        return result


if __name__ == "__main__":
    # Simple test of the recommender
    recommender = RestaurantRecommender()
    if recommender.load_all_data():
        print("Данные успешно загружены")
        print(f"Найдено {len(recommender.restaurant_names)} ресторанов")
        
        # Find indices for a couple of test restaurants
        test_restaurant_name = recommender.restaurant_names[0]
        test_idx = 0
        print(f"Тестовый ресторан: {test_restaurant_name} (индекс {test_idx})")
        
        # Get recommendations based on one liked restaurant
        results = recommender.get_recommendations([test_idx], [], 3)
        print("\nРекомендации на основе одного понравившегося ресторана:")
        for i, restaurant in enumerate(results):
            print(f"{i+1}. {restaurant['Название']} (схожесть: {restaurant['similarity_score']:.2f})")
    else:
        print("Ошибка загрузки данных")