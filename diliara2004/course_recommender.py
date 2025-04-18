#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Default distance method (3 = Euclidean)
DEFAULT_DISTANCE_METHOD = 3

class CourseRecommender:
    def __init__(self):
        self.course_data = None
        self.course_names = []
        self.features = {'числовые': {}, 'категориальные': {}}
        self.similarity_matrices = {}
        self.proximity_matrix = None
        self.distance_method = DEFAULT_DISTANCE_METHOD  # Euclidean

    def load_all_data(self):
        # Load course data
        try:
            self.course_data = pd.read_csv('data/courses.csv', sep=';')
            self.course_names = self.course_data['Название'].tolist()
        except Exception as e:
            print(f"Ошибка при загрузке данных о программах: {e}")
            return False

        # Load university similarity matrix
        try:
            self.similarity_matrices['Название университета'] = pd.read_csv('data/universities_similarity.csv', sep=';', index_col=0)
        except Exception as e:
            print(f"Ошибка при загрузке матрицы сходства университетов: {e}")
            return False

        # Load city similarity matrix
        try:
            self.similarity_matrices['Расположение'] = pd.read_csv('data/cities_similarity.csv', sep=';', index_col=0)
        except Exception as e:
            print(f"Ошибка при загрузке матрицы сходства городов: {e}")
            return False
            
        # Load exams similarity matrix
        try:
            self.similarity_matrices['Языковые требования'] = pd.read_csv('data/exams_similarity.csv', sep=';', index_col=0)
        except Exception as e:
            print(f"Ошибка при загрузке матрицы сходства экзаменов: {e}")
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
            numeric_data = self.course_data[numeric_columns].copy()
            for col in numeric_columns:
                numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
            
            # Normalize numeric data
            scaler = MinMaxScaler()
            self.course_data[numeric_columns] = scaler.fit_transform(numeric_data)
        
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

    def calculate_similarity(self, course1, course2):
        # Extract numeric features
        numeric_features1 = []
        numeric_features2 = []
        
        for feature, weight in self.features['числовые'].items():
            if feature in course1 and feature in course2:
                val1 = float(course1[feature]) * weight
                val2 = float(course2[feature]) * weight
                numeric_features1.append(val1)
                numeric_features2.append(val2)
        
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
            if feature in self.similarity_matrices and feature in course1 and feature in course2:
                val1 = course1[feature]
                val2 = course2[feature]
                
                if val1 in self.similarity_matrices[feature].index and val2 in self.similarity_matrices[feature].columns:
                    similarity = float(self.similarity_matrices[feature].loc[val1, val2]) * weight
                    categorical_similarities.append(similarity)
        
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
        
        n_courses = len(self.course_data)
        proximity_matrix = np.ones((n_courses, n_courses))
        
        # Calculate similarity for each pair of courses
        for i in range(n_courses):
            for j in range(i + 1, n_courses):
                course1 = self.course_data.iloc[i]
                course2 = self.course_data.iloc[j]
                
                similarity = self.calculate_similarity(course1, course2)
                
                proximity_matrix[i, j] = similarity
                proximity_matrix[j, i] = similarity
        
        return proximity_matrix

    def get_recommendations(self, liked_indices, disliked_indices, top_n=None):
        """
        Get course recommendations based on liked and disliked courses.
        
        Args:
            liked_indices: List of indices of liked courses
            disliked_indices: List of indices of disliked courses
            top_n: Number of recommendations to return
            
        Returns:
            List of dictionaries with course info and combined similarity score
        """
        if top_n is None:
            top_n = len(self.course_data)
            
        n_courses = len(self.course_data)
        combined_scores = np.zeros(n_courses)
        
        # Add similarity scores for liked courses
        for idx in liked_indices:
            for i in range(n_courses):
                combined_scores[i] += self.proximity_matrix[idx, i]
        
        # Subtract similarity scores for disliked courses
        for idx in disliked_indices:
            for i in range(n_courses):
                combined_scores[i] -= self.proximity_matrix[idx, i]
        
        # Normalize the scores by the total number of courses used as reference
        total_reference_courses = len(liked_indices) + len(disliked_indices)
        if total_reference_courses > 0:
            combined_scores = combined_scores / total_reference_courses
        
        # Create a list of (index, score) tuples, excluding liked/disliked courses
        excluded_indices = liked_indices + disliked_indices
        recommendations = [(i, combined_scores[i]) for i in range(n_courses) if i not in excluded_indices]
        
        # Sort by score in descending order
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to list of dictionaries with course info
        result = []
        for idx, score in recommendations[:top_n]:
            course_info = self.course_data.iloc[idx].to_dict()
            course_info['similarity_score'] = score
            result.append(course_info)
        
        return result


if __name__ == "__main__":
    # Simple test of the recommender
    recommender = CourseRecommender()
    if recommender.load_all_data():
        print("Данные успешно загружены")
        print(f"Найдено {len(recommender.course_names)} программ")
        
        # Find indices for a couple of test courses
        test_course_name = recommender.course_names[0]
        test_idx = 0
        print(f"Тестовая программа: {test_course_name} (индекс {test_idx})")
        
        # Get recommendations based on one liked course
        results = recommender.get_recommendations([test_idx], [], 3)
        print("\nРекомендации на основе одной понравившейся программы:")
        for i, course in enumerate(results):
            print(f"{i+1}. {course['Название']} (схожесть: {course['similarity_score']:.2f})")
    else:
        print("Ошибка загрузки данных")