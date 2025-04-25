#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from PyQt5.QtWidgets import QApplication, QMessageBox

# Import the CourseRecommender class and GUI components
from cosmetics_recommender import CourseRecommender
from gui_classes import CourseRecommenderGUI


def main():
    app = QApplication(sys.argv)
    
    # Make sure data directory exists and is accessible
    if not os.path.isdir('data'):
        QMessageBox.critical(None, "Ошибка", 
                           "Директория 'data' не найдена. Убедитесь, что запускаете программу из правильной директории.")
        sys.exit(1)
    
    # Create recommender backend
    recommender = CourseRecommender()
    if not recommender.load_all_data():
        QMessageBox.critical(None, "Ошибка", 
                           "Не удалось загрузить данные о косметических продуктах. Проверьте наличие всех необходимых файлов в директории 'data'.")
        sys.exit(1)
    
    # Create and show GUI
    window = CourseRecommenderGUI(recommender)
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
