import sys
import os
import numpy as np
import pandas as pd
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QLabel, QPushButton, QComboBox,
                            QScrollArea, QMessageBox, QFrame, QGridLayout,
                            QDialog)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

# Constants
MAX_LIKED_DISLIKED_RESTAURANTS = 10  # Maximum number of restaurants that can be liked/disliked
RESULTS_PER_PAGE = 3  # Number of results to show per page


class RestaurantButton(QPushButton):
    """Custom button for restaurant selection with a signal for removal"""
    removeRequested = pyqtSignal(str)
    
    def __init__(self, restaurant_name, parent=None):
        super().__init__(parent)
        self.restaurant_name = restaurant_name
        self.setStyleSheet("""
            QPushButton {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 5px;
                background-color: #f0f0f0;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        # Add text with delete icon on the right
        self.setText(f"{restaurant_name}")
        
        # Create a layout to position the text and X button
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 0)
        
        # Add restaurant name label
        name_label = QLabel(restaurant_name)
        layout.addWidget(name_label)
        
        # Add spacer to push X to the right
        layout.addStretch()
        
        # Add X button (larger)
        delete_button = QLabel("✕")
        delete_font = QFont()
        delete_font.setPointSize(14)  # Make the X twice as large
        delete_button.setFont(delete_font)
        delete_button.setStyleSheet("color: #555;")
        layout.addWidget(delete_button)
        
        self.setLayout(layout)
        self.clicked.connect(self.request_removal)
        
    def request_removal(self):
        self.removeRequested.emit(self.restaurant_name)


class RestaurantRecommenderGUI(QMainWindow):
    def __init__(self, recommender):
        super().__init__()
        
        # Store the recommender
        self.recommender = recommender
        
        # Store the original restaurant data (before normalization)
        self.original_data = pd.read_csv('data/restaurants.csv', sep=';')
        
        self.liked_restaurants = []
        self.disliked_restaurants = []
        
        self.init_ui()
    
    def init_ui(self):
        # Set window properties
        self.setWindowTitle("Система рекомендации ресторанов Москвы")
        self.setGeometry(100, 100, 800, 320)
        
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(10)
        
        # Create selection section
        selection_layout = QHBoxLayout()
        selection_layout.setSpacing(20)
        
        # Liked restaurants section
        liked_section = QVBoxLayout()
        liked_section.setSpacing(1)
        liked_label = QLabel("Понравившиеся рестораны:")
        liked_label.setFont(QFont("", 16, QFont.Bold))
        liked_section.addWidget(liked_label)
        
        # Dropdown for liked restaurants
        self.liked_dropdown = RestaurantDropdown(self.recommender.restaurant_names)
        self.liked_dropdown.restaurantSelected.connect(self.add_liked_restaurant)
        liked_section.addWidget(self.liked_dropdown)
        
        # Container for liked restaurant buttons
        self.liked_container = QWidget()
        self.liked_layout = QVBoxLayout(self.liked_container)
        self.liked_layout.setAlignment(Qt.AlignTop)
        self.liked_layout.setContentsMargins(0, 0, 0, 0)
        self.liked_layout.setSpacing(4)  # 4px spacing between restaurant buttons
        self.liked_container.setLayout(self.liked_layout)
        
        # Scroll area for liked restaurants
        liked_scroll = QScrollArea()
        liked_scroll.setWidgetResizable(True)
        liked_scroll.setWidget(self.liked_container)
        liked_scroll.setFixedHeight(150)
        liked_section.addWidget(liked_scroll)
        
        # Disliked restaurants section
        disliked_section = QVBoxLayout()
        disliked_section.setSpacing(1)
        disliked_label = QLabel("Не понравившиеся рестораны:")
        disliked_label.setFont(QFont("", 16, QFont.Bold))
        disliked_section.addWidget(disliked_label)
        
        # Dropdown for disliked restaurants
        self.disliked_dropdown = RestaurantDropdown(self.recommender.restaurant_names)
        self.disliked_dropdown.restaurantSelected.connect(self.add_disliked_restaurant)
        disliked_section.addWidget(self.disliked_dropdown)
        
        # Container for disliked restaurant buttons
        self.disliked_container = QWidget()
        self.disliked_layout = QVBoxLayout(self.disliked_container)
        self.disliked_layout.setAlignment(Qt.AlignTop)
        self.disliked_layout.setContentsMargins(0, 0, 0, 0)
        self.disliked_layout.setSpacing(4)  # 4px spacing between restaurant buttons
        self.disliked_container.setLayout(self.disliked_layout)
        
        # Scroll area for disliked restaurants
        disliked_scroll = QScrollArea()
        disliked_scroll.setWidgetResizable(True)
        disliked_scroll.setWidget(self.disliked_container)
        disliked_scroll.setFixedHeight(150)
        disliked_section.addWidget(disliked_scroll)
        
        # Add sections to selection layout
        selection_layout.addLayout(liked_section)
        selection_layout.addLayout(disliked_section)
        main_layout.addLayout(selection_layout)
        
        # Search button
        search_button = QPushButton("Поиск")
        search_button.clicked.connect(self.search_restaurants)
        search_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        main_layout.addWidget(search_button)
        
        # Set main layout to central widget
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def add_liked_restaurant(self, restaurant_name):
        # Check if the restaurant is already in disliked restaurants
        if restaurant_name in self.disliked_restaurants:
            QMessageBox.warning(self, "Предупреждение", 
                               f"Ресторан '{restaurant_name}' уже добавлен в непонравившиеся рестораны.")
            return
        
        # Check if the restaurant is already in liked restaurants
        if restaurant_name in self.liked_restaurants:
            QMessageBox.information(self, "Информация", 
                                  f"Ресторан '{restaurant_name}' уже добавлен в понравившиеся рестораны.")
            return
        
        # Check if maximum number of liked/disliked restaurants is reached
        if len(self.liked_restaurants) + len(self.disliked_restaurants) >= MAX_LIKED_DISLIKED_RESTAURANTS:
            QMessageBox.warning(self, "Предупреждение", 
                               f"Достигнуто максимальное количество выбранных ресторанов ({MAX_LIKED_DISLIKED_RESTAURANTS}).")
            return
        
        # Add restaurant to liked restaurants
        self.liked_restaurants.append(restaurant_name)
        
        # Create button for the restaurant
        restaurant_button = RestaurantButton(restaurant_name)
        restaurant_button.removeRequested.connect(self.remove_liked_restaurant)
        self.liked_layout.addWidget(restaurant_button)
    
    def remove_liked_restaurant(self, restaurant_name):
        # Remove restaurant from liked restaurants
        if restaurant_name in self.liked_restaurants:
            self.liked_restaurants.remove(restaurant_name)
        
        # Remove button
        for i in range(self.liked_layout.count()):
            widget = self.liked_layout.itemAt(i).widget()
            if isinstance(widget, RestaurantButton) and widget.restaurant_name == restaurant_name:
                widget.deleteLater()
                break
    
    def add_disliked_restaurant(self, restaurant_name):
        # Check if the restaurant is already in liked restaurants
        if restaurant_name in self.liked_restaurants:
            QMessageBox.warning(self, "Предупреждение", 
                               f"Ресторан '{restaurant_name}' уже добавлен в понравившиеся рестораны.")
            return
        
        # Check if the restaurant is already in disliked restaurants
        if restaurant_name in self.disliked_restaurants:
            QMessageBox.information(self, "Информация", 
                                  f"Ресторан '{restaurant_name}' уже добавлен в непонравившиеся рестораны.")
            return
        
        # Check if maximum number of liked/disliked restaurants is reached
        if len(self.liked_restaurants) + len(self.disliked_restaurants) >= MAX_LIKED_DISLIKED_RESTAURANTS:
            QMessageBox.warning(self, "Предупреждение", 
                               f"Достигнуто максимальное количество выбранных ресторанов ({MAX_LIKED_DISLIKED_RESTAURANTS}).")
            return
        
        # Add restaurant to disliked restaurants
        self.disliked_restaurants.append(restaurant_name)
        
        # Create button for the restaurant
        restaurant_button = RestaurantButton(restaurant_name)
        restaurant_button.removeRequested.connect(self.remove_disliked_restaurant)
        self.disliked_layout.addWidget(restaurant_button)
    
    def remove_disliked_restaurant(self, restaurant_name):
        # Remove restaurant from disliked restaurants
        if restaurant_name in self.disliked_restaurants:
            self.disliked_restaurants.remove(restaurant_name)
        
        # Remove button
        for i in range(self.disliked_layout.count()):
            widget = self.disliked_layout.itemAt(i).widget()
            if isinstance(widget, RestaurantButton) and widget.restaurant_name == restaurant_name:
                widget.deleteLater()
                break
    
    def search_restaurants(self):
        # Check if at least one restaurant is selected
        if not self.liked_restaurants and not self.disliked_restaurants:
            QMessageBox.warning(self, "Предупреждение", 
                               "Выберите хотя бы один понравившийся или непонравившийся ресторан.")
            return
        
        # Get indices of liked and disliked restaurants
        liked_indices = [self.recommender.restaurant_names.index(restaurant) for restaurant in self.liked_restaurants]
        disliked_indices = [self.recommender.restaurant_names.index(restaurant) for restaurant in self.disliked_restaurants]
        
        # Get recommendations
        results = self.recommender.get_recommendations(liked_indices, disliked_indices)
        
        # Create and show results dialog
        results_dialog = ResultsDialog(results, self.original_data, self)
        results_dialog.exec_()


class RestaurantDropdown(QComboBox):
    """Custom dropdown for restaurant selection"""
    restaurantSelected = pyqtSignal(str)
    
    def __init__(self, restaurants, parent=None):
        super().__init__(parent)
        self.addItem("Выберите ресторан...")
        self.addItems(restaurants)
        self.setCurrentIndex(0)
        self.currentIndexChanged.connect(self.on_selection)
        
    def on_selection(self, index):
        if index > 0:
            selected_restaurant = self.itemText(index)
            self.restaurantSelected.emit(selected_restaurant)
            self.setCurrentIndex(0)


class RestaurantCard(QFrame):
    """Widget to display restaurant information"""
    def __init__(self, restaurant_info, original_data, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Box)
        self.setFrameShadow(QFrame.Raised)
        self.setLineWidth(1)
        self.setStyleSheet("""
            RestaurantCard {
                border: 1px solid #ccc;
                border-radius: 8px;
                background-color: white;
                margin: 5px;
                padding: 10px;
            }
        """)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create title label with centered and larger font
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(16)  # Double the size
        
        title_label = QLabel(restaurant_info['Название'])
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)  # Center the title
        layout.addWidget(title_label)
        
        # Add horizontal line after title
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # Create grid for restaurant details
        details_grid = QGridLayout()
        row = 0
        
        # Find the original restaurant data to display original values
        restaurant_name = restaurant_info['Название']
        original_restaurant = original_data[original_data['Название'] == restaurant_name].iloc[0].to_dict() if not original_data.empty else None
        
        # Display all restaurant information
        for key, value in restaurant_info.items():
            if key != 'Название' and key != 'similarity_score':
                # Format display value
                display_value = value
                
                # Special handling for numeric fields
                if key in ['Средний чек, Р', 'Wi-Fi', 'Среднее t ожидания блюда', 'Наличие завтраков']:
                    # Use original values from data
                    if original_restaurant:
                        display_value = original_restaurant[key]
                
                if key == 'Wi-Fi':
                    label_key = QLabel(f"Wi-Fi:")
                    label_value = QLabel("Да" if int(float(value)) == 1 else "Нет")
                elif key == 'Наличие завтраков':
                    label_key = QLabel(f"Завтраки:")
                    label_value = QLabel("Да" if int(float(value)) == 1 else "Нет")
                elif key == 'Среднее t ожидания блюда':
                    label_key = QLabel(f"Время ожидания:")
                    label_value = QLabel(f"{display_value} мин.")
                else:
                    label_key = QLabel(f"{key}:")
                    label_value = QLabel(str(display_value))
                
                label_key.setStyleSheet("color: #555; font-weight: bold;")
                details_grid.addWidget(label_key, row, 0)
                details_grid.addWidget(label_value, row, 1)
                row += 1
        
        # Add similarity score
        if 'similarity_score' in restaurant_info:
            score_label = QLabel("Оценка схожести:")
            score_value = QLabel(f"{restaurant_info['similarity_score']:.2f}")
            score_label.setStyleSheet("color: #555; font-weight: bold;")
            score_value.setStyleSheet("font-weight: bold; color: #007bff;")
            details_grid.addWidget(score_label, row, 0)
            details_grid.addWidget(score_value, row, 1)
        
        # Add details grid to layout
        layout.addLayout(details_grid)
        
        # Set layout
        self.setLayout(layout)


class ResultsDialog(QDialog):
    """Dialog window to display recommendation results"""
    def __init__(self, results, original_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Результаты поиска")
        self.setMinimumSize(600, 500)
        self.current_results = results
        self.original_data = original_data
        self.current_page = 0
        
        # Create main layout
        main_layout = QVBoxLayout(self)
        
        # Results title
        results_label = QLabel("Рекомендуемые рестораны:")
        results_label.setFont(QFont("", 12, QFont.Bold))
        main_layout.addWidget(results_label)
        
        # Container for results
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setAlignment(Qt.AlignTop)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_container.setLayout(self.results_layout)
        
        # Scroll area for results
        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setWidget(self.results_container)
        main_layout.addWidget(self.results_scroll)
        
        # "More" button
        self.more_button = QPushButton("Еще")
        self.more_button.clicked.connect(self.show_more_results)
        self.more_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        main_layout.addWidget(self.more_button)
        
        # Close button
        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(self.close)
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #808080;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
        """)
        main_layout.addWidget(close_button)
        
        # Show initial results
        self.show_results_page()
    
    def clear_results(self):
        # Clear results layout
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    
    def show_results_page(self):
        # Calculate range for current page
        start_idx = self.current_page * RESULTS_PER_PAGE
        end_idx = min(start_idx + RESULTS_PER_PAGE, len(self.current_results))
        
        # If no results, show message
        if not self.current_results:
            no_results = QLabel("Нет подходящих ресторанов")
            no_results.setAlignment(Qt.AlignCenter)
            self.results_layout.addWidget(no_results)
            self.more_button.setVisible(False)
            return
        
        # Add restaurant cards for current page
        for i in range(start_idx, end_idx):
            restaurant_card = RestaurantCard(self.current_results[i], self.original_data)
            self.results_layout.addWidget(restaurant_card)
        
        # Show "More" button if there are more results
        self.more_button.setVisible(end_idx < len(self.current_results))
    
    def show_more_results(self):
        # Increment page
        self.current_page += 1
        
        # Show next page of results
        self.show_results_page()

# Main function for running the GUI
def run_gui():
    from restaurant_recommender import RestaurantRecommender  # Import only when needed
    
    app = QApplication(sys.argv)
    
    # Create recommender backend
    recommender = RestaurantRecommender()
    if not recommender.load_all_data():
        QMessageBox.critical(None, "Ошибка", "Не удалось загрузить данные о ресторанах.")
        sys.exit(1)
    
    # Create GUI
    window = RestaurantRecommenderGUI(recommender)
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_gui()