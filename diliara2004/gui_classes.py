import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QLabel, QPushButton, QComboBox,
                            QScrollArea, QMessageBox, QFrame, QGridLayout,
                            QDialog)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

# Constants
MAX_LIKED_DISLIKED_COURSES = 10  # Maximum number of courses that can be liked/disliked
RESULTS_PER_PAGE = 3  # Number of results to show per page


class CourseButton(QPushButton):
    """Custom button for course selection with a signal for removal"""
    removeRequested = pyqtSignal(str)
    
    def __init__(self, course_name, parent=None):
        super().__init__(parent)
        self.course_name = course_name
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
        self.setText(f"{course_name}")
        
        # Create a layout to position the text and X button
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 0)
        
        # Add course name label
        name_label = QLabel(course_name)
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
        self.removeRequested.emit(self.course_name)


class CourseRecommenderGUI(QMainWindow):
    def __init__(self, recommender):
        super().__init__()
        
        # Store the recommender
        self.recommender = recommender
        
        # Store the original course data (before normalization)
        self.original_data = pd.read_csv('data/courses.csv', sep=';')
        
        self.liked_courses = []
        self.disliked_courses = []
        
        self.init_ui()
    
    def init_ui(self):
        # Set window properties
        self.setWindowTitle("Система рекомендации образовательных программ в Китае")
        self.setGeometry(100, 100, 800, 320)
        
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(10)
        
        # Create selection section
        selection_layout = QHBoxLayout()
        selection_layout.setSpacing(20)
        
        # Liked courses section
        liked_section = QVBoxLayout()
        liked_section.setSpacing(1)
        liked_label = QLabel("Понравившиеся программы:")
        liked_label.setFont(QFont("", 16, QFont.Bold))
        liked_section.addWidget(liked_label)
        
        # Dropdown for liked courses
        self.liked_dropdown = CourseDropdown(self.recommender.course_names)
        self.liked_dropdown.courseSelected.connect(self.add_liked_course)
        liked_section.addWidget(self.liked_dropdown)
        
        # Container for liked course buttons
        self.liked_container = QWidget()
        self.liked_layout = QVBoxLayout(self.liked_container)
        self.liked_layout.setAlignment(Qt.AlignTop)
        self.liked_layout.setContentsMargins(0, 0, 0, 0)
        self.liked_layout.setSpacing(4)  # 4px spacing between course buttons
        self.liked_container.setLayout(self.liked_layout)
        
        # Scroll area for liked courses
        liked_scroll = QScrollArea()
        liked_scroll.setWidgetResizable(True)
        liked_scroll.setWidget(self.liked_container)
        liked_scroll.setFixedHeight(150)
        liked_section.addWidget(liked_scroll)
        
        # Disliked courses section
        disliked_section = QVBoxLayout()
        disliked_section.setSpacing(1)
        disliked_label = QLabel("Не понравившиеся программы:")
        disliked_label.setFont(QFont("", 16, QFont.Bold))
        disliked_section.addWidget(disliked_label)
        
        # Dropdown for disliked courses
        self.disliked_dropdown = CourseDropdown(self.recommender.course_names)
        self.disliked_dropdown.courseSelected.connect(self.add_disliked_course)
        disliked_section.addWidget(self.disliked_dropdown)
        
        # Container for disliked course buttons
        self.disliked_container = QWidget()
        self.disliked_layout = QVBoxLayout(self.disliked_container)
        self.disliked_layout.setAlignment(Qt.AlignTop)
        self.disliked_layout.setContentsMargins(0, 0, 0, 0)
        self.disliked_layout.setSpacing(4)  # 4px spacing between course buttons
        self.disliked_container.setLayout(self.disliked_layout)
        
        # Scroll area for disliked courses
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
        search_button.clicked.connect(self.search_courses)
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
    
    def add_liked_course(self, course_name):
        # Check if the course is already in disliked courses
        if course_name in self.disliked_courses:
            QMessageBox.warning(self, "Предупреждение", 
                               f"Программа '{course_name}' уже добавлена в непонравившиеся программы.")
            return
        
        # Check if the course is already in liked courses
        if course_name in self.liked_courses:
            QMessageBox.information(self, "Информация", 
                                  f"Программа '{course_name}' уже добавлена в понравившиеся программы.")
            return
        
        # Check if maximum number of liked/disliked courses is reached
        if len(self.liked_courses) + len(self.disliked_courses) >= MAX_LIKED_DISLIKED_COURSES:
            QMessageBox.warning(self, "Предупреждение", 
                               f"Достигнуто максимальное количество выбранных программ ({MAX_LIKED_DISLIKED_COURSES}).")
            return
        
        # Add course to liked courses
        self.liked_courses.append(course_name)
        
        # Create button for the course
        course_button = CourseButton(course_name)
        course_button.removeRequested.connect(self.remove_liked_course)
        self.liked_layout.addWidget(course_button)
    
    def remove_liked_course(self, course_name):
        # Remove course from liked courses
        if course_name in self.liked_courses:
            self.liked_courses.remove(course_name)
        
        # Remove button
        for i in range(self.liked_layout.count()):
            widget = self.liked_layout.itemAt(i).widget()
            if isinstance(widget, CourseButton) and widget.course_name == course_name:
                widget.deleteLater()
                break
    
    def add_disliked_course(self, course_name):
        # Check if the course is already in liked courses
        if course_name in self.liked_courses:
            QMessageBox.warning(self, "Предупреждение", 
                               f"Программа '{course_name}' уже добавлена в понравившиеся программы.")
            return
        
        # Check if the course is already in disliked courses
        if course_name in self.disliked_courses:
            QMessageBox.information(self, "Информация", 
                                  f"Программа '{course_name}' уже добавлена в непонравившиеся программы.")
            return
        
        # Check if maximum number of liked/disliked courses is reached
        if len(self.liked_courses) + len(self.disliked_courses) >= MAX_LIKED_DISLIKED_COURSES:
            QMessageBox.warning(self, "Предупреждение", 
                               f"Достигнуто максимальное количество выбранных программ ({MAX_LIKED_DISLIKED_COURSES}).")
            return
        
        # Add course to disliked courses
        self.disliked_courses.append(course_name)
        
        # Create button for the course
        course_button = CourseButton(course_name)
        course_button.removeRequested.connect(self.remove_disliked_course)
        self.disliked_layout.addWidget(course_button)
    
    def remove_disliked_course(self, course_name):
        # Remove course from disliked courses
        if course_name in self.disliked_courses:
            self.disliked_courses.remove(course_name)
        
        # Remove button
        for i in range(self.disliked_layout.count()):
            widget = self.disliked_layout.itemAt(i).widget()
            if isinstance(widget, CourseButton) and widget.course_name == course_name:
                widget.deleteLater()
                break
    
    def search_courses(self):
        # Check if at least one course is selected
        if not self.liked_courses and not self.disliked_courses:
            QMessageBox.warning(self, "Предупреждение", 
                               "Выберите хотя бы одну понравившуюся или непонравившуюся программу.")
            return
        
        # Get indices of liked and disliked courses
        liked_indices = [self.recommender.course_names.index(course) for course in self.liked_courses]
        disliked_indices = [self.recommender.course_names.index(course) for course in self.disliked_courses]
        
        # Get recommendations
        results = self.recommender.get_recommendations(liked_indices, disliked_indices)
        
        # Create and show results dialog
        results_dialog = ResultsDialog(results, self.original_data, self)
        results_dialog.exec_()


class CourseDropdown(QComboBox):
    """Custom dropdown for course selection"""
    courseSelected = pyqtSignal(str)
    
    def __init__(self, courses, parent=None):
        super().__init__(parent)
        self.addItem("Выберите программу...")
        self.addItems(courses)
        self.setCurrentIndex(0)
        self.currentIndexChanged.connect(self.on_selection)
        
    def on_selection(self, index):
        if index > 0:
            selected_course = self.itemText(index)
            self.courseSelected.emit(selected_course)
            self.setCurrentIndex(0)


class CourseCard(QFrame):
    """Widget to display course information"""
    def __init__(self, course_info, original_data, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Box)
        self.setFrameShadow(QFrame.Raised)
        self.setLineWidth(1)
        self.setStyleSheet("""
            CourseCard {
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
        
        title_label = QLabel(course_info['Название'])
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)  # Center the title
        layout.addWidget(title_label)
        
        # Add horizontal line after title
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # Create grid for course details
        details_grid = QGridLayout()
        row = 0
        
        # Find the original course data to display original values
        course_name = course_info['Название']
        original_course = original_data[original_data['Название'] == course_name].iloc[0].to_dict() if not original_data.empty else None
        
        # Display all course information
        for key, value in course_info.items():
            if key != 'Название' and key != 'similarity_score':
                # Format display value
                display_value = value
                
                # Special handling for numeric fields
                if key in ['Стоимость', 'Продолжительность', 'Процент трудоустройства']:
                    # Use original values from data
                    if original_course:
                        display_value = original_course[key]
                
                if key == 'Стоимость':
                    label_key = QLabel(f"Стоимость:")
                    label_value = QLabel(str(display_value) + " юаней")
                elif key == 'Продолжительность':
                    label_key = QLabel(f"Продолжительность:")
                    label_value = QLabel(str(display_value) + " мес.")
                elif key == 'Наличие стипендии':
                    label_key = QLabel(f"Наличие стипендии:")
                    label_value = QLabel("Да" if int(float(value)) == 1 else "Нет")
                else:
                    label_key = QLabel(f"{key}:")
                    label_value = QLabel(str(display_value))
                    
                label_key.setStyleSheet("color: #555; font-weight: bold;")
                details_grid.addWidget(label_key, row, 0)
                details_grid.addWidget(label_value, row, 1)
                row += 1
        
        # Add similarity score
        if 'similarity_score' in course_info:
            score_label = QLabel("Оценка схожести:")
            score_value = QLabel(f"{course_info['similarity_score']:.2f}")
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
        results_label = QLabel("Рекомендуемые программы:")
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
            no_results = QLabel("Нет подходящих программ")
            no_results.setAlignment(Qt.AlignCenter)
            self.results_layout.addWidget(no_results)
            self.more_button.setVisible(False)
            return
        
        # Add course cards for current page
        for i in range(start_idx, end_idx):
            course_card = CourseCard(self.current_results[i], self.original_data)
            self.results_layout.addWidget(course_card)
        
        # Show "More" button if there are more results
        self.more_button.setVisible(end_idx < len(self.current_results))
    
    def show_more_results(self):
        # Increment page
        self.current_page += 1
        
        # Show next page of results
        self.show_results_page()

# Main function for running the GUI
def run_gui():
    from course_recommender import CourseRecommender  # Import only when needed
    
    app = QApplication(sys.argv)
    
    # Create recommender backend
    recommender = CourseRecommender()
    if not recommender.load_all_data():
        QMessageBox.critical(None, "Ошибка", "Не удалось загрузить данные о программах.")
        sys.exit(1)
    
    # Create GUI
    window = CourseRecommenderGUI(recommender)
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_gui()