from my_stack import Stack  
from my_queue import Queue  
import networkx as nx  
import matplotlib.pyplot as plt  
import os  
import tempfile  # Для создания временных файлов
import subprocess  # Для открытия файлов


class GraphNode:
    """
    Класс для представления узла графа
    """
    def __init__(self, value):
        """
        Инициализация узла графа
        
        Args:
            value (int): Значение узла (номер вершины)
        """
        self.value = value
        self.neighbors = []
    
    def add_neighbor(self, neighbor):
        """
        Добавление соседнего узла
        
        Args:
            neighbor (GraphNode): Соседний узел
        """
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
    
    def __str__(self):
        """
        Строковое представление узла
        
        Returns:
            str: Строковое представление узла и его соседей
        """
        return f"Узел {self.value} -> Соседи: {[node.value for node in self.neighbors]}"


class Graph:
    """
    Класс для представления графа
    """
    def __init__(self, directed=False):
        """
        Инициализация пустого графа
        
        Args:
            directed (bool): Флаг, указывающий является ли граф ориентированным
        """
        self.nodes = {}  # Словарь узлов, где ключ - номер вершины, значение - объект GraphNode
        self.directed = directed
    
    def add_node(self, value):
        """
        Добавление нового узла в граф
        
        Args:
            value (int): Значение узла (номер вершины)
        
        Returns:
            GraphNode: Созданный узел
        """
        if value not in self.nodes:
            self.nodes[value] = GraphNode(value)
        return self.nodes[value]
    
    def add_edge(self, source, target):
        """
        Добавление ребра между узлами
        
        Args:
            source (int): Номер исходного узла
            target (int): Номер целевого узла
        """
        if source not in self.nodes:
            self.add_node(source)
        if target not in self.nodes:
            self.add_node(target)
            
        self.nodes[source].add_neighbor(self.nodes[target])
        
        # Если граф неориентированный, добавляем ребро и в обратном направлении
        if not self.directed:
            self.nodes[target].add_neighbor(self.nodes[source])
    
    def breadth_first_search(self, start_vertex):
        """
        Обход графа в ширину (BFS) с использованием очереди
        
        Args:
            start_vertex (int): Начальная вершина для обхода
            
        Returns:
            list: Список вершин в порядке обхода
        """
        if start_vertex not in self.nodes:
            return []
            
        visited = set()  # Множество посещенных вершин
        traversal_result = []  # Результат обхода
        
        my_queue = Queue()  # Создаем очередь для BFS
        my_queue.enqueue(start_vertex)
        visited.add(start_vertex)
        
        while not my_queue.is_empty():
            current_vertex = my_queue.dequeue()
            traversal_result.append(current_vertex)
            
            # Добавляем всех непосещенных соседей в очередь
            for neighbor in self.nodes[current_vertex].neighbors:
                if neighbor.value not in visited:
                    my_queue.enqueue(neighbor.value)
                    visited.add(neighbor.value)
        
        return traversal_result
    
    def depth_first_search(self, start_vertex):
        """
        Обход графа в глубину (DFS) с использованием стека
        
        Args:
            start_vertex (int): Начальная вершина для обхода
            
        Returns:
            list: Список вершин в порядке обхода
        """
        if start_vertex not in self.nodes:
            return []
            
        visited = set()  # Множество посещенных вершин
        traversal_result = []  # Результат обхода
        
        my_stack = Stack()  # Создаем стек для DFS
        my_stack.push(start_vertex)
        
        while not my_stack.is_empty():
            current_vertex = my_stack.pop()
            
            if current_vertex not in visited:
                visited.add(current_vertex)
                traversal_result.append(current_vertex)
                
                # Добавляем всех непосещенных соседей в стек (в обратном порядке)
                neighbors = self.nodes[current_vertex].neighbors
                for neighbor in reversed(neighbors):
                    if neighbor.value not in visited:
                        my_stack.push(neighbor.value)
        
        return traversal_result
    
    def __str__(self):
        """
        Строковое представление графа
        
        Returns:
            str: Строковое представление графа
        """
        result = f"{'Ориентированный' if self.directed else 'Неориентированный'} граф:\n"
        for node in self.nodes.values():
            result += str(node) + "\n"
        return result


def read_adjacency_matrix(file_path):
    """
    Чтение матрицы смежности из файла
    
    Args:
        file_path (str): Путь к файлу с матрицей смежности
        
    Returns:
        list: Матрица смежности в виде списка списков
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    # Преобразуем строки в числа
    matrix = []
    for line in lines:
        row = [int(val) for val in line.strip().split()]
        matrix.append(row)
    
    return matrix


def determine_graph_type(matrix):
    """
    Определить тип графа (ориентированный или неориентированный) на основе матрицы смежности
    
    Args:
        matrix (list): Матрица смежности
        
    Returns:
        bool: True если граф ориентированный, False если неориентированный
    """
    n = len(matrix)
    for i in range(n):
        for j in range(i+1, n):  # Проверяем только верхнюю половину матрицы
            if matrix[i][j] != matrix[j][i]:
                return True  # Если матрица несимметрична, граф ориентированный
    return False  # Если матрица симметрична, граф неориентированный


def create_graph_from_matrix(matrix, is_directed=None):
    """
    Создание графа из матрицы смежности
    
    Args:
        matrix (list): Матрица смежности
        is_directed (bool, optional): Флаг, указывающий является ли граф ориентированным.
                                     Если None, тип определяется автоматически.
        
    Returns:
        tuple: (Graph, NetworkX Graph) - созданный граф и его NetworkX представление
    """
    # Определяем тип графа, если не указан
    if is_directed is None:
        is_directed = determine_graph_type(matrix)
    
    graph = Graph(directed=is_directed)
    
    # Создаем NetworkX граф соответствующего типа
    if is_directed:
        nx_graph = nx.DiGraph()  # Ориентированный граф
    else:
        nx_graph = nx.Graph()  # Неориентированный граф
    
    # Добавляем все вершины
    num_vertices = len(matrix)
    for i in range(num_vertices):
        graph.add_node(i)
        nx_graph.add_node(i)  # Добавляем вершину в NetworkX граф
    
    # Добавляем рёбра на основе матрицы смежности
    for i in range(num_vertices):
        for j in range(num_vertices):
            if matrix[i][j] == 1:  # Если есть ребро
                graph.add_edge(i, j)
                nx_graph.add_edge(i, j)  # Добавляем ребро в NetworkX граф
    
    return graph, nx_graph


def generate_graph_images(nx_graph, start_vertex=None, bfs_path=None, dfs_path=None, is_tree=False):
    """
    Генерирует изображения графа и сохраняет их во временные файлы
    
    Args:
        nx_graph (nx.Graph): NetworkX граф для визуализации
        start_vertex (int, optional): Начальная вершина для обхода
        bfs_path (list, optional): Путь обхода в ширину
        dfs_path (list, optional): Путь обхода в глубину
        is_tree (bool): Флаг, указывающий является ли граф деревом
        
    Returns:
        list: Список путей к созданным изображениям
    """
    # Создаем временную директорию для изображений
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    
    # Определяем позиции узлов для визуализации
    if is_tree:
        # Для деревьев используем специальную компоновку
        try:
            # Сначала пробуем использовать настраиваемую компоновку для дерева без pygraphviz
            root = 0  # Предполагаем, что корень дерева - это вершина 0
            pos = nx.nx_agraph.graphviz_layout(nx_graph, prog='dot')
        except (ImportError, Exception):
            try:
                # Если не получилось, используем встроенную компоновку для дерева
                pos = nx.drawing.nx_pydot.pydot_layout(nx_graph, prog='dot')
            except (ImportError, Exception):
                # Если и это не получилось, используем другие алгоритмы
                if isinstance(nx_graph, nx.DiGraph):
                    try:
                        # Для ориентированных графов используем компоновку для направленных ациклических графов
                        pos = nx.planar_layout(nx_graph)
                    except:
                        pos = nx.shell_layout(nx_graph)
                else:
                    # Для неориентированных используем компоновку с радиальным размещением
                    try:
                        # Найдем центральную вершину
                        center_candidates = sorted(nx.degree_centrality(nx_graph).items(), 
                                                key=lambda x: x[1], reverse=True)
                        if center_candidates:
                            root = center_candidates[0][0]
                        pos = nx.spring_layout(nx_graph, seed=42)
                    except:
                        pos = nx.spring_layout(nx_graph, seed=42)
    else:
        # Для обычных графов используем пружинную компоновку
        pos = nx.spring_layout(nx_graph, seed=42)  # Используем seed для воспроизводимости
    
    # Создаем изображение исходного графа
    plt.figure(figsize=(10, 8))
    plt.title("Исходный граф" + (" (дерево)" if is_tree else ""))
    
    # Цвета узлов
    node_colors = ['lightblue' for _ in range(len(nx_graph.nodes()))]
    
    # Выделяем стартовую вершину, если она указана
    if start_vertex is not None:
        node_colors[start_vertex] = 'red'
    
    # Отображаем граф с узлами и ребрами
    nx.draw_networkx_nodes(nx_graph, pos, node_color=node_colors, node_size=700)
    
    # Рисуем ребра с учетом типа графа
    if isinstance(nx_graph, nx.DiGraph):
        nx.draw_networkx_edges(nx_graph, pos, width=1.5, alpha=0.7, arrowsize=20)
    else:
        nx.draw_networkx_edges(nx_graph, pos, width=1.5, alpha=0.7)
    
    # Добавляем метки узлов
    labels = {node: str(node) for node in nx_graph.nodes()}
    nx.draw_networkx_labels(nx_graph, pos, labels, font_size=15, font_weight='bold')
    
    # Сохраняем изображение
    base_graph_path = os.path.join(temp_dir, "base_graph.png")
    plt.savefig(base_graph_path)
    plt.close()
    image_paths.append(base_graph_path)
    
    # Отображаем пути обхода, если они указаны
    if bfs_path and len(bfs_path) > 1:
        plt.figure(figsize=(10, 8))
        plt.title(f"Обход в ширину (BFS) начиная с вершины {start_vertex}")
        
        # Создаем копию графа для визуализации BFS
        bfs_colors = ['lightblue' for _ in range(len(nx_graph.nodes()))]
        bfs_colors[start_vertex] = 'red'
        
        # Рисуем узлы
        nx.draw_networkx_nodes(nx_graph, pos, node_color=bfs_colors, node_size=700)
        
        # Рисуем все ребра графа (слабо выделенные)
        if isinstance(nx_graph, nx.DiGraph):
            nx.draw_networkx_edges(nx_graph, pos, width=1.0, alpha=0.3, arrowsize=15)
        else:
            nx.draw_networkx_edges(nx_graph, pos, width=1.0, alpha=0.3)
        
        # Рисуем ребра пути BFS
        bfs_edges = []
        for i in range(len(bfs_path) - 1):
            # Ищем подходящее ребро
            for neighbor in nx_graph.neighbors(bfs_path[i]):
                if neighbor == bfs_path[i+1]:
                    bfs_edges.append((bfs_path[i], bfs_path[i+1]))
                    break
        
        if isinstance(nx_graph, nx.DiGraph):
            nx.draw_networkx_edges(nx_graph, pos, edgelist=bfs_edges, width=3.0, alpha=0.9, 
                                  edge_color='r', arrowsize=25)
        else:
            nx.draw_networkx_edges(nx_graph, pos, edgelist=bfs_edges, width=3.0, alpha=0.9, 
                                  edge_color='r')
        
        # Добавляем порядок обхода как метки узлов
        bfs_labels = {node: f"{node}\n({bfs_path.index(node)+1})" for node in bfs_path}
        nx.draw_networkx_labels(nx_graph, pos, bfs_labels, font_size=15, font_weight='bold')
        
        # Сохраняем изображение
        bfs_graph_path = os.path.join(temp_dir, "bfs_graph.png")
        plt.savefig(bfs_graph_path)
        plt.close()
        image_paths.append(bfs_graph_path)
    
    if dfs_path and len(dfs_path) > 1:
        plt.figure(figsize=(10, 8))
        plt.title(f"Обход в глубину (DFS) начиная с вершины {start_vertex}")
        
        # Создаем копию графа для визуализации DFS
        dfs_colors = ['lightblue' for _ in range(len(nx_graph.nodes()))]
        dfs_colors[start_vertex] = 'red'
        
        # Рисуем узлы
        nx.draw_networkx_nodes(nx_graph, pos, node_color=dfs_colors, node_size=700)
        
        # Рисуем все ребра графа (слабо выделенные)
        if isinstance(nx_graph, nx.DiGraph):
            nx.draw_networkx_edges(nx_graph, pos, width=1.0, alpha=0.3, arrowsize=15)
        else:
            nx.draw_networkx_edges(nx_graph, pos, width=1.0, alpha=0.3)
        
        # Рисуем ребра пути DFS
        dfs_edges = []
        for i in range(len(dfs_path) - 1):
            # Ищем подходящее ребро
            for neighbor in nx_graph.neighbors(dfs_path[i]):
                if neighbor == dfs_path[i+1]:
                    dfs_edges.append((dfs_path[i], dfs_path[i+1]))
                    break
        
        if isinstance(nx_graph, nx.DiGraph):
            nx.draw_networkx_edges(nx_graph, pos, edgelist=dfs_edges, width=3.0, alpha=0.9, 
                                  edge_color='g', arrowsize=25)
        else:
            nx.draw_networkx_edges(nx_graph, pos, edgelist=dfs_edges, width=3.0, alpha=0.9, 
                                  edge_color='g')
        
        # Добавляем порядок обхода как метки узлов
        dfs_labels = {node: f"{node}\n({dfs_path.index(node)+1})" for node in dfs_path}
        nx.draw_networkx_labels(nx_graph, pos, dfs_labels, font_size=15, font_weight='bold')
        
        # Сохраняем изображение
        dfs_graph_path = os.path.join(temp_dir, "dfs_graph.png")
        plt.savefig(dfs_graph_path)
        plt.close()
        image_paths.append(dfs_graph_path)
    
    return image_paths


def open_image(image_path):
    """
    Открывает изображение с помощью системного средства просмотра
    
    Args:
        image_path (str): Путь к изображению
    """
    try:
        # Определяем ОС и выбираем соответствующую команду
        if os.name == 'posix':  # macOS или Linux
            if 'darwin' in os.sys.platform:  # macOS
                subprocess.call(('open', image_path))
            else:  # Linux
                subprocess.call(('xdg-open', image_path))
        elif os.name == 'nt':  # Windows
            os.startfile(image_path)
    except Exception as e:
        print(f"Ошибка при открытии изображения: {e}")


def main():
    try:
        # Проверяем наличие необходимых библиотек
        import networkx
        import matplotlib.pyplot
    except ImportError:
        print("Ошибка: для визуализации графа необходимо установить библиотеки networkx и matplotlib.")
        print("Установите их с помощью команды: pip install networkx matplotlib")
        return
    
    # Проверяем наличие папки data
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Ошибка: папка {data_dir} не найдена. Создайте папку и поместите в нее файлы с матрицами смежности.")
        return
    
    # Получаем список файлов с матрицами смежности в папке data
    matrix_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    if not matrix_files:
        print(f"В папке {data_dir} не найдено файлов с матрицами смежности (.txt).")
        return
    
    # Выводим список доступных файлов на выбор
    print("Доступные файлы с матрицами смежности:")
    for i, file_name in enumerate(matrix_files, 1):
        print(f"{i}. {file_name}")
    
    # Пользователь выбирает файл
    while True:
        try:
            choice = int(input(f"Выберите файл (1-{len(matrix_files)}): "))
            if 1 <= choice <= len(matrix_files):
                break
            else:
                print(f"Пожалуйста, введите число от 1 до {len(matrix_files)}.")
        except ValueError:
            print("Пожалуйста, введите корректное число.")
    
    # Получаем путь к выбранному файлу
    file_name = matrix_files[choice - 1]
    file_path = os.path.join(data_dir, file_name)
    
    try:
        # Проверяем, является ли граф деревом (на основе имени файла)
        is_tree = file_name.startswith('tree')
        
        # Читаем матрицу смежности из файла
        adjacency_matrix = read_adjacency_matrix(file_path)
        
        # Определяем тип графа автоматически (ориентированный или неориентированный)
        is_directed = determine_graph_type(adjacency_matrix)
        
        # Создаем граф на основе матрицы смежности
        graph, nx_graph = create_graph_from_matrix(adjacency_matrix, is_directed)
        
        # Выводим структуру графа в текстовом виде
        print(graph)
        
        # Получаем стартовую вершину для обхода
        start_vertex = int(input(f"Введите стартовую вершину (0-{len(adjacency_matrix)-1}): "))
        
        # Проверяем корректность ввода
        if start_vertex < 0 or start_vertex >= len(adjacency_matrix):
            print(f"Ошибка: вершина должна быть в диапазоне от 0 до {len(adjacency_matrix)-1}")
            return
        
        # Выполняем обход в ширину (BFS)
        bfs_result = graph.breadth_first_search(start_vertex)
        print(f"Результат обхода в ширину (BFS) начиная с вершины {start_vertex}:")
        print(" -> ".join(map(str, bfs_result)))
        
        # Выполняем обход в глубину (DFS)
        dfs_result = graph.depth_first_search(start_vertex)
        print(f"Результат обхода в глубину (DFS) начиная с вершины {start_vertex}:")
        print(" -> ".join(map(str, dfs_result)))
        
        # Генерируем изображения графа
        print("Создание визуализаций графа...")
        image_paths = generate_graph_images(nx_graph, start_vertex, bfs_result, dfs_result, is_tree)
        
        # Открываем все изображения для просмотра
        print("Открытие визуализаций...")
        for image_path in image_paths:
            open_image(image_path)
        
        # Ждем, пока пользователь закончит просмотр
        input("Нажмите Enter для завершения программы и удаления временных файлов...")
        
        # Удаляем временные файлы
        for image_path in image_paths:
            try:
                os.remove(image_path)
            except:
                pass
        try:
            os.rmdir(os.path.dirname(image_paths[0]))  # Удаляем временную директорию
        except:
            pass
        
    except FileNotFoundError:
        print(f"Ошибка: файл {file_path} не найден")
    except ValueError as e:
        print(f"Ошибка при обработке данных: {e}")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
        # Добавляем вывод более подробной информации об ошибке для отладки
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()