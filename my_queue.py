class Queue:
    def __init__(self):
        self.items = []
    
    def enqueue(self, item):
        if not isinstance(item, int):
            raise TypeError("Очередь работает только с целыми числами")
        self.items.append(item)
    
    def dequeue(self):
        if self.is_empty():
            raise IndexError("Невозможно удалить элемент из пустой очереди")
        return self.items.pop(0)
    
    def front(self):
        if self.is_empty():
            raise IndexError("Невозможно просмотреть элемент в пустой очереди")
        return self.items[0]
    
    def rear(self):
        if self.is_empty():
            raise IndexError("Невозможно просмотреть элемент в пустой очереди")
        return self.items[-1]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)
    
    def clear(self):
        self.items = []
    
    def __str__(self):
        return f"q{self.items}"


def main():
    # Создание экземпляра очереди
    queue = Queue()
    print("Создана новая очередь:", queue)
    
    # Пример использования enqueue()
    print("\n--- Демонстрация enqueue() ---")
    queue.enqueue(10)
    print("После queue.enqueue(10):", queue)
    queue.enqueue(20)
    print("После queue.enqueue(20):", queue)
    queue.enqueue(30)
    print("После queue.enqueue(30):", queue)
    
    # Пример использования front()
    print("\n--- Демонстрация front() ---")
    front_element = queue.front()
    print("Первый элемент (front):", front_element)
    print("Очередь после front():", queue)
    
    # Пример использования rear()
    print("\n--- Демонстрация rear() ---")
    rear_element = queue.rear()
    print("Последний элемент (rear):", rear_element)
    print("Очередь после rear():", queue)
    
    # Пример использования dequeue()
    print("\n--- Демонстрация dequeue() ---")
    dequeued_element = queue.dequeue()
    print("Удаленный элемент (dequeue):", dequeued_element)
    print("Очередь после dequeue():", queue)
    
    # Пример использования is_empty()
    print("\n--- Демонстрация is_empty() ---")
    print("Очередь пуста?", queue.is_empty())
    
    # Пример использования size()
    print("\n--- Демонстрация size() ---")
    print("Размер очереди:", queue.size())
    
    # Пример использования clear()
    print("\n--- Демонстрация clear() ---")
    queue.clear()
    print("Очередь после clear():", queue)
    print("Очередь пуста?", queue.is_empty())
    
    # Демонстрация обработки ошибок
    print("\n--- Демонстрация обработки ошибок ---")
    try:
        queue.dequeue()
    except IndexError as e:
        print("Ошибка при вызове dequeue() на пустой очереди:", e)
    
    try:
        queue.front()
    except IndexError as e:
        print("Ошибка при вызове front() на пустой очереди:", e)
    
    try:
        queue.rear()
    except IndexError as e:
        print("Ошибка при вызове rear() на пустой очереди:", e)
    
    try:
        queue.enqueue("строка")
    except TypeError as e:
        print("Ошибка при добавлении не целого числа:", e)


if __name__ == "__main__":
    main()