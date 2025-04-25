class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        if not isinstance(item, int):
            raise TypeError("Стек работает только с целыми числами")
        self.items.append(item)
    
    def pop(self):
        if self.is_empty():
            raise IndexError("Невозможно удалить элемент из пустого стека")
        return self.items.pop()
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Невозможно просмотреть элемент в пустом стеке")
        return self.items[-1]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)
    
    def clear(self):
        self.items = []
    
    def __str__(self):
        return f"st{self.items}<-"


def main():
    # Создание экземпляра стека
    stack = Stack()
    print("Создан новый стек:", stack)
    
    # Пример использования push()
    print("\n--- Демонстрация push() ---")
    stack.push(10)
    print("После stack.push(10):", stack)
    stack.push(20)
    print("После stack.push(20):", stack)
    stack.push(30)
    print("После stack.push(30):", stack)
    
    # Пример использования peek()
    print("\n--- Демонстрация peek() ---")
    top_element = stack.peek()
    print("Верхний элемент (peek):", top_element)
    print("Стек после peek():", stack)
    
    # Пример использования pop()
    print("\n--- Демонстрация pop() ---")
    popped_element = stack.pop()
    print("Удаленный элемент (pop):", popped_element)
    print("Стек после pop():", stack)
    
    # Пример использования is_empty()
    print("\n--- Демонстрация is_empty() ---")
    print("Стек пуст?", stack.is_empty())
    
    # Пример использования size()
    print("\n--- Демонстрация size() ---")
    print("Размер стека:", stack.size())
    
    # Пример использования clear()
    print("\n--- Демонстрация clear() ---")
    stack.clear()
    print("Стек после clear():", stack)
    print("Стек пуст?", stack.is_empty())
    
    # Демонстрация обработки ошибок
    print("\n--- Демонстрация обработки ошибок ---")
    try:
        stack.pop()
    except IndexError as e:
        print("Ошибка при вызове pop() на пустом стеке:", e)
    
    try:
        stack.peek()
    except IndexError as e:
        print("Ошибка при вызове peek() на пустом стеке:", e)
    
    try:
        stack.push("строка")
    except TypeError as e:
        print("Ошибка при добавлении не целого числа:", e)


if __name__ == "__main__":
    main()
