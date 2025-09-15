import pygame
from sys import exit
import random
import numpy as np
import time 
import gc


def reset_game():
    global snake_x_pos, snake_y_pos, snake_move_x, snake_move_y, snake_body, \
        apple_x_pos, apple_y_pos, apple_rect, score, fps, current_direction
    
    snake_body.clear() 

    # Сброс змейки
    snake_x_pos = GRID_WIDTH // 2 * CELL_SIZE + CELL_SIZE / 2
    snake_y_pos = GRID_HEIGHT // 2 * CELL_SIZE + FIELD_Y + CELL_SIZE / 2
    snake_move_x, snake_move_y = 0, -snake_speed
    current_direction = 1  # Начальное направление - вверх
    
    snake_body = [
        (snake_x_pos, snake_y_pos),
        (snake_x_pos, snake_y_pos + CELL_SIZE),
        (snake_x_pos, snake_y_pos + 2 * CELL_SIZE)
    ]
    
    gc.collect() 

    # Новое яблоко
    apple_surf = pygame.Surface((CELL_SIZE // 2, CELL_SIZE // 2), pygame.SRCALPHA)
    apple_surf.fill('Red')
    apple_x_pos = apple_spawn(snake_body, True)
    apple_y_pos = apple_spawn(snake_body, False)
    apple_rect = apple_surf.get_rect(center=(apple_x_pos, apple_y_pos))
    
    # Сброс счета и скорости
    score = 0
    fps = 10


def save_q_table(filename="qtable.npy"):
    np.save(filename, Q)
    print(f"Q-таблица сохранена в {filename}")


def load_q_table(filename="qtable.npy"):
    try:
        loaded_q = np.load(filename)
        print(f"Q-таблица загружена из {filename}")
        return loaded_q
    except FileNotFoundError:
        print(f"Файл {filename} не найден, создается новая Q-таблица")
        return np.zeros((GRID_WIDTH * 2 + 1, #dx
                        GRID_HEIGHT * 2 + 1, #dy
                        4,                   #directions
                        2, 2, 2, 2,          # danger (0 или 1)
                        4,                   # actions
                        ), dtype=np.float32)

def apple_spawn(snake_body, index): # True - x / False - y
    apple_x_pos = random.randint(0, GRID_WIDTH - 1) * CELL_SIZE + CELL_SIZE / 2
    apple_y_pos = random.randint(0, GRID_HEIGHT - 1) * CELL_SIZE + FIELD_Y + CELL_SIZE / 2
    if index:
        while any(segment[0] == apple_x_pos for segment in snake_body) and any(segment[1] == apple_y_pos for segment in snake_body):
            apple_x_pos = random.randint(0, GRID_WIDTH - 1) * CELL_SIZE + CELL_SIZE / 2
        return apple_x_pos
    else:
        while any(segment[0] == apple_x_pos for segment in snake_body) and any(segment[1] == apple_y_pos for segment in snake_body):
            apple_y_pos = random.randint(0, GRID_HEIGHT - 1) * CELL_SIZE + FIELD_Y + CELL_SIZE / 2
        return apple_y_pos

pygame.init()

fps = 10
screen = pygame.display.set_mode((900, 1000), pygame.HWSURFACE | pygame.DOUBLEBUF) #Рендре видеокартой + в фоновом режиме = оптимизация

pygame.display.set_caption('Smart snake')
clock = pygame.time.Clock()
font = pygame.font.Font(None, 30)  # Замените на ваш шрифт

# Параметры поля
FIELD_X, FIELD_Y = 0, 100
FIELD_WIDTH, FIELD_HEIGHT = 900, 900
CELL_SIZE = 50
GRID_WIDTH = FIELD_WIDTH // CELL_SIZE
GRID_HEIGHT = FIELD_HEIGHT // CELL_SIZE

background = pygame.Surface((900, 1000))
background.fill(0)
for row in range(GRID_HEIGHT):
    for col in range(GRID_WIDTH):
        color = 'chartreuse4' if (row + col) % 2 == 0 else 'chartreuse3'
        pygame.draw.rect(background, color, (col*CELL_SIZE, FIELD_Y+row*CELL_SIZE, CELL_SIZE, CELL_SIZE))
pygame.draw.rect(background, 'white', (FIELD_X, FIELD_Y, FIELD_WIDTH, FIELD_HEIGHT), 5)


# Инициализация змейки
snake_surf = pygame.Surface((CELL_SIZE // 1.5, CELL_SIZE // 1.5))
snake_surf.fill('cornflowerblue')

snake_head_surf = pygame.Surface((CELL_SIZE // 1.5, CELL_SIZE // 1.5))
snake_head_surf.fill('darkblue')  # Тёмный оттенок головы

snake_x_pos = GRID_WIDTH // 2 * CELL_SIZE + CELL_SIZE / 2
snake_y_pos = GRID_HEIGHT // 2 * CELL_SIZE + FIELD_Y + CELL_SIZE / 2
snake_speed = CELL_SIZE
snake_move_x, snake_move_y = 0, -snake_speed  # Начальное движение вверх

snake_body = [
    (snake_x_pos, snake_y_pos),
    (snake_x_pos, snake_y_pos + CELL_SIZE),
    (snake_x_pos, snake_y_pos + 2 * CELL_SIZE)
]

# Яблоко
apple_x_pos = apple_spawn(snake_body, True)
apple_y_pos = apple_spawn(snake_body, False)
apple_surf = pygame.Surface((CELL_SIZE // 2, CELL_SIZE // 2))
apple_surf.fill('Red')
apple_rect = apple_surf.get_rect(center=(apple_x_pos, apple_y_pos))

# Q-learning параметры
alpha = 0.2 # Скорость обучения
gamma = 0.9 # Коэфицент дисконтинирования
epsilon = 0.0 

# Дискретизация состояний
def discretize(value, cell_size):
    return int(round(value / cell_size))

def get_state(snake_body, apple_pos, direction):
    # Текущие координаты головы
    head_x, head_y = snake_body[0]
    
    # Дискретизированные координаты яблока относительно головы
    dx = discretize(apple_pos[0] - head_x, CELL_SIZE)
    dy = discretize(apple_pos[1] - head_y, CELL_SIZE)
    
    # Проверка соседних клеток (0 = свободно, 1 = стена/тело)
    danger_left = 1 if (head_x - CELL_SIZE < FIELD_X or 
                        any((head_x - CELL_SIZE, head_y) == segment for segment in snake_body[1:])) else 0
    danger_right = 1 if (head_x + CELL_SIZE >= FIELD_X + FIELD_WIDTH or 
                        any((head_x + CELL_SIZE, head_y) == segment for segment in snake_body[1:])) else 0
    danger_up = 1 if (head_y - CELL_SIZE < FIELD_Y or 
                    any((head_x, head_y - CELL_SIZE) == segment for segment in snake_body[1:])) else 0
    danger_down = 1 if (head_y + CELL_SIZE >= FIELD_Y + FIELD_HEIGHT or 
                        any((head_x, head_y + CELL_SIZE) == segment for segment in snake_body[1:])) else 0
    
    # Возвращаем кортеж с полным состоянием
    return (dx, dy, direction, 
            danger_left, danger_right, danger_up, danger_down
            )

# Инициализация Q-таблицы с разумными размерами
Q = load_q_table()

def choose_action(state):
    if random.random() < epsilon:
        return random.randint(0, 3)  # 0=влево, 1=вверх, 2=вправо, 3=вниз
    else:
        return np.argmax(Q[state])

def update_q_table(state, action, reward, next_state):
    Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action]) # уравнение Беллмана

def get_reward(snake_body, apple_pos, game_over):
    snake_head = snake_body[0]
    head_x, head_y = snake_head
    if game_over:
        return -100
    if (head_x == apple_pos[0] and head_y == apple_pos[1]):
        return 100
    next_x = head_x + snake_move_x
    next_y = head_y + snake_move_y
    if any((next_x, next_y) == segment for segment in snake_body[1:]):
        return -50  # Сильный штраф за опасный ход
    

    # Добавляем награду за приближение к яблоку
    # Награда за приближение с нелинейным усилением
    prev_distance = ((apple_pos[0] - (head_x - snake_move_x))**2 + 
                    (apple_pos[1] - (head_y - snake_move_y))**2)**0.5
    curr_distance = ((apple_pos[0] - head_x)**2 + 
                    (apple_pos[1] - head_y)**2)**0.5
    
    distance_reward = 0.0
    if curr_distance < prev_distance:  # Движемся к яблоку
        progress = (prev_distance - curr_distance) / CELL_SIZE
        distance_reward = 5.0 * (progress ** 2)  # Квадратичное усиление (max = 10)
        
    elif curr_distance > prev_distance:  # Удаляемся
        distance_reward = -5.0
    return distance_reward

score = 0
current_direction = 1  # 0=влево, 1=вверх, 2=вправо, 3=вниз
last_cleanup = time.time()  # Для периодической очистки памяти

while True:
    
    pygame.event.pump() 
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            save_q_table()  # Сохраняем перед выходом
            pygame.quit()
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:  # Закрытие по кнопке Q
                save_q_table()
                pygame.quit()
                exit()

    # Получаем текущее состояние
    state = get_state(snake_body, (apple_x_pos, apple_y_pos), current_direction)
    
    # Выбираем действие
    action = choose_action(state)
    
    # Обновляем направление
    if action == 0:  # Влево
        new_direction = (current_direction - 1) % 4
    elif action == 2:  # Вправо
        new_direction = (current_direction + 1) % 4
    else:
        new_direction = action  # Вверх или вниз
    
    # Устанавливаем движение в соответствии с направлением
    if new_direction == 0 and current_direction!=2:  # Влево
        snake_move_x, snake_move_y = -snake_speed, 0
    elif new_direction == 1 and current_direction!=3:  # Вверх
        snake_move_x, snake_move_y = 0, -snake_speed
    elif new_direction == 2 and current_direction!=0:  # Вправо
        snake_move_x, snake_move_y = snake_speed, 0
    elif new_direction == 3 and current_direction!=1:  # Вниз
        snake_move_x, snake_move_y = 0, snake_speed
    else:
        new_direction = current_direction
    current_direction = new_direction
    
    # Двигаем змейку
    snake_x_pos += snake_move_x
    snake_y_pos += snake_move_y
    new_head = (snake_x_pos, snake_y_pos)
    snake_body.insert(0, new_head)
    
    # Проверяем столкновения и получаем награду
    game_over = False
    if (snake_x_pos < FIELD_X or snake_x_pos >= FIELD_X + FIELD_WIDTH or
        snake_y_pos < FIELD_Y or snake_y_pos >= FIELD_Y + FIELD_HEIGHT or
        new_head in snake_body[1:]):
        game_over = True
    
    # Получаем следующее состояние и награду
    next_state = get_state(snake_body, (apple_x_pos, apple_y_pos), current_direction)
    reward = get_reward(snake_body, (apple_x_pos, apple_y_pos), game_over)
    
    # Обновляем Q-таблицу
    update_q_table(state, action, reward, next_state)
    
    if game_over:
        print("=== Игра окончена ===")
        print(f"Причина: {'Граница' if (snake_x_pos < FIELD_X or snake_x_pos >= FIELD_X+FIELD_WIDTH or snake_y_pos < FIELD_Y or snake_y_pos >= FIELD_Y+FIELD_HEIGHT) else 'Столкновение с телом'}")
        save_q_table()  # Сохраняем прогресс
        reset_game()  # Перезапускаем игру вместо выхода
        continue  # Переходим к следующей итерации цикла
    
    # Проверяем съедание яблока
    snake_rect_head = snake_surf.get_rect(center=new_head)
    if snake_rect_head.colliderect(apple_rect):
        score += 1
        if score % 2 == 0 and fps < 200:
            fps += 0.2
        apple_x_pos = apple_spawn(snake_body, True)
        apple_y_pos = apple_spawn(snake_body, False)
        apple_rect = apple_surf.get_rect(center=(apple_x_pos, apple_y_pos))
    else:
        snake_body.pop()
    
    # Отрисовка
    screen.blit(background, (0, 0))  # Рисуем готовый фон за 1 операцию
    screen.blit(apple_surf, apple_rect)
    
    for i, segment in enumerate(snake_body):
        if i == 0:
            screen.blit(snake_head_surf, snake_head_surf.get_rect(center=segment))  # Голова
        else:
            screen.blit(snake_surf, snake_surf.get_rect(center=segment))  # Остальное тело
    
    text_surf = font.render('Score: ' + str(score), False, 'White')
    screen.blit(text_surf, (FIELD_WIDTH - 200, 60))
    
    # Периодическая очистка памяти (каждые 5 секунд)
    if time.time() - last_cleanup > 5:
        pygame.time.delay(1)
        last_cleanup = time.time()

    pygame.display.flip()
    clock.tick(fps)