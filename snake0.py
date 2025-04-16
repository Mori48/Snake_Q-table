import pygame
from sys import exit
import random
import numpy as np


def reset_game():
    global snake_x_pos, snake_y_pos, snake_move_x, snake_move_y, snake_body, \
        apple_x_pos, apple_y_pos, apple_rect, score, fps, current_direction
    
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
    
    # Новое яблоко
    apple_x_pos = apple_spawn(snake_body, True)
    apple_y_pos = apple_spawn(snake_body, False)
    apple_rect = apple_surf.get_rect(center=(apple_x_pos, apple_y_pos))
    
    # Сброс счета и скорости
    score = 0
    fps = 5


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
        return np.zeros((GRID_WIDTH * 2 + 1, GRID_HEIGHT * 2 + 1, 4, 4))

def apple_spawn(snake_body, index):
    if index:
        apple_x_pos = random.randint(0, GRID_WIDTH - 1) * CELL_SIZE + CELL_SIZE / 2
        while any(segment[0] == apple_x_pos for segment in snake_body):
            apple_x_pos = random.randint(0, GRID_WIDTH - 1) * CELL_SIZE + CELL_SIZE / 2
        return apple_x_pos
    else:
        apple_y_pos = random.randint(0, GRID_HEIGHT - 1) * CELL_SIZE + FIELD_Y + CELL_SIZE / 2
        while any(segment[1] == apple_y_pos for segment in snake_body):
            apple_y_pos = random.randint(0, GRID_HEIGHT - 1) * CELL_SIZE + FIELD_Y + CELL_SIZE / 2
        return apple_y_pos

pygame.init()

fps = 5
screen = pygame.display.set_mode((900, 1000))
pygame.display.set_caption('Smart snake')
clock = pygame.time.Clock()
font = pygame.font.Font(None, 30)  # Замените на ваш шрифт

# Параметры поля
FIELD_X, FIELD_Y = 0, 100
FIELD_WIDTH, FIELD_HEIGHT = 900, 900
CELL_SIZE = 50
GRID_WIDTH = FIELD_WIDTH // CELL_SIZE
GRID_HEIGHT = FIELD_HEIGHT // CELL_SIZE

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
alpha = 0.1
gamma = 0.9
epsilon = 0.2

# Дискретизация состояний
def discretize(value, cell_size):
    return int(round(value / cell_size))

def get_state(snake_head, apple_pos, direction):
    dx = discretize(apple_pos[0] - snake_head[0], CELL_SIZE)
    dy = discretize(apple_pos[1] - snake_head[1], CELL_SIZE)
    return (dx, dy, direction)

# Инициализация Q-таблицы с разумными размерами
Q = load_q_table()

def choose_action(state):
    if random.random() < epsilon:
        return random.randint(0, 3)  # 0=влево, 1=вверх, 2=вправо, 3=вниз
    else:
        return np.argmax(Q[state])

def update_q_table(state, action, reward, next_state):
    Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

def get_reward(snake_head, apple_pos, game_over):
    if game_over:
        return -100
    elif (snake_head[0] == apple_pos[0] and snake_head[1] == apple_pos[1]):
        return 100
    else:
        # Добавляем награду за приближение к яблоку
        distance = ((apple_pos[0] - snake_head[0])**2 + (apple_pos[1] - snake_head[1])**2)**0.5
        return -0.1 * distance / CELL_SIZE  # Нормализованный штраф за расстояние

score = 0
current_direction = 1  # 0=влево, 1=вверх, 2=вправо, 3=вниз

while True:
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
    state = get_state(snake_body[0], (apple_x_pos, apple_y_pos), current_direction)
    
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
        any(segment[0] == snake_x_pos and segment[1] == snake_y_pos for segment in snake_body[1:])):
        game_over = True
    
    # Получаем следующее состояние и награду
    next_state = get_state(new_head, (apple_x_pos, apple_y_pos), current_direction)
    reward = get_reward(new_head, (apple_x_pos, apple_y_pos), game_over)
    
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
        if score % 2 == 0:
            fps += 0.2
        apple_x_pos = apple_spawn(snake_body, True)
        apple_y_pos = apple_spawn(snake_body, False)
        apple_rect = apple_surf.get_rect(center=(apple_x_pos, apple_y_pos))
    else:
        snake_body.pop()
    
    # Отрисовка
    screen.fill('Black')
    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            color = 'chartreuse4' if (row + col) % 2 == 0 else 'chartreuse3'
            pygame.draw.rect(screen, color, (col * CELL_SIZE, FIELD_Y + row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    pygame.draw.rect(screen, 'white', (FIELD_X, FIELD_Y, FIELD_WIDTH, FIELD_HEIGHT), 5)
    screen.blit(apple_surf, apple_rect)
    
    for i, segment in enumerate(snake_body):
        if i == 0:
            screen.blit(snake_head_surf, snake_head_surf.get_rect(center=segment))  # Голова
        else:
            screen.blit(snake_surf, snake_surf.get_rect(center=segment))  # Остальное тело
    
    text_surf = font.render('Score: ' + str(score), False, 'White')
    screen.blit(text_surf, (FIELD_WIDTH - 200, 60))
    
    pygame.display.update()
    clock.tick(fps)