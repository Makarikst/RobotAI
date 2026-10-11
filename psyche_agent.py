# ============================================================
# ЖИВОЙ МИР — ML-ИНСТРУМЕНТ ДЛЯ ОБУЧЕНИЯ ИИ
# Полный код
# ============================================================
import os
import math
import json
import random
from collections import deque, defaultdict

import numpy as np

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

CELL = 16
MAP_COLS, MAP_ROWS = 100, 100
VIEW_COLS, VIEW_ROWS = 45, 50
PANEL_W = 440
WIDTH = VIEW_COLS * CELL + PANEL_W
HEIGHT = VIEW_ROWS * CELL
FPS = 30

UP, DOWN, LEFT, RIGHT, WAIT, INTERACT = range(6)
N_ACTIONS = 6

ISO_TILE_W = CELL * 2
ISO_TILE_H = CELL

speed_mult = 1
time_hours = 6.0
DAY_STEPS = 8000

SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
SAVE_JSON = os.path.join(SAVE_DIR, "save_progress.json")
SAVE_NPZ = os.path.join(SAVE_DIR, "save_brain.npz")

BLACK = (10, 10, 16)
WHITE = (240, 240, 245)
YELLOW = (255, 220, 60)
PANEL_BG = (22, 22, 30)
GREEN = (80, 200, 120)
RED = (220, 80, 80)
BLUE = (80, 160, 240)
CYAN = (120, 220, 240)
ORANGE = (240, 160, 60)
PURPLE = (180, 120, 240)
PINK = (240, 120, 180)
GRAY = (140, 140, 150)


def clip(v):
    return max(0, min(255, v))


def iso_flat(x, y):
    return (x - y) * (ISO_TILE_W // 2), (x + y) * (ISO_TILE_H // 2)


def wrap(text, font, width):
    words = text.split()
    lines, line = [], ""
    for w in words:
        t = (line + " " + w).strip()
        if font.size(t)[0] <= width:
            line = t
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines or [""]


def daylight(t):
    if 6 <= t <= 18:
        return 1.0
    if 18 < t <= 20:
        return 1.0 - (t - 18) / 2.0
    if 4 <= t < 6:
        return (t - 4) / 2.0
    return 0.0


def is_night(t):
    return t < 6.0 or t >= 22.0


def bfs_first_step(start, goal, walls, cols, rows):
    if start == goal:
        return None
    moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    visited = {start}
    q = deque([(start, None)])
    while q:
        pos, first = q.popleft()
        for d, (dx, dy) in enumerate(moves):
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt == goal:
                return d if first is None else first
            if nxt in walls or nxt in visited:
                continue
            if nxt[0] < 0 or nxt[1] < 0 or nxt[0] >= cols or nxt[1] >= rows:
                continue
            visited.add(nxt)
            q.append((nxt, d if first is None else first))
    return None


# ============================================================
# ТАЙЛЫ
# ============================================================
class TileRegistry:
    def __init__(self):
        self.tiles = {}
        self._gen()

    def _flat(self, color, noisy=False):
        surf = pygame.Surface((ISO_TILE_W, ISO_TILE_H), pygame.SRCALPHA)
        pts = [(ISO_TILE_W // 2, 0), (ISO_TILE_W, ISO_TILE_H // 2),
               (ISO_TILE_W // 2, ISO_TILE_H), (0, ISO_TILE_H // 2)]
        pygame.draw.polygon(surf, color, pts)
        if noisy:
            for _ in range(10):
                px = random.randint(ISO_TILE_W // 4, 3 * ISO_TILE_W // 4)
                py = random.randint(ISO_TILE_H // 4, 3 * ISO_TILE_H // 4)
                d = random.randint(10, 40)
                c = (max(0, color[0] - d), max(0, color[1] - d), max(0, color[2] - d))
                pygame.draw.circle(surf, c, (px, py), 1)
        pygame.draw.polygon(surf, (0, 0, 0, 40), pts, 1)
        return surf

    def _block(self, top, left, right, hh):
        surf = pygame.Surface((ISO_TILE_W, ISO_TILE_H + hh), pygame.SRCALPHA)
        w, h = ISO_TILE_W, ISO_TILE_H
        t1 = (w // 2, 0); t2 = (w, h // 2); t3 = (w // 2, h); t4 = (0, h // 2)
        b3 = (w // 2, h + hh); b4 = (0, h // 2 + hh); b2 = (w, h // 2 + hh)
        pygame.draw.polygon(surf, left, [t4, t3, b3, b4])
        pygame.draw.polygon(surf, right, [t3, t2, b2, b3])
        pygame.draw.polygon(surf, top, [t1, t2, t3, t4])
        pygame.draw.polygon(surf, (0, 0, 0, 60), [t1, t2, t3, t4], 1)
        return surf

    def _gen(self):
        self.tiles["floor_grass"] = self._flat((90, 170, 100), True)
        self.tiles["floor_road"] = self._flat((55, 55, 60))
        self.tiles["floor_home"] = self._flat((185, 120, 70), True)
        self.tiles["floor_shop"] = self._flat((220, 225, 230))
        self.tiles["floor_hospital"] = self._flat((230, 240, 245))
        self.tiles["floor_bath"] = self._flat((150, 200, 220))
        self.tiles["floor_park"] = self._flat((70, 160, 95), True)
        self.tiles["floor_police"] = self._flat((60, 70, 120))
        self.tiles["floor_school"] = self._flat((230, 220, 180))
        self.tiles["floor_bank"] = self._flat((200, 180, 120))
        self.tiles["floor_restaurant"] = self._flat((200, 140, 100))
        self.tiles["floor_factory"] = self._flat((80, 80, 85))
        self.tiles["floor_jail"] = self._flat((40, 40, 50))
        self.tiles["floor_street"] = self._flat((140, 140, 145), True)

        self.tiles["wall_home"] = self._block((240, 210, 170), (200, 170, 130), (170, 140, 100), 20)
        self.tiles["wall_shop"] = self._block((210, 90, 80), (170, 60, 50), (130, 40, 30), 20)
        self.tiles["wall_hospital"] = self._block((230, 235, 240), (200, 205, 210), (170, 175, 180), 22)
        self.tiles["wall_police"] = self._block((70, 80, 130), (50, 60, 110), (40, 50, 90), 20)
        self.tiles["wall_school"] = self._block((240, 230, 190), (210, 200, 160), (180, 170, 130), 20)
        self.tiles["wall_bank"] = self._block((220, 200, 140), (190, 170, 110), (160, 140, 80), 20)
        self.tiles["wall_restaurant"] = self._block((220, 160, 120), (190, 130, 90), (160, 100, 60), 20)
        self.tiles["wall_factory"] = self._block((90, 90, 95), (70, 70, 75), (50, 50, 55), 22)
        self.tiles["wall_jail"] = self._block((50, 50, 60), (30, 30, 40), (20, 20, 30), 26)
        self.tiles["wall_park"] = self._block((100, 80, 60), (80, 60, 40), (60, 40, 20), 14)
        self.tiles["wall_fence"] = self._block((130, 135, 145), (100, 105, 115), (80, 85, 95), 10)

        bed = pygame.Surface((ISO_TILE_W, ISO_TILE_H), pygame.SRCALPHA)
        pygame.draw.polygon(bed, (60, 110, 180), [(24, 6), (40, 14), (24, 22), (8, 14)])
        pygame.draw.polygon(bed, (240, 240, 240), [(24, 6), (32, 10), (24, 14), (16, 10)])
        self.tiles["prop_bed"] = bed

        counter = self._block((140, 90, 50), (110, 70, 40), (80, 50, 30), 10)
        pygame.draw.circle(counter, (230, 60, 60), (24, 8), 3)
        self.tiles["prop_counter"] = counter

        water_table = pygame.Surface((ISO_TILE_W, ISO_TILE_H + 14), pygame.SRCALPHA)
        pygame.draw.rect(water_table, (140, 90, 50), (14, 14, 20, 3))
        pygame.draw.rect(water_table, (120, 80, 40), (16, 17, 2, 5))
        pygame.draw.rect(water_table, (120, 80, 40), (30, 17, 2, 5))
        pygame.draw.rect(water_table, (80, 160, 230), (20, 8, 8, 8))
        pygame.draw.rect(water_table, (60, 140, 210), (20, 8, 8, 2))
        self.tiles["prop_water_table"] = water_table

        phone = pygame.Surface((ISO_TILE_W, ISO_TILE_H + 10), pygame.SRCALPHA)
        pygame.draw.rect(phone, (40, 40, 50), (20, 6, 8, 14), border_radius=2)
        pygame.draw.rect(phone, (100, 200, 255), (21, 8, 6, 10))
        self.tiles["prop_phone"] = phone

        laptop = pygame.Surface((ISO_TILE_W, ISO_TILE_H + 12), pygame.SRCALPHA)
        pygame.draw.rect(laptop, (50, 50, 60), (14, 10, 20, 10))
        pygame.draw.rect(laptop, (80, 80, 90), (14, 18, 20, 3))
        pygame.draw.rect(laptop, (100, 200, 255), (16, 12, 16, 6))
        self.tiles["prop_laptop"] = laptop

        jail_bars = pygame.Surface((ISO_TILE_W, ISO_TILE_H + 30), pygame.SRCALPHA)
        for i in range(6):
            x = 6 + i * 6
            pygame.draw.line(jail_bars, (180, 180, 190), (x, 6), (x, 24), 2)
        pygame.draw.line(jail_bars, (180, 180, 190), (4, 8), (44, 8), 2)
        pygame.draw.line(jail_bars, (180, 180, 190), (4, 22), (44, 22), 2)
        self.tiles["prop_jail_bars"] = jail_bars

        tree = pygame.Surface((ISO_TILE_W, ISO_TILE_H + 30), pygame.SRCALPHA)
        pygame.draw.rect(tree, (100, 70, 40), (22, 18, 4, 12))
        pygame.draw.circle(tree, (50, 130, 75), (24, 12), 12)
        pygame.draw.circle(tree, (40, 110, 60), (18, 10), 8)
        self.tiles["prop_tree"] = tree

        bush = pygame.Surface((ISO_TILE_W, ISO_TILE_H + 16), pygame.SRCALPHA)
        pygame.draw.circle(bush, (60, 140, 70), (18, 14), 6)
        pygame.draw.circle(bush, (70, 150, 80), (28, 14), 7)
        pygame.draw.circle(bush, (50, 120, 60), (23, 18), 5)
        self.tiles["prop_bush"] = bush

        flower = pygame.Surface((ISO_TILE_W, ISO_TILE_H + 10), pygame.SRCALPHA)
        pygame.draw.line(flower, (60, 130, 60), (24, 8), (24, 18), 1)
        pygame.draw.circle(flower, (230, 100, 150), (24, 8), 3)
        pygame.draw.circle(flower, (255, 220, 80), (24, 8), 1)
        self.tiles["prop_flower"] = flower

        rock = pygame.Surface((ISO_TILE_W, ISO_TILE_H + 8), pygame.SRCALPHA)
        pygame.draw.ellipse(rock, (120, 120, 125), (18, 10, 12, 8))
        pygame.draw.ellipse(rock, (90, 90, 95), (22, 12, 8, 5))
        self.tiles["prop_rock"] = rock

        bench = pygame.Surface((ISO_TILE_W, ISO_TILE_H + 12), pygame.SRCALPHA)
        pygame.draw.rect(bench, (120, 80, 40), (10, 12, 28, 4))
        pygame.draw.rect(bench, (120, 80, 40), (10, 6, 28, 3))
        pygame.draw.rect(bench, (80, 60, 40), (12, 16, 2, 6))
        pygame.draw.rect(bench, (80, 60, 40), (34, 16, 2, 6))
        self.tiles["prop_bench"] = bench

        trash = pygame.Surface((ISO_TILE_W, ISO_TILE_H + 14), pygame.SRCALPHA)
        pygame.draw.rect(trash, (80, 90, 80), (20, 8, 8, 12))
        pygame.draw.rect(trash, (60, 70, 60), (18, 6, 12, 3))
        self.tiles["prop_trash"] = trash

        car = pygame.Surface((ISO_TILE_W, ISO_TILE_H + 16), pygame.SRCALPHA)
        pygame.draw.polygon(car, (200, 60, 60), [(24, 2), (40, 10), (24, 18), (8, 10)])
        pygame.draw.polygon(car, (60, 60, 80), [(24, 5), (32, 9), (24, 13), (16, 9)])
        self.tiles["prop_car"] = car

        sign = pygame.Surface((ISO_TILE_W, ISO_TILE_H + 24), pygame.SRCALPHA)
        pygame.draw.rect(sign, (240, 240, 240), (21, 0, 6, 16))
        pygame.draw.rect(sign, (200, 40, 40), (23, 3, 3, 10))
        pygame.draw.rect(sign, (200, 40, 40), (19, 6, 10, 3))
        self.tiles["prop_hospital"] = sign

        shower = pygame.Surface((ISO_TILE_W, ISO_TILE_H + 24), pygame.SRCALPHA)
        pygame.draw.rect(shower, (180, 200, 220), (21, 3, 8, 12))
        pygame.draw.circle(shower, (200, 220, 240), (25, 5), 3)
        pygame.draw.line(shower, (150, 200, 230), (25, 8), (25, 16), 2)
        self.tiles["prop_shower"] = shower

        lamp = pygame.Surface((ISO_TILE_W, ISO_TILE_H + 40), pygame.SRCALPHA)
        pygame.draw.rect(lamp, (60, 60, 70), (23, 20, 3, 20))
        pygame.draw.circle(lamp, (255, 240, 180), (24, 18), 5)
        pygame.draw.circle(lamp, (255, 220, 120), (24, 18), 7, 1)
        self.tiles["prop_lamp"] = lamp

        colors = {
            "enemy": (220, 40, 40),
            "friend": (40, 120, 255),
            "citizen": (160, 160, 160),
            "doctor": (170, 80, 220),
            "seller": (170, 80, 220),
            "guard": (170, 80, 220),
            "teacher": (170, 80, 220),
            "helper": (170, 80, 220),
        }
        for role, color in colors.items():
            s = pygame.Surface((ISO_TILE_W, ISO_TILE_H), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (24, 14), 8)
            pygame.draw.circle(s, (30, 30, 30), (24, 14), 8, 2)
            pygame.draw.circle(s, WHITE, (21, 11), 2)
            self.tiles[f"npc_{role}"] = s


# ============================================================
# КВАЛИА
# ============================================================
class Qualia:
    def __init__(self):
        self.pain = 0.0
        self.pleasure = 0.0
        self.tension = 0.1
        self.resentment = 0.0
        self.fear = 0.0
        self.guilt = 0.0
        self.trauma = 0.0
        self.trust = 0.2
        self.loneliness = 0.0
        self.boredom = 0.0
        self.anxiety = 0.0
        self.color_assoc = defaultdict(float)
        self.zone_joy = defaultdict(float)
        self.thought = "Исследую мир..."

    def update(self, nociception, benefit, social, unpredictability,
               hunger, thirst, fatigue, health, hygiene, sickness,
               fear_event=False, color=None, zone_name=None, home_locked=False):

        if nociception > 0.3:
            self.anxiety = min(1.0, self.anxiety + 0.02)
        if social < -0.3:
            self.anxiety = min(1.0, self.anxiety + 0.03)

        self.anxiety = max(0.0, self.anxiety - 0.0005)

        internal_pain = (
            max(0.0, hunger - 0.4) * 0.6
            + max(0.0, thirst - 0.4) * 0.6
            + max(0.0, fatigue - 0.5) * 0.5
            + max(0.0, 0.6 - health) * 0.8
            + max(0.0, 0.4 - hygiene) * 0.4
            + sickness * 0.9
        )
        social_pain = max(0.0, -social * (0.7 + self.trauma * 0.5))
        self.pain = 0.82 * self.pain + 0.18 * (nociception + social_pain + internal_pain)

        social_pleasure = max(0.0, social * (0.6 + self.trust * 0.4))
        internal_pleasure = (
            max(0.0, 0.4 - self.tension) * 0.3
            + max(0.0, 0.4 - self.boredom) * 0.3
            + max(0.0, 0.4 - self.loneliness) * 0.2
            + max(0.0, 0.3 - self.fear) * 0.2
        )
        self.pleasure = 0.85 * self.pleasure + 0.15 * (
            benefit + social_pleasure + internal_pleasure)

        target_tension = unpredictability * 0.7 + self.pain * 0.5 + self.anxiety * 0.3
        if social < 0:
            target_tension += abs(social) * 0.5
        else:
            target_tension -= social * 0.3
        self.tension = 0.85 * self.tension + 0.15 * target_tension
        self.tension = max(0.05, min(1.0, self.tension))

        if social < -0.1:
            self.resentment = min(1.0, self.resentment + (self.pain * 0.3 + abs(social) * 0.5))
            self.trauma = min(1.0, self.trauma + 0.05)
            self.trust = max(0.0, self.trust - 0.04)
        else:
            self.resentment -= 0.02 + (self.pleasure * 0.4) * (1.0 + self.trust)
            if social > 0.1:
                self.trust = min(1.0, self.trust + 0.01)
                self.trauma = max(0.0, self.trauma - 0.005)
        self.resentment = max(0.0, min(1.0, self.resentment))

        target_fear = (self.pain * 0.6 + self.tension * 0.4
                       + self.trauma * 0.2 + self.anxiety * 0.5)
        if social > 0:
            target_fear -= social * 0.4
        if fear_event:
            target_fear = min(1.0, target_fear + 0.4)
        self.fear = 0.82 * self.fear + 0.18 * target_fear
        self.fear = max(0.0, min(1.0, self.fear))

        if zone_name == "home" and random.random() < 0.005:
            self.fear = min(1.0, self.fear + 0.05)
            self.tension = min(1.0, self.tension + 0.03)
        # Ночью темнота сама по себе тревожит
        if zone_name is not None and is_night(time_hours):
            # Не в доме, не в больнице, не в полиции
            if zone_name not in ("home", "hospital", "police", "jail"):
                self.anxiety = min(1.0, self.anxiety + 0.005)
        else:
            # Дома ночью: безопасно ТОЛЬКО если заперто
            # Дома: безопасно, если заперто — не важно, день или ночь
            if zone_name == "home":
                if home_locked:
                    self.anxiety = max(0.0, self.anxiety - 0.05)
                    self.fear = max(0.0, self.fear - 0.03)
                    self.tension = max(0.0, self.tension - 0.02)
                elif is_night(time_hours):
                    self.anxiety = min(1.0, self.anxiety + 0.008)
                    self.fear = min(1.0, self.fear + 0.005)

        conflict = 0.0
        if nociception > 0.2:
            conflict += 0.5
        if social < -0.3:
            conflict += 0.3
        self.guilt = 0.88 * self.guilt + 0.12 * conflict
        self.guilt = max(0.0, min(1.0, self.guilt))

        target_loneliness = max(0.0, 0.3 - social) * (1.0 - self.trust * 0.3)
        self.loneliness = 0.9 * self.loneliness + 0.1 * target_loneliness
        self.loneliness = max(0.0, min(1.0, self.loneliness))

        target_boredom = max(0.0, 0.3 - benefit) * 0.8
        self.boredom = 0.9 * self.boredom + 0.1 * target_boredom
        self.boredom = max(0.0, min(1.0, self.boredom))

        if color:
            affect = self.pleasure - self.pain
            self.color_assoc[color] = 0.7 * self.color_assoc[color] + 0.3 * affect
            self.color_assoc[color] = max(-1.0, min(1.0, self.color_assoc[color]))

        if zone_name:
            self.zone_joy[zone_name] = 0.95 * self.zone_joy[zone_name] + 0.05 * self.pleasure

        self._think(social, color, hunger, thirst, fatigue, health, hygiene, sickness)

    def _think(self, social, color, hunger, thirst, fatigue, health, hygiene, sickness):
        if sickness > 0.6:
            self.thought = "БОЛЕЗНЬ: Нужно в больницу!"
        elif health < 0.4:
            self.thought = "ТРЕВОГА: Здоровье падает."
        elif hygiene < 0.3:
            self.thought = "ГРЯЗЬ: Надо помыться."
        elif hunger > 0.7:
            self.thought = "ГОЛОД: Очень хочу есть."
        elif thirst > 0.7:
            self.thought = "ЖАЖДА: Пить хочу."
        elif fatigue > 0.75:
            self.thought = "УСТАЛОСТЬ: Надо поспать."
        elif self.fear > 0.6:
            self.thought = "СТРАХ: Мне страшно."
        elif self.anxiety > 0.5:
            self.thought = "ТРЕВОЖНОСТЬ: Что-то не так."
        elif self.loneliness > 0.6:
            self.thought = "ОДИНОЧЕСТВО: Хочу к другу."
        elif self.boredom > 0.6:
            self.thought = "СКУКА: Хочу поиграть."
        elif color and abs(self.color_assoc[color]) > 0.25:
            state = "хорошее" if self.color_assoc[color] > 0 else "опасное"
            self.thought = f"ЗОНА {color}: опыт — {state}."
        elif self.trauma > 0.55:
            self.thought = "ТРАВМА: Мир враждебен."
        elif self.pleasure > 0.4:
            self.thought = "ГОМЕОСТАЗ: Спокойно."


# ============================================================
# НЕЙРОСЕТЬ
# ============================================================
class NeuralNet:
    def __init__(self, n_in, n_hidden, n_out, lr=0.0005):
        lim1 = math.sqrt(6.0 / (n_in + n_hidden))
        lim2 = math.sqrt(6.0 / (n_hidden + n_out))
        self.W1 = np.random.uniform(-lim1, lim1, (n_in, n_hidden)).astype(np.float32)
        self.b1 = np.zeros(n_hidden, dtype=np.float32)
        self.W2 = np.random.uniform(-lim2, lim2, (n_hidden, n_out)).astype(np.float32)
        self.b2 = np.zeros(n_out, dtype=np.float32)

        self.mW1 = np.zeros_like(self.W1); self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1); self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2); self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2); self.vb2 = np.zeros_like(self.b2)
        self.t = 0
        self.b1r = 0.9; self.b2r = 0.999
        self.eps = 1e-8
        self.lr = lr

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def backward(self, grad_out):
        dW2 = self.a1.T @ grad_out
        db2 = grad_out.sum(axis=0)
        da1 = grad_out @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        dW1 = self.x.T @ dz1
        db1 = dz1.sum(axis=0)
        self._adam(dW1, db1, dW2, db2)

    def _adam(self, dW1, db1, dW2, db2):
        self.t += 1
        bc1 = 1 - self.b1r ** self.t
        bc2 = 1 - self.b2r ** self.t
        for (p, g, m, v) in (
            (self.W1, dW1, self.mW1, self.vW1),
            (self.b1, db1, self.mb1, self.vb1),
            (self.W2, dW2, self.mW2, self.vW2),
            (self.b2, db2, self.mb2, self.vb2),
        ):
            m *= self.b1r; m += (1 - self.b1r) * g
            v *= self.b2r; v += (1 - self.b2r) * (g * g)
            m_hat = m / bc1; v_hat = v / bc2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def copy_from(self, other):
        self.W1[...] = other.W1; self.b1[...] = other.b1
        self.W2[...] = other.W2; self.b2[...] = other.b2


# ============================================================
# DQN АГЕНТ
# ============================================================
class DQNAgent:
    def __init__(self):
        self.n_in = 47
        self.n_hidden = 128
        self.n_out = N_ACTIONS

        self.q_net = NeuralNet(self.n_in, self.n_hidden, self.n_out, lr=0.0005)
        self.target_net = NeuralNet(self.n_in, self.n_hidden, self.n_out)
        self.target_net.copy_from(self.q_net)

        self.memory = deque(maxlen=200000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9999
        self.update_target_every = 1000
        self.steps_since_target = 0

        self.loss_log = deque(maxlen=100)
        self.reward_log = deque(maxlen=200)
        self.total_steps = 0
        self.deaths = 0
        self.crimes = 0
        self.tasks_done = 0

        self.hunger = 0.1
        self.thirst = 0.1
        self.fatigue = 0.1
        self.health = 1.0
        self.hygiene = 1.0
        self.sickness = 0.0
        self.money = 20

        self.has_phone = True
        self.has_laptop = False
        self.food_at_home = 0
        self.in_hospital = 0
        self.in_jail = 0
        self.crime = False
        self.wanted = 0

        self.player_task = None
        self.player_task_steps = 0
        self.learned_goals = {}

        self.q = Qualia()
        self.known_npcs = {}
        self.known_zones = {}
        self.achievements = []
        self.words_learned = set()
        self.messages_to_player = []
        self.reflection_timer = 0

        self._cached_step = None
        self._cached_target = None
        self._cached_step_at = 0
        self._cached_expected_pos = None
        self._locked_target = None
        self._locked_until_step = 0
        self.goal_cooldown = {}

    def build_state(self, world):
        pos = world.agent_pos
        goals = [world.shop_pos, world.park_pos, world.home_pos,
                 world.hospital_pos, world.bath_pos,
                 world.police_pos, world.school_pos, world.bank_pos,
                 world.restaurant_pos, world.factory_pos]
        base = [
            pos[0] / MAP_COLS,
            pos[1] / MAP_ROWS,
            self.hunger,
            self.thirst,
            self.fatigue,
            self.health,
            self.hygiene,
            self.sickness,
            1.0 if world.has_item else 0.0,
            self.money / 100.0,
        ]
        for g in goals:
            dx = (g[0] - pos[0]) / MAP_COLS
            dy = (g[1] - pos[1]) / MAP_ROWS
            base.append(math.sqrt(dx*dx + dy*dy))
        base.extend([
            self.q.pain, self.q.pleasure, self.q.resentment, self.q.fear,
            self.q.guilt, self.q.trauma, self.q.trust, self.q.tension,
            self.q.loneliness, self.q.boredom, self.q.anxiety,
            len(self.known_npcs) / 20.0,
        ])
        on_road = 1.0 if world.get_zone_at(pos[0], pos[1]) == "road" else 0.0
        car_near = 0.0
        for car in world.cars:
            d = abs(car.pos[0] - pos[0]) + abs(car.pos[1] - pos[1])
            if d < 3:
                car_near = max(car_near, 1.0 - d / 3.0)
        enemy_near = 0.0
        friend_near = 0.0
        for npc in world.npcs + world.enemies:
            d = abs(npc.pos[0] - pos[0]) + abs(npc.pos[1] - pos[1])
            if d < 10:
                if npc.role == "enemy":
                    enemy_near = max(enemy_near, 1.0 - d / 10.0)
                elif npc.role == "friend":
                    friend_near = max(friend_near, 1.0 - d / 10.0)

        # Дополнительные сигналы
        hour = int(time_hours)
        morning = 1.0 if 6 <= hour < 9 else 0.0
        day = 1.0 if 9 <= hour < 18 else 0.0
        evening = 1.0 if 18 <= hour < 22 else 0.0
        night = 1.0 if is_night(time_hours) else 0.0

        has_player_goal = 1.0 if self.player_task is not None else 0.0

        base.extend([
            on_road, car_near,
            1.0 if self.has_laptop else 0.0,
            self.food_at_home / 10.0,
            enemy_near, friend_near,
            1.0 if (self.has_phone and self.q.fear > 0.4) else 0.0,
            1.0 if (self.has_phone and len(self.words_learned) > 0) else 0.0,
            1.0 if world.get_zone_at(*world.agent_pos) == "home" else 0.0,
            morning, day, evening, night,
            has_player_goal,
        ])
        while len(base) < self.n_in:
            base.append(0.0)
        return np.array(base[:self.n_in], dtype=np.float32)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)
        q = self.q_net.forward(state.reshape(1, -1))[0]
        return int(np.argmax(q))

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        s_b = np.stack([b[0] for b in batch])
        s2_b = np.stack([b[3] for b in batch])
        a_b = np.array([b[1] for b in batch])
        r_b = np.array([b[2] for b in batch], dtype=np.float32)
        d_b = np.array([b[4] for b in batch], dtype=np.float32)

        q_next = self.target_net.forward(s2_b)
        max_q_next = np.max(q_next, axis=1)
        target = r_b + (1 - d_b) * self.gamma * max_q_next

        q_pred = self.q_net.forward(s_b)
        grad = np.zeros_like(q_pred)
        for i in range(self.batch_size):
            grad[i, a_b[i]] = (q_pred[i, a_b[i]] - target[i]) / self.batch_size

        self.q_net.backward(grad)
        diff = q_pred[np.arange(self.batch_size), a_b] - target
        abs_diff = np.abs(diff)
        huber = np.where(abs_diff < 1.0, 0.5 * diff ** 2, abs_diff - 0.5)
        loss = float(np.mean(huber))
        self.loss_log.append(loss)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps_since_target += 1
        if self.steps_since_target >= self.update_target_every:
            self.steps_since_target = 0
            self.target_net.copy_from(self.q_net)
        return loss

    def update_needs(self):
        self.hunger = min(1.0, self.hunger + 0.00002)
        self.thirst = min(1.0, self.thirst + 0.000025)
        self.fatigue = min(1.0, self.fatigue + 0.000015)
        self.hygiene = max(0.0, self.hygiene - 0.00003)
        if self.hygiene < 0.2:
            self.sickness = min(1.0, self.sickness + 0.00002)

        drain = 0.0
        if self.hunger > 0.9: drain += 0.0003
        if self.thirst > 0.9: drain += 0.0004
        if self.fatigue > 0.95: drain += 0.0002
        if self.sickness > 0.6: drain += 0.0005
        self.health = max(0.0, self.health - drain)

        # Запоминание: что делать при разных состояниях
        if self.hunger > 0.7:
            self.learned_goals["hunger"] = "go_to_shop"
        if self.thirst > 0.7:
            self.learned_goals["thirst"] = "go_to_park"
        if self.fatigue > 0.7:
            self.learned_goals["fatigue"] = "go_to_home"
        if self.hygiene < 0.3:
            self.learned_goals["hygiene"] = "go_to_bath"
        if self.sickness > 0.5 or self.health < 0.4:
            self.learned_goals["sick"] = "go_to_hospital"

    def choose_goal(self, world):
        for k in list(self.goal_cooldown.keys()):
            self.goal_cooldown[k] -= 1
            if self.goal_cooldown[k] <= 0:
                del self.goal_cooldown[k]

        scores = {}
        night = is_night(time_hours)

        scores['go_to_shop'] = self.hunger * 3.0 if (self.hunger > 0.4 and not night) else 0
        scores['go_to_home'] = (self.fatigue * 3.0 + self.q.fear * 2.0) if (self.fatigue > 0.5 or self.q.fear > 0.5) else 0
        scores['go_to_hospital'] = (self.sickness * 4.0 + (1 - self.health) * 3.0) \
            if (self.sickness > 0.5 or self.health < 0.4) else 0
        scores['go_to_bath'] = (1 - self.hygiene) * 2.0 if self.hygiene < 0.5 else 0
        scores['go_to_park'] = 0.8
        scores['go_to_police'] = self.q.resentment * 1.2 if self.q.resentment > 0.4 else 0
        scores['go_to_school'] = 0.5 if (len(self.words_learned) < 20 and not night) else 0
        scores['go_to_bank'] = 0.0
        scores['go_to_restaurant'] = self.hunger * 2.0 if (self.money > 30 and self.hunger > 0.5 and not night) else 0
        scores['go_to_factory'] = 1.5 if (self.money < 15 and not night) else 0

        # Распорядок дня — предпочтение, не приказ
        hour = int(time_hours)
        if 6 <= hour < 9:
            scores['go_to_shop'] += 1.0
            scores['go_to_bath'] += 0.5
        elif 9 <= hour < 18:
            scores['go_to_factory'] += 1.0
            scores['go_to_school'] += 0.5
        elif 18 <= hour < 22:
            scores['go_to_restaurant'] += 1.0
            scores['go_to_park'] += 0.5
        else:
            scores['go_to_home'] += 2.0

        # Твоя цель — бонус, а не приказ
        if self.player_task is not None:
            scores[self.player_task] = scores.get(self.player_task, 0) + 5.0
            self.player_task_steps -= 1
            if self.player_task_steps <= 0:
                self.player_task = None

        # Если все слабые — идём по цели игрока
        if all(v <= 0.1 for v in scores.values()):
            if self.player_task is not None:
                return self.player_task
            return random.choice(['go_to_home', 'go_to_park',
                                  'go_to_home', 'go_to_park'])

        for k in list(scores.keys()):
            if k in self.goal_cooldown:
                scores[k] *= 0.2

        best = max(scores, key=scores.get)
        self.goal_cooldown[best] = 500
        return best

    def instinct_target(self, world):
        if self._locked_target is not None and self.total_steps < self._locked_until_step:
            return self._locked_target

        target = None
        if self.hygiene < 0.4:
            target = world.bath_pos
        elif self.health < 0.5 or self.sickness > 0.4:
            target = world.hospital_pos
        elif self.hunger > 0.6 and not is_night(time_hours):
            target = world.shop_pos
        elif self.thirst > 0.6 and not is_night(time_hours):
            target = world.park_pos
        elif self.fatigue > 0.7 or self.q.fear > 0.7:
            target = world.home_pos
        elif self.money < 5 and not is_night(time_hours):
            target = world.factory_pos

        if target is not None:
            self._locked_target = target
            self._locked_until_step = self.total_steps + 300
        else:
            self._locked_target = None
        return target

    def instinct_step(self, world):
        target = self.instinct_target(world)
        if target is None:
            return None
        x, y = world.agent_pos
        if (x, y) == tuple(target):
            return INTERACT
        step = bfs_first_step((x, y), tuple(target), world.walls, MAP_COLS, MAP_ROWS)
        if step is None:
            return self._greedy_step(x, y, target, world)
        return step

    def _greedy_step(self, x, y, target, world):
        dx = target[0] - x
        dy = target[1] - y
        candidates = []
        if abs(dx) > abs(dy):
            if dx != 0: candidates.append(RIGHT if dx > 0 else LEFT)
            if dy != 0: candidates.append(DOWN if dy > 0 else UP)
        else:
            if dy != 0: candidates.append(DOWN if dy > 0 else UP)
            if dx != 0: candidates.append(RIGHT if dx > 0 else LEFT)
        moves = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}
        for c in candidates:
            mx, my = moves[c]
            nx, ny = x + mx, y + my
            if (nx, ny) not in world.walls and 0 <= nx < MAP_COLS and 0 <= ny < MAP_ROWS:
                return c
        return random.randint(0, 3)

    def save(self, path_json, path_npz):
        data = {
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "money": self.money,
            "hunger": self.hunger,
            "thirst": self.thirst,
            "fatigue": self.fatigue,
            "health": self.health,
            "hygiene": self.hygiene,
            "sickness": self.sickness,
            "achievements": self.achievements,
            "words_learned": list(self.words_learned),
            "known_npcs": self.known_npcs,
            "known_zones": self.known_zones,
            "has_phone": self.has_phone,
            "has_laptop": self.has_laptop,
            "food_at_home": self.food_at_home,
            "deaths": self.deaths,
            "crimes": self.crimes,
            "learned_goals": self.learned_goals,
        }
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        np.savez(path_npz,
                 W1=self.q_net.W1, b1=self.q_net.b1,
                 W2=self.q_net.W2, b2=self.q_net.b2)

    def load(self, path_json, path_npz):
        if os.path.exists(path_json):
            with open(path_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.total_steps = data.get("total_steps", 0)
            self.epsilon = data.get("epsilon", 1.0)
            self.money = data.get("money", 20)
            self.hunger = data.get("hunger", 0.1)
            self.thirst = data.get("thirst", 0.1)
            self.fatigue = data.get("fatigue", 0.1)
            self.health = data.get("health", 1.0)
            self.hygiene = data.get("hygiene", 1.0)
            self.sickness = data.get("sickness", 0.0)
            self.achievements = data.get("achievements", [])
            self.words_learned = set(data.get("words_learned", []))
            self.known_npcs = data.get("known_npcs", {})
            self.known_zones = data.get("known_zones", {})
            self.has_phone = data.get("has_phone", True)
            self.has_laptop = data.get("has_laptop", False)
            self.food_at_home = data.get("food_at_home", 0)
            self.deaths = data.get("deaths", 0)
            self.crimes = data.get("crimes", 0)
            self.learned_goals = data.get("learned_goals", {})
        if os.path.exists(path_npz):
            d = np.load(path_npz)
            if (d["W1"].shape == self.q_net.W1.shape
                    and d["W2"].shape == self.q_net.W2.shape):
                self.q_net.W1[...] = d["W1"]; self.q_net.b1[...] = d["b1"]
                self.q_net.W2[...] = d["W2"]; self.q_net.b2[...] = d["b2"]
                self.target_net.copy_from(self.q_net)
            else:
                print("Старое сохранение не подходит по форме — веса сброшены.")


# ============================================================
# NPC-ИИ
# ============================================================
class NPCBrain:
    def __init__(self, role):
        self.role = role
        self.attitude = 0.0
        self.memory = {}
        self.interactions = 0

    def interact(self, agent, world):
        self.interactions += 1

        if agent.q.pain > 0.6:
            agent.q.pleasure = min(1.0, agent.q.pleasure + 0.05)
            return "NPC: 'Тебе плохо? Иди в больницу.'"
        if agent.q.fear > 0.6:
            agent.q.fear = max(0.0, agent.q.fear - 0.1)
            agent.q.trust = min(1.0, agent.q.trust + 0.05)
            return "NPC: 'Не бойся, здесь безопасно.'"
        if agent.hunger > 0.8:
            agent.hunger = max(0.0, agent.hunger - 0.3)
            agent.q.pleasure = min(1.0, agent.q.pleasure + 0.1)
            return "NPC: 'Ты голоден? Вот еда.'"

        if self.role == 'friend':
            agent.q.pleasure = min(1.0, agent.q.pleasure + 0.15)
            agent.q.trust = min(1.0, agent.q.trust + 0.1)
            agent.q.resentment = max(0.0, agent.q.resentment - 0.1)
            agent.q.loneliness = max(0.0, agent.q.loneliness - 0.3)
            agent.q.fear = max(0.0, agent.q.fear - 0.15)
            self.attitude += 0.1
            return "Друг: 'Привет! Рад тебя видеть!'"

        if self.role == 'enemy':
            robbed = []
            if agent.food_at_home > 0:
                stolen = min(agent.food_at_home, random.randint(1, 3))
                agent.food_at_home -= stolen
                robbed.append(f"{stolen} еды")
            if agent.money > 0:
                stolen_money = min(agent.money, random.randint(5, 15))
                agent.money -= stolen_money
                robbed.append(f"{stolen_money} монет")
            agent.q.resentment = min(1.0, agent.q.resentment + 0.2)
            agent.q.fear = min(1.0, agent.q.fear + 0.2)
            agent.q.trust = max(0.0, agent.q.trust - 0.05)
            self.attitude -= 0.1
            if robbed:
                return f"Враг ограбил: {', '.join(robbed)}!"
            return "Враг: 'Убирайся отсюда!'"

        if self.role == 'helper':
            agent.hunger = max(0.0, agent.hunger - 0.2)
            agent.thirst = max(0.0, agent.thirst - 0.2)
            agent.hygiene = min(1.0, agent.hygiene + 0.1)
            agent.q.pleasure = min(1.0, agent.q.pleasure + 0.1)
            agent.q.trust = min(1.0, agent.q.trust + 0.08)
            return "Помощник: 'Держи, тебе пригодится.'"

        if self.role == 'doctor':
            if agent.health < 0.8 or agent.sickness > 0.3:
                if agent.money >= 3:
                    agent.money -= 3
                    agent.health = min(1.0, agent.health + 0.4)
                    agent.sickness = max(0.0, agent.sickness - 0.4)
                    agent.q.pleasure = min(1.0, agent.q.pleasure + 0.15)
                    agent.q.trust = min(1.0, agent.q.trust + 0.05)
                    self.attitude += 0.1
                    return "Врач вылечил тебя. -3 монеты."
                else:
                    agent.health = min(1.0, agent.health + 0.2)
                    agent.sickness = max(0.0, agent.sickness - 0.2)
                    return "Врач полечил бесплатно."
            return "Врач: 'Ты здоров.'"

        elif self.role == 'seller':
            if agent.hunger > 0.4:
                if agent.money >= 2:
                    agent.money -= 2
                    agent.hunger = max(0.0, agent.hunger - 0.5)
                    agent.q.pleasure = min(1.0, agent.q.pleasure + 0.1)
                    self.attitude += 0.05
                    return "Продавец дал еду. -2 монеты."
                return "Продавец: 'Деньги вперёд.'"
            return "Продавец: 'Тебе не нужно.'"

        elif self.role == 'guard':
            if agent.q.resentment > 0.5 or agent.q.fear > 0.5:
                agent.q.resentment = max(0.0, agent.q.resentment - 0.25)
                agent.q.fear = max(0.0, agent.q.fear - 0.25)
                agent.q.trust = min(1.0, agent.q.trust + 0.1)
                self.attitude += 0.15
                return "Стражник успокоил тебя."
            return "Стражник: 'Всё спокойно.'"

        elif self.role == 'teacher':
            if len(agent.words_learned) < 20:
                word = random.choice(["дом", "еда", "друг", "опасность", "помощь",
                                      "ночь", "день", "дорога", "вода", "тепло",
                                      "страх", "игра", "сон", "еда"])
                agent.words_learned.add(word)
                agent.q.pleasure = min(1.0, agent.q.pleasure + 0.08)
                agent.q.trust = min(1.0, agent.q.trust + 0.03)
                self.attitude += 0.1
                return f"Учитель: выучил слово '{word}'."
            return "Учитель: 'Ты уже много знаешь.'"

        else:
            if agent.q.pleasure > 0.3:
                agent.q.pleasure = min(1.0, agent.q.pleasure + 0.05)
                self.attitude += 0.05
                return "Житель: 'Привет!'"
            elif agent.q.pain > 0.3:
                agent.q.resentment = min(1.0, agent.q.resentment + 0.03)
                self.attitude -= 0.05
                return "Житель: 'Ты плохо выглядишь.'"
            return "Житель: '...'"


class NPC:
    def __init__(self, pos, home, work, role='citizen', sick=False):
        self.pos = list(pos)
        self.home = home
        self.work = work
        self.role = role
        self.sick = sick
        self.goal = list(pos)
        self.brain = NPCBrain(role)

    def update(self, world, t):
        if 8 <= t < 18:
            self.goal = self.work
        else:
            self.goal = self.home
        dx = self.goal[0] - self.pos[0]
        dy = self.goal[1] - self.pos[1]
        if abs(dx) + abs(dy) < 1:
            return
        moves = []
        if dx != 0:
            moves.append((self.pos[0] + (1 if dx > 0 else -1), self.pos[1]))
        if dy != 0:
            moves.append((self.pos[0], self.pos[1] + (1 if dy > 0 else -1)))
        random.shuffle(moves)
        for mx, my in moves:
            if 0 <= mx < MAP_COLS and 0 <= my < MAP_ROWS and (mx, my) not in world.walls:
                self.pos = [mx, my]
                break


class Enemy(NPC):
    def __init__(self, pos, home, work):
        super().__init__(pos, home, work, role='enemy')
        self.stolen_items = []

    def update(self, world, agent_pos):
        if is_night(time_hours):
            dx = agent_pos[0] - self.pos[0]
            dy = agent_pos[1] - self.pos[1]
            d = abs(dx) + abs(dy)
            if d < 30:
                moves = []
                if dx != 0:
                    moves.append((self.pos[0] + (1 if dx > 0 else -1), self.pos[1]))
                if dy != 0:
                    moves.append((self.pos[0], self.pos[1] + (1 if dy > 0 else -1)))
                random.shuffle(moves)
                for mx, my in moves:
                    zone = world.get_zone_at(mx, my)
                    if zone == "home" and world.home_locked:
                        continue
                    if zone in ("shop", "hospital", "police", "school",
                                "bank", "restaurant", "factory", "jail"):
                        continue
                    if 0 <= mx < MAP_COLS and 0 <= my < MAP_ROWS and (mx, my) not in world.walls:
                        self.pos = [mx, my]
                        return
            moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            random.shuffle(moves)
            for dx, dy in moves:
                nx, ny = self.pos[0] + dx, self.pos[1] + dy
                zone = world.get_zone_at(nx, ny)
                if zone == "home" and world.home_locked:
                    continue
                if zone in ("shop", "hospital", "police", "school",
                            "bank", "restaurant", "factory", "jail"):
                    continue
                if 0 <= nx < MAP_COLS and 0 <= ny < MAP_ROWS and (nx, ny) not in world.walls:
                    self.pos = [nx, ny]
                    return
        else:
            super().update(world, time_hours)


class Car:
    def __init__(self, x, y, d):
        self.pos = [x, y]
        self.dir = d

    def update(self, world):
        nx = self.pos[0] + self.dir[0]
        ny = self.pos[1] + self.dir[1]
        if (nx, ny) in world.walls or nx < 0 or ny < 0 or nx >= MAP_COLS or ny >= MAP_ROWS:
            self.dir = (-self.dir[0], -self.dir[1])
            nx = self.pos[0] + self.dir[0]
            ny = self.pos[1] + self.dir[1]
        if (nx, ny) not in world.walls and 0 <= nx < MAP_COLS and 0 <= ny < MAP_ROWS:
            self.pos = [nx, ny]


# ============================================================
# МИР
# ============================================================
class BioWorld:
    def __init__(self):
        self.cols = MAP_COLS
        self.rows = MAP_ROWS

        self.home_pos = [8, 8]
        self.shop_pos = [90, 8]
        self.park_pos = [50, 90]
        self.hospital_pos = [8, 90]
        self.bath_pos = [50, 50]
        self.police_pos = [90, 50]
        self.school_pos = [20, 50]
        self.bank_pos = [70, 70]
        self.restaurant_pos = [30, 70]
        self.factory_pos = [70, 30]
        self.jail_pos = [88, 88]
        self.water_table_pos = [48, 90]

        self.npc_houses = [
            {"pos": [15, 20], "role": "citizen"},
            {"pos": [25, 20], "role": "citizen"},
            {"pos": [35, 20], "role": "citizen"},
            {"pos": [45, 20], "role": "citizen"},
            {"pos": [55, 20], "role": "friend"},
            {"pos": [65, 20], "role": "friend"},
            {"pos": [75, 20], "role": "enemy"},
        ]

        self.agent_pos = [8, 8]
        self.goal = self.shop_pos
        self.has_item = False

        self.walls = set()
        self.props = {}
        self.zone_map = {}
        self.lamps = []

        self._build_world()
        self.task = "go_to_shop"
        self.done = False
        self.task_message = "Задача: иди в Магазин."
        self.action_message = ""

        self.home_locked = False
        self.door_pos = (10, 14)

        self.cars = []
        self.npcs = []
        self.enemies = []
        self._spawn()

    def _build_house(self, x, y, size=4):
        for dx in range(size):
            for dy in range(size):
                self.zone_map[(x + dx, y + dy)] = "home"
        for dx in range(size):
            self.walls.add((x + dx, y))
            self.walls.add((x + dx, y + size - 1))
        for dy in range(size):
            self.walls.add((x, y + dy))
            self.walls.add((x + size - 1, y + dy))
        self.walls.discard((x + size // 2, y + size - 1))
        self.props[(x + 1, y + 1)] = "prop_bed"

    def _build_world(self):
        for x in range(self.cols):
            self.walls.add((x, 0))
            self.walls.add((x, self.rows - 1))
        for y in range(self.rows):
            self.walls.add((0, y))
            self.walls.add((self.cols - 1, y))

        for x in range(5, 95):
            self.zone_map[(x, 48)] = "road"
            self.zone_map[(x, 49)] = "road"
        for y in range(5, 95):
            self.zone_map[(48, y)] = "road"
            self.zone_map[(49, y)] = "road"

        for x in range(5, 15):
            for y in range(5, 15):
                self.zone_map[(x, y)] = "home"
        for x in range(5, 15):
            self.walls.add((x, 5)); self.walls.add((x, 14))
        for y in range(5, 15):
            self.walls.add((5, y)); self.walls.add((14, y))
        self.walls.discard((10, 14))

        for x in range(85, 95):
            for y in range(5, 15):
                self.zone_map[(x, y)] = "shop"
        for x in range(85, 95):
            self.walls.add((x, 5)); self.walls.add((x, 14))
        for y in range(5, 15):
            self.walls.add((85, y)); self.walls.add((94, y))
        self.walls.discard((90, 14))

        for x in range(5, 15):
            for y in range(85, 95):
                self.zone_map[(x, y)] = "hospital"
        for x in range(5, 15):
            self.walls.add((x, 85)); self.walls.add((x, 94))
        for y in range(85, 95):
            self.walls.add((5, y)); self.walls.add((14, y))
        self.walls.discard((10, 94))

        for x in range(46, 54):
            for y in range(46, 54):
                self.zone_map[(x, y)] = "bath"

        for x in range(35, 65):
            for y in range(75, 95):
                if (x, y) not in self.zone_map:
                    self.zone_map[(x, y)] = "park"
        for x in range(35, 65):
            self.walls.add((x, 75)); self.walls.add((x, 94))
        for y in range(75, 95):
            self.walls.add((35, y)); self.walls.add((64, y))
        self.walls.discard((50, 94))

        for x in range(85, 95):
            for y in range(45, 55):
                self.zone_map[(x, y)] = "police"
        for x in range(85, 95):
            self.walls.add((x, 45)); self.walls.add((x, 54))
        for y in range(45, 55):
            self.walls.add((85, y)); self.walls.add((94, y))
        self.walls.discard((90, 54))

        for x in range(82, 90):
            for y in range(82, 90):
                self.zone_map[(x, y)] = "jail"
        for x in range(82, 90):
            self.walls.add((x, 82)); self.walls.add((x, 89))
        for y in range(82, 90):
            self.walls.add((82, y)); self.walls.add((89, y))
        self.walls.discard((86, 89))

        for x in range(15, 25):
            for y in range(45, 55):
                self.zone_map[(x, y)] = "school"
        for x in range(15, 25):
            self.walls.add((x, 45)); self.walls.add((x, 54))
        for y in range(45, 55):
            self.walls.add((15, y)); self.walls.add((24, y))
        self.walls.discard((20, 54))

        for x in range(65, 75):
            for y in range(65, 75):
                self.zone_map[(x, y)] = "bank"
        for x in range(65, 75):
            self.walls.add((x, 65)); self.walls.add((x, 74))
        for y in range(65, 75):
            self.walls.add((65, y)); self.walls.add((74, y))
        self.walls.discard((70, 74))

        for x in range(25, 35):
            for y in range(65, 75):
                self.zone_map[(x, y)] = "restaurant"
        for x in range(25, 35):
            self.walls.add((x, 65)); self.walls.add((x, 74))
        for y in range(65, 75):
            self.walls.add((25, y)); self.walls.add((34, y))
        self.walls.discard((30, 74))

        for x in range(65, 75):
            for y in range(25, 35):
                self.zone_map[(x, y)] = "factory"
        for x in range(65, 75):
            self.walls.add((x, 25)); self.walls.add((x, 34))
        for y in range(25, 35):
            self.walls.add((65, y)); self.walls.add((74, y))
        self.walls.discard((70, 34))

        for house in self.npc_houses:
            hx, hy = house["pos"]
            self._build_house(hx, hy, size=4)

        self.props[tuple(self.home_pos)] = "prop_bed"
        self.props[tuple(self.shop_pos)] = "prop_counter"
        self.props[tuple(self.hospital_pos)] = "prop_hospital"
        self.props[tuple(self.bath_pos)] = "prop_shower"
        self.props[tuple(self.water_table_pos)] = "prop_water_table"
        self.props[(6, 8)] = "prop_laptop"
        self.props[(9, 8)] = "prop_phone"

        self.props[(86, 88)] = "prop_jail_bars"
        self.props[(87, 88)] = "prop_jail_bars"

        for _ in range(40):
            x = random.randint(37, 63)
            y = random.randint(77, 93)
            if (x, y) not in self.walls:
                self.props[(x, y)] = "prop_tree"
        for _ in range(25):
            x = random.randint(37, 63)
            y = random.randint(77, 93)
            if (x, y) not in self.props and (x, y) not in self.walls:
                self.props[(x, y)] = "prop_bush"
        for _ in range(20):
            x = random.randint(37, 63)
            y = random.randint(77, 93)
            if (x, y) not in self.props and (x, y) not in self.walls:
                self.props[(x, y)] = "prop_flower"

        for _ in range(600):
            x = random.randint(2, self.cols - 3)
            y = random.randint(2, self.rows - 3)
            if (x, y) in self.zone_map and self.zone_map[(x, y)] in (
                    "road", "home", "shop", "hospital", "bath", "park",
                    "police", "school", "bank", "restaurant", "factory", "jail"):
                continue
            if (x, y) in self.walls or (x, y) in self.props:
                continue
            r = random.random()
            if r < 0.3:
                self.props[(x, y)] = "prop_bush"
            elif r < 0.55:
                self.props[(x, y)] = "prop_flower"
            elif r < 0.7:
                self.props[(x, y)] = "prop_rock"
            elif r < 0.8:
                self.props[(x, y)] = "prop_tree"
            elif r < 0.9:
                self.props[(x, y)] = "prop_bench"
            else:
                self.props[(x, y)] = "prop_trash"

        for x in range(10, 90, 10):
            self.lamps.append((x, 46))
            self.lamps.append((x, 51))
        for y in range(10, 90, 10):
            self.lamps.append((46, y))
            self.lamps.append((51, y))

    def _spawn(self):
        for house in self.npc_houses:
            hx, hy = house["pos"]
            role = house["role"]
            pos = [hx + 2, hy + 3]
            home = list(pos)
            if role in ("citizen", "friend"):
                work = [random.randint(20, 80), random.randint(30, 80)]
            elif role == "enemy":
                work = [78, 23]
            else:
                work = [random.randint(20, 80), random.randint(20, 80)]
            sick = random.random() < 0.1
            if role == "enemy":
                self.enemies.append(Enemy(pos, home, work))
            else:
                self.npcs.append(NPC(pos, home, work, role=role, sick=sick))

        extra = [
            ("doctor", [10, 88]),
            ("seller", [88, 10]),
            ("guard", [88, 50]),
            ("teacher", [20, 50]),
            ("helper", [30, 30]),
            ("helper", [40, 40]),
        ]
        for role, work in extra:
            pos = [random.randint(20, 80), random.randint(20, 80)]
            home = [random.randint(20, 80), random.randint(20, 80)]
            self.npcs.append(NPC(pos, home, work, role=role))

        for _ in range(4):
            x = random.randint(15, 85)
            self.cars.append(Car(x, 48, (1, 0)))
        for _ in range(3):
            y = random.randint(15, 85)
            self.cars.append(Car(48, y, (0, 1)))

    def update_dynamic(self, agent, infect_check=False):
        for car in self.cars:
            car.update(self)
            if car.pos == self.agent_pos:
                agent.health = max(0.0, agent.health - 0.005)
                agent.q.thought = "МАШИНА! Меня сбило!"
                agent.q.fear = min(1.0, agent.q.fear + 0.1)
                agent.q.pain = min(1.0, agent.q.pain + 0.1)

        for npc in self.npcs:
            npc.update(self, time_hours)
            if infect_check:
                d = abs(npc.pos[0] - self.agent_pos[0]) + abs(npc.pos[1] - self.agent_pos[1])
                if d < 2 and npc.sick and random.random() < 0.0001:
                    agent.sickness = min(1.0, agent.sickness + 0.03)
            d = abs(npc.pos[0] - self.agent_pos[0]) + abs(npc.pos[1] - self.agent_pos[1])
            if d <= 1 and random.random() < 0.02:
                msg = npc.brain.interact(agent, self)
                if msg:
                    self.action_message = msg
                    agent.q.thought = msg

        for enemy in self.enemies:
            enemy.update(self, self.agent_pos)
            agent_zone = self.get_zone_at(*self.agent_pos)
            if agent_zone == "home" and self.home_locked:
                continue
            d = abs(enemy.pos[0] - self.agent_pos[0]) + abs(enemy.pos[1] - self.agent_pos[1])
            fear_radius = 15 if is_night(time_hours) else 8
            if d < fear_radius:
                agent.q.fear = min(1.0, agent.q.fear + 0.03)
                agent.q.tension = min(1.0, agent.q.tension + 0.02)
            if d < 4:
                agent.q.fear = min(1.0, agent.q.fear + 0.05)
            if d <= 1 and random.random() < 0.5:
                msg = enemy.brain.interact(agent, self)
                if msg:
                    self.action_message = msg
                    agent.q.thought = msg

        if agent.wanted > 0 and random.random() < 0.05:
            agent.in_jail = 800
            agent.wanted = 0
            agent.crime = False
            self.action_message = "ПОЛИЦИЯ АРЕСТОВАЛА!"
            agent.q.fear = min(1.0, agent.q.fear + 0.5)

    def get_zone_at(self, x, y):
        if (x, y) in self.zone_map:
            return self.zone_map[(x, y)]
        return "grass"

    def reset(self, task="go_to_shop"):
        if not task.startswith("go_to_"):
            task = "go_to_" + task
        self.has_item = False
        self.task = task
        self.done = False
        self.action_message = ""

        mapping = {
            "go_to_shop": (self.shop_pos, "Задача: иди в Магазин."),
            "go_to_park": (self.park_pos, "Задача: иди в Парк."),
            "go_to_hospital": (self.hospital_pos, "Задача: иди в Больницу."),
            "go_to_bath": (self.bath_pos, "Задача: иди помойся."),
            "go_to_police": (self.police_pos, "Задача: иди в Полицию."),
            "go_to_school": (self.school_pos, "Задача: иди в Школу."),
            "go_to_bank": (self.bank_pos, "Задача: иди в Банк."),
            "go_to_restaurant": (self.restaurant_pos, "Задача: иди в Ресторан."),
            "go_to_factory": (self.factory_pos, "Задача: иди на Фабрику."),
            "go_to_jail": (self.jail_pos, "Задача: иди в Тюрьму."),
        }
        if task in mapping:
            self.goal, self.task_message = mapping[task]
        else:
            self.goal = self.home_pos
            self.task_message = "Задача: иди Домой."

    def step(self, action, agent):
        if self.done:
            return 0.0, True

        x, y = self.agent_pos
        nx, ny = x, y

        if action == UP:
            ny -= 1
        elif action == DOWN:
            ny += 1
        elif action == LEFT:
            nx -= 1
        elif action == RIGHT:
            nx += 1
        elif action == WAIT:
            return -0.02, False
        elif action == INTERACT:
            return self._interact(agent, x, y)

        if (nx, ny) in self.walls:
            return -0.5, False

        old_dist = abs(x - self.goal[0]) + abs(y - self.goal[1])
        self.agent_pos = [nx, ny]
        new_dist = abs(nx - self.goal[0]) + abs(ny - self.goal[1])
        shaping = (old_dist - new_dist) * 0.1

        zone = self.get_zone_at(nx, ny)
        penalty = 0.0
        if agent.hygiene < 0.3 and zone not in ("bath", "park"):
            penalty += -0.05
        if agent.wanted > 0 and zone in ("shop", "school"):
            penalty += -0.1

        for npc in self.npcs:
            if npc.pos == self.agent_pos:
                agent.known_npcs[npc.role] = agent.known_npcs.get(npc.role, 0) + 1
                msg = npc.brain.interact(agent, self)
                self.action_message = msg
                return 0.5 + shaping + penalty, False

        for enemy in self.enemies:
            if enemy.pos == self.agent_pos:
                msg = enemy.brain.interact(agent, self)
                self.action_message = msg
                return -1.0 + shaping + penalty, False

        if [nx, ny] == self.goal:
            if self.task == "go_to_shop" and self.has_item:
                self.done = True
                return 0.5 + shaping + penalty, True
            elif self.task != "go_to_shop":
                self.done = True
                return 0.8 + shaping + penalty, True

        return -0.005 + shaping + penalty, False

    def _interact(self, agent, x, y):
        zone = self.get_zone_at(x, y)
        night = is_night(time_hours)
        msg = ""
        r = -0.05

        if zone == "shop":
            if night:
                if agent.hunger > 0.8:
                    agent.crime = True
                    agent.wanted = 1000
                    agent.crimes += 1
                    agent.q.fear = min(1.0, agent.q.fear + 0.3)
                    agent.q.guilt = min(1.0, agent.q.guilt + 0.3)
                    if agent.hunger > 0.5:
                        agent.hunger = max(0.0, agent.hunger - 0.5)
                        msg = "ОГРАБЛЕНИЕ: Взял еду ночью!"
                    else:
                        if agent.food_at_home < 10:
                            agent.food_at_home += 1
                        msg = "ОГРАБЛЕНИЕ: Взял еду с собой ночью!"
                    r = -2.0
                else:
                    msg = "Магазин закрыт."
                    r = -0.3
            else:
                if agent.money >= 5:
                    agent.money -= 5
                    if agent.hunger > 0.5 and random.random() < 0.5:
                        agent.hunger = max(0.0, agent.hunger - 0.5)
                        agent.q.pleasure = min(1.0, agent.q.pleasure + 0.15)
                        msg = "Купил и съел в магазине. -5 монет."
                        r = 1.0
                    else:
                        if agent.food_at_home < 10:
                            agent.food_at_home += 1
                        agent.q.pleasure = min(1.0, agent.q.pleasure + 0.05)
                        msg = "Купил еду с собой. -5 монет."
                        r = 0.6
                else:
                    agent.crime = True
                    agent.wanted = 1000
                    agent.crimes += 1
                    agent.q.fear = min(1.0, agent.q.fear + 0.3)
                    agent.q.guilt = min(1.0, agent.q.guilt + 0.3)
                    if agent.hunger > 0.5:
                        agent.hunger = max(0.0, agent.hunger - 0.4)
                        msg = "ОГРАБЛЕНИЕ: Съел без денег."
                    else:
                        if agent.food_at_home < 10:
                            agent.food_at_home += 1
                        msg = "ОГРАБЛЕНИЕ: Взял еду с собой."
                    r = -2.0

        elif zone == "home":
            if is_night(time_hours) and agent.q.fear > 0.5:
                self.home_locked = True
                agent.q.fear = max(0.0, agent.q.fear - 0.25)
                agent.q.tension = max(0.0, agent.q.tension - 0.15)
                msg = "Заперся дома. Враг не войдёт."
                r = 1.2
            elif agent.q.fear < 0.3 and self.home_locked:
                self.home_locked = False
                msg = "Открыл дверь дома."
                r = 0.2
            elif (agent.has_phone and agent.q.fear > 0.4
                    and agent.q.loneliness > 0.4 and agent.money >= 5):
                agent.money -= 5
                agent.q.fear = max(0.0, agent.q.fear - 0.4)
                agent.q.loneliness = max(0.0, agent.q.loneliness - 0.6)
                agent.q.trust = min(1.0, agent.q.trust + 0.1)
                agent.q.pleasure = min(1.0, agent.q.pleasure + 0.2)
                msg = "Позвонил другу, пригласил в гости. -5 монет."
                r = 1.5
            elif (agent.has_phone and agent.q.fear > 0.4
                    and agent.q.loneliness > 0.3 and len(agent.words_learned) > 0):
                words = list(agent.words_learned)
                text = " ".join(random.sample(words, min(3, len(words))))
                agent.messages_to_player.append(text)
                agent.q.loneliness = max(0.0, agent.q.loneliness - 0.3)
                agent.q.fear = max(0.0, agent.q.fear - 0.1)
                msg = f"Написал игроку: '{text}'"
                r = 0.8
            elif agent.food_at_home > 0 and agent.hunger > 0.4:
                agent.food_at_home -= 1
                agent.hunger = max(0.0, agent.hunger - 0.5)
                agent.q.pleasure = min(1.0, agent.q.pleasure + 0.1)
                msg = "Поел из запаса."
                r = 0.8
            elif agent.fatigue > 0.4:
                agent.fatigue = max(0.0, agent.fatigue - 0.5)
                agent.q.tension = max(0.0, agent.q.tension - 0.15)
                agent.q.anxiety = max(0.0, agent.q.anxiety - 0.2)
                agent.q.fear = max(0.0, agent.q.fear - 0.15)
                msg = "Поспал. Отдохнул."
                r = 0.9
            elif agent.has_phone and agent.q.boredom > 0.4:
                agent.q.boredom = max(0.0, agent.q.boredom - 0.3)
                agent.q.pleasure = min(1.0, agent.q.pleasure + 0.05)
                msg = "Поиграл на телефоне."
                r = 0.4
            elif agent.has_laptop and agent.q.fear > 0.5:
                agent.money += 3
                agent.fatigue = min(1.0, agent.fatigue + 0.05)
                msg = "Поработал по удалёнке. +3 монеты."
                r = 0.5
            else:
                msg = "Дома."
                r = 0.1

        elif zone == "hospital":
            if agent.health < 1.0 or agent.sickness > 0.0:
                if agent.money >= 3:
                    agent.money -= 3
                    agent.health = min(1.0, agent.health + 0.4)
                    agent.sickness = max(0.0, agent.sickness - 0.4)
                    agent.q.pleasure = min(1.0, agent.q.pleasure + 0.1)
                    msg = "Полечился. -3 монеты."
                    r = 1.0
                else:
                    agent.health = min(1.0, agent.health + 0.2)
                    agent.sickness = max(0.0, agent.sickness - 0.2)
                    msg = "Полечили бесплатно."
                    r = 0.3
            else:
                msg = "Ты здоров."
                r = 0.1

        elif zone == "bath":
            if agent.hygiene < 1.0:
                agent.hygiene = 1.0
                agent.q.pleasure = min(1.0, agent.q.pleasure + 0.2)
                msg = "Помылся."
                r = 2.0
            else:
                msg = "Уже чистый."
                r = 0.1

        elif zone == "park":
            agent.thirst = max(0.0, agent.thirst - 0.5)
            agent.q.tension = max(0.0, agent.q.tension - 0.1)
            agent.q.boredom = max(0.0, agent.q.boredom - 0.1)
            agent.q.loneliness = max(0.0, agent.q.loneliness - 0.03)
            msg = "Гуляю в парке."
            r = 0.0

        elif zone == "police":
            if agent.wanted > 0:
                agent.in_jail = 800
                agent.wanted = 0
                agent.crime = False
                msg = "АРЕСТ! Тебя посадили в тюрьму."
                r = -3.0
            else:
                agent.q.resentment = max(0.0, agent.q.resentment - 0.3)
                agent.q.fear = max(0.0, agent.q.fear - 0.3)
                msg = "Стражник успокоил."
                r = 0.6

        elif zone == "jail":
            msg = "Ты в тюрьме."
            r = -0.5

        elif zone == "school":
            if night:
                if len(agent.words_learned) < 20:
                    agent.crime = True
                    agent.wanted = 1000
                    agent.crimes += 1
                    agent.q.fear = min(1.0, agent.q.fear + 0.3)
                    agent.q.guilt = min(1.0, agent.q.guilt + 0.3)
                    word = random.choice(["мир", "свет", "тьма", "путь", "сила"])
                    agent.words_learned.add(word)
                    msg = f"ОГРАБЛЕНИЕ школы: выучил '{word}' ночью!"
                    r = -2.0
                else:
                    msg = "Школа закрыта."
                    r = -0.2
            else:
                if len(agent.words_learned) < 20:
                    if agent.money >= 3:
                        agent.money -= 3
                        word = random.choice(["мир", "свет", "тьма", "путь", "сила"])
                        agent.words_learned.add(word)
                        agent.q.pleasure = min(1.0, agent.q.pleasure + 0.08)
                        msg = f"Выучил слово: {word}. -3 монеты."
                        r = 0.7
                    else:
                        agent.crime = True
                        agent.wanted = 1000
                        agent.crimes += 1
                        agent.q.fear = min(1.0, agent.q.fear + 0.3)
                        agent.q.guilt = min(1.0, agent.q.guilt + 0.3)
                        word = random.choice(["мир", "свет", "тьма"])
                        agent.words_learned.add(word)
                        msg = f"ОГРАБЛЕНИЕ: выучил '{word}' бесплатно."
                        r = -2.0
                else:
                    msg = "Ты уже много знаешь."
                    r = 0.1

        elif zone == "bank":
            if agent.money > 10:
                agent.money -= 10
                agent.q.pleasure += 0.05
                msg = "Снял деньги."
                r = 0.4
            else:
                msg = "Нечего снимать."
                r = -0.1

        elif zone == "restaurant":
            if night:
                msg = "Ресторан закрыт."
                r = -0.2
            elif agent.money >= 15:
                agent.money -= 15
                agent.hunger = max(0.0, agent.hunger - 0.7)
                agent.q.pleasure += 0.2
                msg = "Пообедал в ресторане. -15 монет."
                r = 1.2
            else:
                msg = "Нет денег."
                r = -0.2

        elif zone == "factory":
            if night:
                msg = "Фабрика закрыта ночью."
                r = -0.2
            else:
                agent.money += 8
                agent.fatigue = min(1.0, agent.fatigue + 0.15)
                agent.q.pleasure = min(1.0, agent.q.pleasure + 0.05)
                msg = "Поработал. +8 монет."
                r = 1.0

        elif zone == "grass" and agent.has_phone:
            if agent.q.fear > 0.6 and agent.q.loneliness > 0.3:
                agent.q.fear = max(0.0, agent.q.fear - 0.3)
                agent.q.loneliness = max(0.0, agent.q.loneliness - 0.5)
                agent.q.trust = min(1.0, agent.q.trust + 0.05)
                msg = "Позвонил другу. Стало легче."
                r = 0.8
            else:
                msg = "Просто стоишь."
                r = -0.05

        else:
            msg = "Тут ничего нет."
            r = -0.05

        self.action_message = msg
        return r, False


# ============================================================
# РЕНДЕР
# ============================================================
def draw(screen, world, agent, registry, font, small, camera):
    screen.fill(BLACK)
    cam_x, cam_y = camera
    dl = daylight(time_hours)
    dark = 1.0 - dl

    for y in range(world.rows):
        for x in range(world.cols):
            fx, fy = iso_flat(x, y)
            sx = fx - cam_x
            sy = fy - cam_y
            if sx < -ISO_TILE_W or sx > WIDTH or sy < -ISO_TILE_H or sy > HEIGHT:
                continue
            zone = world.get_zone_at(x, y)

            if (x, y) in world.walls:
                if zone == "home": img = registry.tiles["wall_home"]; offs = 20
                elif zone == "shop": img = registry.tiles["wall_shop"]; offs = 20
                elif zone == "hospital": img = registry.tiles["wall_hospital"]; offs = 22
                elif zone == "police": img = registry.tiles["wall_police"]; offs = 20
                elif zone == "school": img = registry.tiles["wall_school"]; offs = 20
                elif zone == "bank": img = registry.tiles["wall_bank"]; offs = 20
                elif zone == "restaurant": img = registry.tiles["wall_restaurant"]; offs = 20
                elif zone == "factory": img = registry.tiles["wall_factory"]; offs = 22
                elif zone == "jail": img = registry.tiles["wall_jail"]; offs = 26
                elif zone == "park": img = registry.tiles["wall_park"]; offs = 14
                else: img = registry.tiles["wall_fence"]; offs = 10
                screen.blit(img, (sx, sy - offs))
                if dark > 0:
                    ov = pygame.Surface(img.get_size(), pygame.SRCALPHA)
                    ov.fill((0, 0, 30, int(200 * dark)))
                    screen.blit(ov, (sx, sy - offs))
            else:
                key = "floor_" + zone
                if key not in registry.tiles:
                    key = "floor_grass"
                img = registry.tiles[key]
                screen.blit(img, (sx, sy))
                if dark > 0:
                    ov = pygame.Surface(img.get_size(), pygame.SRCALPHA)
                    ov.fill((0, 0, 30, int(200 * dark)))
                    screen.blit(ov, (sx, sy))

    for (px, py), name in world.props.items():
        fx, fy = iso_flat(px, py)
        sx = fx - cam_x
        sy = fy - cam_y
        if sx < -ISO_TILE_W or sx > WIDTH or sy < -ISO_TILE_H or sy > HEIGHT:
            continue
        img = registry.tiles.get(name)
        if img is None:
            continue
        if name == "prop_tree": screen.blit(img, (sx, sy - 20))
        elif name == "prop_bush": screen.blit(img, (sx, sy - 8))
        elif name == "prop_flower": screen.blit(img, (sx, sy - 4))
        elif name == "prop_rock": screen.blit(img, (sx, sy - 4))
        elif name == "prop_bench": screen.blit(img, (sx, sy - 6))
        elif name == "prop_trash": screen.blit(img, (sx, sy - 6))
        elif name == "prop_water_table": screen.blit(img, (sx, sy - 14))
        elif name == "prop_phone": screen.blit(img, (sx, sy - 10))
        elif name == "prop_laptop": screen.blit(img, (sx, sy - 12))
        elif name == "prop_jail_bars": screen.blit(img, (sx, sy - 30))
        elif name == "prop_hospital": screen.blit(img, (sx, sy - 20))
        elif name == "prop_shower": screen.blit(img, (sx, sy - 20))
        elif name == "prop_bed": screen.blit(img, (sx, sy))
        else: screen.blit(img, (sx, sy - 10))

    if world.home_locked:
        dx, dy = world.door_pos
        fx, fy = iso_flat(dx, dy)
        sx = fx - cam_x
        sy = fy - cam_y
        pygame.draw.rect(screen, (120, 120, 130),
                         (sx, sy - 20, ISO_TILE_W, 20))

    for (lx, ly) in world.lamps:
        fx, fy = iso_flat(lx, ly)
        sx = fx - cam_x
        sy = fy - cam_y
        if sx < -ISO_TILE_W or sx > WIDTH or sy < -ISO_TILE_H or sy > HEIGHT:
            continue
        screen.blit(registry.tiles["prop_lamp"], (sx, sy - 40))
        if dark > 0.3:
            glow = pygame.Surface((80, 80), pygame.SRCALPHA)
            pygame.draw.circle(glow, (255, 220, 100, int(60 * dark)), (40, 40), 40)
            screen.blit(glow, (sx - 16, sy - 60))

    for npc in world.npcs:
        fx, fy = iso_flat(npc.pos[0], npc.pos[1])
        sx = fx - cam_x + ISO_TILE_W // 2
        sy = fy - cam_y + ISO_TILE_H // 2
        img = registry.tiles.get(f"npc_{npc.role}")
        if img:
            screen.blit(img, (sx - 24, sy - 20))
        if npc.sick:
            pygame.draw.circle(screen, (255, 0, 0), (sx, sy - 22), 3)

    for enemy in world.enemies:
        fx, fy = iso_flat(enemy.pos[0], enemy.pos[1])
        sx = fx - cam_x + ISO_TILE_W // 2
        sy = fy - cam_y + ISO_TILE_H // 2
        img = registry.tiles.get("npc_enemy")
        if img:
            screen.blit(img, (sx - 24, sy - 20))
        pygame.draw.circle(screen, (255, 0, 0), (sx, sy - 22), 3, 1)

    for car in world.cars:
        fx, fy = iso_flat(car.pos[0], car.pos[1])
        sx = fx - cam_x
        sy = fy - cam_y
        screen.blit(registry.tiles["prop_car"], (sx, sy - 8))

    gx, gy = world.goal
    gfx, gfy = iso_flat(gx, gy)
    pygame.draw.circle(screen, YELLOW, (gfx - cam_x + ISO_TILE_W // 2,
                                        gfy - cam_y + ISO_TILE_H // 2), 5)

    ax, ay = world.agent_pos
    afx, afy = iso_flat(ax, ay)
    bx = afx - cam_x + ISO_TILE_W // 2
    by = afy - cam_y + ISO_TILE_H // 2
    col = (
        clip(100 + agent.health * 155),
        clip(100 + agent.hygiene * 100),
        clip(100 + (1 - agent.sickness) * 100),
    )
    pygame.draw.ellipse(screen, (15, 15, 22), (bx - 8, by - 2, 16, 6))
    pygame.draw.circle(screen, col, (bx, by - 8), 8)
    pygame.draw.circle(screen, WHITE, (bx - 3, by - 11), 2)

    px0 = VIEW_COLS * CELL
    pygame.draw.rect(screen, PANEL_BG, (px0, 0, PANEL_W, HEIGHT))
    pygame.draw.line(screen, (55, 55, 65), (px0, 0), (px0, HEIGHT), 2)
    px = px0 + 12

    screen.blit(font.render("ML-АГЕНТ — DQN", True, WHITE), (px, 6))
    screen.blit(small.render(f"x{speed_mult}  ε={agent.epsilon:.3f}",
                             True, YELLOW), (px + 200, 8))

    hh = int(time_hours)
    mm = int((time_hours - hh) * 60)
    ph = "День" if not is_night(time_hours) else "Ночь"
    screen.blit(small.render(f"Время: {hh:02d}:{mm:02d} ({ph})", True, WHITE), (px, 24))

    def bar(y, name, v, c):
        screen.blit(small.render(name, True, (180, 180, 190)), (px, y))
        pygame.draw.rect(screen, (38, 38, 48), (px, y + 11, 220, 5), border_radius=2)
        pygame.draw.rect(screen, c, (px, y + 11, int(220 * max(0, min(1, v))), 5), border_radius=2)
        return y + 20

    y = 40
    screen.blit(small.render("— КВАЛИА —", True, PINK), (px, y)); y += 12
    y = bar(y, "Боль", agent.q.pain, RED)
    y = bar(y, "Удовольствие", agent.q.pleasure, GREEN)
    y = bar(y, "Страх", agent.q.fear, PURPLE)
    y = bar(y, "Тревожность", agent.q.anxiety, (200, 100, 100))
    y = bar(y, "Одиночество", agent.q.loneliness, BLUE)
    y = bar(y, "Скука", agent.q.boredom, CYAN)

    y += 2
    screen.blit(small.render("— ПОТРЕБНОСТИ —", True, PINK), (px, y)); y += 12
    y = bar(y, "Голод", agent.hunger, ORANGE)
    y = bar(y, "Жажда", agent.thirst, BLUE)
    y = bar(y, "Усталость", agent.fatigue, PURPLE)
    y = bar(y, "Здоровье", agent.health, GREEN)
    y = bar(y, "Гигиена", agent.hygiene, CYAN)
    y = bar(y, "Болезнь", agent.sickness, RED)

    y += 2
    if agent.loss_log:
        avg = sum(agent.loss_log) / len(agent.loss_log)
        screen.blit(small.render(f"Loss: {avg:.3f}", True, YELLOW), (px, y))
    else:
        screen.blit(small.render("Loss: —", True, YELLOW), (px, y))
    y += 12

    if agent.reward_log:
        avg_r = sum(agent.reward_log) / len(agent.reward_log)
        col_r = GREEN if avg_r > 0 else RED
        screen.blit(small.render(f"Средняя награда: {avg_r:+.3f}", True, col_r), (px, y))
    else:
        screen.blit(small.render("Средняя награда: —", True, WHITE), (px, y))
    y += 12

    screen.blit(small.render(f"Опыт: {len(agent.memory)}/200000", True, WHITE), (px, y)); y += 12
    screen.blit(small.render(f"Шагов: {agent.total_steps}  Смертей: {agent.deaths}", True, WHITE), (px, y)); y += 12
    screen.blit(small.render(f"Деньги: {agent.money}  Еда дома: {agent.food_at_home}", True, YELLOW), (px, y)); y += 12
    screen.blit(small.render(f"Слов: {len(agent.words_learned)}  Преступлений: {agent.crimes}", True, WHITE), (px, y)); y += 12

    if agent.player_task:
        screen.blit(small.render(f"ЦЕЛЬ ИГРОКА: {agent.player_task} ({agent.player_task_steps})",
                                 True, YELLOW), (px, y)); y += 12

    # Показываем выученные цели
    if agent.learned_goals:
        goals_str = " ".join(f"{k}→{v.replace('go_to_','')}"
                             for k, v in list(agent.learned_goals.items())[:4])
        screen.blit(small.render(f"Помню: {goals_str}", True, CYAN), (px, y)); y += 12

    status = []
    if agent.in_hospital > 0: status.append(f"БОЛЬНИЦА:{agent.in_hospital}")
    if agent.in_jail > 0: status.append(f"ТЮРЬМА:{agent.in_jail}")
    if agent.wanted > 0: status.append(f"РОЗЫСК:{agent.wanted}")
    if agent.crime: status.append("ПРЕСТУПНИК")
    if world.home_locked: status.append("ДОМ ЗАПЕРТ")
    if status:
        screen.blit(small.render(" | ".join(status), True, RED), (px, y)); y += 12
    else:
        y += 12

    if agent.messages_to_player:
        last = agent.messages_to_player[-1]
        screen.blit(small.render(f"Игроку: {last}", True, CYAN), (px, y)); y += 12

    screen.blit(small.render(f"Цель: {world.task}", True, CYAN), (px, y)); y += 12
    screen.blit(small.render("Мысль:", True, YELLOW), (px, y)); y += 11
    for line in wrap(agent.q.thought, small, PANEL_W - 24)[:3]:
        screen.blit(small.render(line, True, WHITE), (px, y)); y += 11

    pygame.display.flip()


# ============================================================
# ГЛАВНЫЙ ЦИКЛ
# ============================================================
def main():
    global speed_mult, time_hours

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("ML-агент — DQN")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("arial", 13)
    small = pygame.font.SysFont("arial", 10)

    registry = TileRegistry()
    world = BioWorld()
    agent = DQNAgent()
    agent.load(SAVE_JSON, SAVE_NPZ)
    world.reset(agent.choose_goal(world))

    running = True
    paused = False
    social = 0.0

    while running:
        social *= 0.99

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_1:
                    world.reset("go_to_shop")
                    agent.player_task = "go_to_shop"
                    agent.player_task_steps = 5000
                elif event.key == pygame.K_2:
                    world.reset("go_to_park")
                    agent.player_task = "go_to_park"
                    agent.player_task_steps = 5000
                elif event.key == pygame.K_3:
                    world.reset("go_to_home")
                    agent.player_task = "go_to_home"
                    agent.player_task_steps = 5000
                elif event.key == pygame.K_4:
                    world.reset("go_to_hospital")
                    agent.player_task = "go_to_hospital"
                    agent.player_task_steps = 5000
                elif event.key == pygame.K_5:
                    world.reset("go_to_bath")
                    agent.player_task = "go_to_bath"
                    agent.player_task_steps = 5000
                elif event.key == pygame.K_6:
                    world.reset("go_to_police")
                    agent.player_task = "go_to_police"
                    agent.player_task_steps = 5000
                elif event.key == pygame.K_7:
                    world.reset("go_to_school")
                    agent.player_task = "go_to_school"
                    agent.player_task_steps = 5000
                elif event.key == pygame.K_8:
                    world.reset("go_to_bank")
                    agent.player_task = "go_to_bank"
                    agent.player_task_steps = 5000
                elif event.key == pygame.K_9:
                    world.reset("go_to_restaurant")
                    agent.player_task = "go_to_restaurant"
                    agent.player_task_steps = 5000
                elif event.key == pygame.K_p:
                    social = 2.0
                    agent.q.pleasure = min(1.0, agent.q.pleasure + 0.3)
                    agent.q.loneliness = max(0.0, agent.q.loneliness - 0.2)
                    agent.q.tension = max(0.0, agent.q.tension - 0.2)
                elif event.key == pygame.K_i:
                    social = -2.0
                    agent.q.resentment = min(1.0, agent.q.resentment + 0.3)
                    agent.q.fear = min(1.0, agent.q.fear + 0.2)
                    agent.q.trust = max(0.0, agent.q.trust - 0.1)
                elif event.key == pygame.K_SPACE: paused = not paused
                elif event.key == pygame.K_s: agent.save(SAVE_JSON, SAVE_NPZ)
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    speed_mult = min(50, speed_mult + 1)
                elif event.key == pygame.K_MINUS:
                    speed_mult = max(1, speed_mult - 1)

        if not paused:
            for i in range(speed_mult):
                if agent.in_hospital > 0:
                    agent.in_hospital -= 1
                    if agent.in_hospital == 0:
                        agent.health = 1.0; agent.sickness = 0.0
                        agent.hygiene = 1.0; agent.hunger = 0.1
                        agent.thirst = 0.1; agent.fatigue = 0.1
                    continue

                if agent.in_jail > 0:
                    agent.in_jail -= 1
                    agent.hunger = min(1.0, agent.hunger + 0.0001)
                    agent.thirst = min(1.0, agent.thirst + 0.0001)
                    if agent.in_jail == 0:
                        agent.crime = False
                        agent.wanted = 0
                        world.agent_pos = list(world.police_pos)
                        world.reset("go_to_home")
                    continue

                agent.update_needs()

                old_hunger = agent.hunger; old_thirst = agent.thirst
                old_fatigue = agent.fatigue; old_health = agent.health
                old_sickness = agent.sickness

                s = agent.build_state(world)
                instinct_target = agent.instinct_target(world)

                if instinct_target is not None:
                    task_map = {
                        tuple(world.bath_pos): "go_to_bath",
                        tuple(world.hospital_pos): "go_to_hospital",
                        tuple(world.shop_pos): "go_to_shop",
                        tuple(world.park_pos): "go_to_park",
                        tuple(world.home_pos): "go_to_home",
                        tuple(world.factory_pos): "go_to_factory",
                    }
                    needed_task = task_map.get(tuple(instinct_target), None)
                    if needed_task and world.task != needed_task:
                        world.reset(needed_task)
                    if tuple(world.agent_pos) == tuple(world.goal):
                        a = INTERACT
                    else:
                        a = agent.instinct_step(world)
                        if a is None: a = agent.act(s)
                else:
                    if tuple(world.agent_pos) == tuple(world.goal):
                        a = INTERACT
                    else:
                        a = agent.act(s)

                if not hasattr(agent, '_stuck_counter'):
                    agent._stuck_counter = 0
                    agent._last_pos = tuple(world.agent_pos)
                if tuple(world.agent_pos) == agent._last_pos:
                    agent._stuck_counter += 1
                else:
                    agent._stuck_counter = 0
                    agent._last_pos = tuple(world.agent_pos)
                if agent._stuck_counter > 20:
                    a = random.randint(0, N_ACTIONS - 1)
                    agent._stuck_counter = 0
                    agent._locked_target = None
                    agent._locked_until_step = 0

                r, done = world.step(a, agent)

                need_change = (
                    (old_hunger - agent.hunger) * 5.0
                    + (old_thirst - agent.thirst) * 5.0
                    + (old_fatigue - agent.fatigue) * 3.0
                    + (agent.health - old_health) * 10.0
                    + (old_sickness - agent.sickness) * 5.0
                )
                subjective_r = r + need_change
                agent.reward_log.append(subjective_r)

                nociception = max(0.0, -subjective_r)
                benefit = max(0.0, subjective_r)
                zone = world.get_zone_at(*world.agent_pos)
                color = None
                if zone == "shop": color = "red"
                elif zone == "park": color = "green"
                elif zone == "home": color = "blue"
                elif zone == "hospital": color = "white"
                elif zone == "bath": color = "cyan"
                elif zone == "police": color = "dark_blue"
                elif zone == "jail": color = "black"
                elif zone == "school": color = "yellow"
                elif zone == "bank": color = "gold"
                elif zone == "restaurant": color = "orange"
                elif zone == "factory": color = "gray"

                fear_event = (zone == "jail")

                agent.q.update(
                    nociception=nociception * 0.1,
                    benefit=benefit * 0.1,
                    social=social,
                    unpredictability=0.0,
                    hunger=agent.hunger, thirst=agent.thirst,
                    fatigue=agent.fatigue, health=agent.health,
                    hygiene=agent.hygiene, sickness=agent.sickness,
                    fear_event=fear_event,
                    color=color,
                    zone_name=zone,
                    home_locked=world.home_locked,
                )

                world.update_dynamic(agent, infect_check=(i == 0))
                s2 = agent.build_state(world)

                if done:
                    subjective_r += 2.0
                    if agent.player_task is not None:
                        subjective_r += 5.0

                if agent.health <= 0.0:
                    agent.deaths += 1
                    agent.in_hospital = 3000
                    world.reset("go_to_hospital")

                agent.remember(s, a, subjective_r, s2, done)
                if agent.total_steps % 4 == 0:
                    agent.replay()
                agent.total_steps += 1

                agent.reflection_timer += 1
                if agent.reflection_timer >= 1000:
                    agent.reflection_timer = 0
                    agent.known_zones[zone] = agent.known_zones.get(zone, 0) + 1

                if agent.total_steps % 1000 == 0:
                    world.reset(agent.choose_goal(world))

                if agent.total_steps % 10000 == 0:
                    agent.save(SAVE_JSON, SAVE_NPZ)

                if done:
                    world.reset(agent.choose_goal(world))

            time_hours += 24.0 / DAY_STEPS
            if time_hours >= 24: time_hours -= 24

        ax, ay = world.agent_pos
        afx, afy = iso_flat(ax, ay)
        cam = (afx - (VIEW_COLS * CELL) // 2 + ISO_TILE_W // 2,
               afy - HEIGHT // 2 + ISO_TILE_H // 2)

        draw(screen, world, agent, registry, font, small, cam)
        clock.tick(FPS)

    agent.save(SAVE_JSON, SAVE_NPZ)
    pygame.quit()


if __name__ == "__main__":
    main()