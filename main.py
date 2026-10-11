#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Робот-ИИ с Q-таблицей (Q-learning) + автогенератор уровней.
Портретный интерфейс с большими кнопками под телефон (Pydroid 3 / Termux / ПК).

Что робот знает:
  - ВИДИТ 4 соседние клетки: где стена или край поля;
  - помнит, какие соседние клетки уже посещал в этом заходе;
  - знает, в какую сторону финиш (знак dx, знак dy);
  - помнит, СКОЛЬКО РАЗ он уже был в текущей клетке (0-3) — чтобы не зацикливаться.
Чего он НЕ знает: что в стену идти нельзя, и как выглядит карта целиком.
Это он учит сам, только из награды:
  финиш +10, удар о стену -30 (попытка заканчивается), шаг -0.05, возврат -0.5.

Выбор хода — только по Q-таблице (softmax): чем выше Q, тем вероятнее ход.
«Температура» T со временем падает: сначала робот пробует всё подряд, потом
почти всегда идёт по лучшему. Никаких подсказок из карты и никаких ручных
правил в выборе хода нет — всё, что робот умеет, лежит в Q-таблице.

Новый уровень открывается только после WIN_STREAK проходов подряд без ударов.
Прогресс хранится в robot_progress.json рядом со скриптом.

Управление: тап по кнопкам. Тап по полю — поставить/убрать стену.
"""
import os
import sys
import json
import math
import random
from collections import deque

import pygame

# ----------------------------------------------------------------- настройки
ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # вверх, вправо, вниз, влево
ALPHA, GAMMA = 0.3, 0.95
T_START, T_MIN, T_TAU = 1.0, 0.08, 8000.0  # температура: T = T_MIN + (T_START-T_MIN)*exp(-шаги/TAU)
# CRASH_R должен быть хуже любого «бродить по кругу», иначе в тупике удар выгоднее петли
STEP_R, REVISIT_R, CRASH_R, GOAL_R = -0.05, -0.5, -30.0, 10.0
USE_VISITED = True  # видит ли робот, какие соседние клетки он уже посещал
SPEEDS = [4, 15, 60, 400, 3000]  # шагов в секунду
SPEED_NAMES = ["x1", "x2", "x3", "x4", "TURBO"]
HOLD_FRAMES = [40, 22, 10, 0, 0]  # пауза на итоге попытки (кадры), чтобы было видно результат
WIN_STREAK = 3  # чистых проходов (без ударов) подряд -> следующий уровень
AUTOSAVE_SEC = 20

BG = (24, 26, 36)
PANEL = (38, 42, 58)
GRID_BG = (52, 58, 78)
WALL = (110, 118, 150)
TRAIL = (70, 110, 150)
ROBOT = (255, 190, 60)
GOAL = (80, 220, 120)
TEXT = (235, 238, 248)
DIM = (150, 156, 180)
BTN = (64, 78, 128)
BTN_ON = (60, 150, 110)
BTN_RED = (160, 70, 80)
SENSE = (255, 120, 120)
BAD = (230, 90, 90)
WARN = (240, 170, 70)

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
SAVE_FILE = os.path.join(BASE_DIR, "robot_progress.json")


# ----------------------------------------------------------------- уровень
class Level:
    def __init__(self, number):
        self.number = number
        self.generate()

    def generate(self):
        n = min(6 + (self.number - 1) // 2, 14)
        density = min(0.15 + 0.012 * self.number, 0.32)
        self.n = n
        while True:
            self.walls = [[random.random() < density for _ in range(n)] for _ in range(n)]
            self.start = (random.randint(0, 1), random.randint(0, n - 1))
            self.goal = (random.randint(n - 2, n - 1), random.randint(0, n - 1))
            for (x, y) in (self.start, self.goal):
                self.walls[y][x] = False
            if self.solvable():
                break

    def is_wall(self, x, y):
        return x < 0 or y < 0 or x >= self.n or y >= self.n or self.walls[y][x]

    def solvable(self):
        seen = {self.start}
        q = deque([self.start])
        while q:
            x, y = q.popleft()
            if (x, y) == self.goal:
                return True
            for dx, dy in ACTIONS:
                nx, ny = x + dx, y + dy
                if not self.is_wall(nx, ny) and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    q.append((nx, ny))
        return False

    def toggle(self, x, y):
        if (x, y) in (self.start, self.goal):
            return False
        self.walls[y][x] = not self.walls[y][x]
        if not self.solvable():
            self.walls[y][x] = not self.walls[y][x]
            return False
        return True


# ----------------------------------------------------------------- ИИ + среда
def sign(v):
    return (v > 0) - (v < 0)


class Game:
    def __init__(self):
        self.Q = {}
        self.level_no = 1
        self.total_eps = 0
        self.total_steps = 0
        self.wins = 0
        self.hits = 0
        self.timeouts = 0
        self.recent = deque(maxlen=20)
        self.paused = False
        self.speed_i = 1
        self.auto_levels = True
        self.show_q = False
        self.acc = 0.0
        self.save_t = 0.0
        self.flash = ""
        self.flash_t = 0
        self.crash = None
        self.result = None  # "win" / "crash" / "timeout" — итог последней попытки
        self.hold = 0  # сколько кадров показываем итог
        self.pending = None  # что сделать после паузы: "reset" или "next"
        self.load()
        self.new_level()

    # --- температура исследования
    @property
    def temp(self):
        return T_MIN + (T_START - T_MIN) * math.exp(-self.total_steps / T_TAU)

    # --- сохранение в JSON
    def save(self):
        data = {
            "version": 4,
            "level": self.level_no,
            "episodes": self.total_eps,
            "steps": self.total_steps,
            "wins": self.wins,
            "hits": self.hits,
            "timeouts": self.timeouts,
            # ключ состояния "walls,vis,dx,dy,cnt" -> 4 значения Q
            "q": {",".join(map(str, k)): [round(v, 4) for v in vals]
                  for k, vals in self.Q.items()},
        }
        try:
            tmp = SAVE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, SAVE_FILE)  # запись целиком или никак
        except Exception as e:
            print("Не удалось сохранить:", e)

    def load(self):
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
        except FileNotFoundError:
            return
        except Exception as e:
            print("Файл прогресса повреждён, начинаю заново:", e)
            return
        try:
            # версия < 4 — несовместима (изменился формат ключа Q)
            if d.get("version", 0) < 4:
                print("Старое сохранение (v<4) — начинаю заново с новой Q-таблицей.")
                self.level_no = 1
                self.total_eps = self.wins = self.hits = self.timeouts = 0
                self.total_steps = 0
                self.Q = {}
                return

            self.level_no = int(d.get("level", 1))
            self.total_eps = int(d.get("episodes", 0))
            self.wins = int(d.get("wins", 0))
            self.hits = int(d.get("hits", 0))
            self.timeouts = int(d.get("timeouts", 0))
            self.total_steps = int(d.get("steps", 0))
            q = {}
            for k, v in d.get("q", {}).items():
                key = tuple(int(p) for p in k.split(","))
                if len(key) == 5 and len(v) == 4:   # 5 полей: walls, vis, dx, dy, cnt
                    q[key] = [float(x) for x in v]
            self.Q = q
        except Exception as e:
            print("Не удалось прочитать прогресс, начинаю заново:", e)
            self.Q = {}

    def reset_ai(self):
        self.Q = {}
        self.level_no = 1
        self.total_eps = self.total_steps = 0
        self.wins = self.hits = self.timeouts = 0
        self.recent.clear()
        self.new_level()
        self.save()
        self.say("ИИ сброшен")

    def say(self, text):
        self.flash, self.flash_t = text, 150

    # --- уровни
    def new_level(self, harder=False):
        if harder:
            self.level_no += 1
        self.level = Level(self.level_no)
        self.streak = 0
        self.ep_on_level = 0
        self.hold, self.pending, self.result, self.crash = 0, None, None, None
        self.reset_episode()

    def reset_episode(self):
        self.pos = self.level.start
        self.visited = {self.pos}
        self.visit_count = {self.pos: 1}   # сколько раз робот был в каждой клетке
        self.steps = 0
        self.max_steps = self.level.n * self.level.n * 3

    # --- зрение робота
    def state(self, pos, visited, level=None):
        lv = level or self.level
        x, y = pos
        gx, gy = lv.goal
        walls = vis = 0
        for i, (dx, dy) in enumerate(ACTIONS):
            nx, ny = x + dx, y + dy
            if lv.is_wall(nx, ny):
                walls |= 1 << i
            elif USE_VISITED and (nx, ny) in visited:
                vis |= 1 << i
        # Сколько раз робот уже был в текущей клетке (0..3) — важно против зацикливания.
        # Если state вызывается из UI для произвольной клетки, а не из self.pos —
        # используем self.visit_count.get(pos, 0), иначе 0.
        cnt = 0
        if hasattr(self, "visit_count"):
            cnt = min(self.visit_count.get(pos, 0), 3)
        return (walls, vis, sign(gx - x), sign(gy - y), cnt)

    def qv(self, s):
        v = self.Q.get(s)
        if v is None:
            v = self.Q[s] = [0.0] * 4
        return v

    @staticmethod
    def best(q):
        m = max(q)
        return random.choice([i for i in range(4) if q[i] == m])

    def choose(self, q):
        """Softmax по Q: выбор ТОЛЬКО по таблице, без подсказок из карты."""
        t = self.temp
        m = max(q)
        w = [math.exp((v - m) / t) for v in q]
        r = random.random() * sum(w)
        for i, wi in enumerate(w):
            r -= wi
            if r <= 0:
                return i
        return 3

    # --- один шаг обучения
    def step(self):
        lv = self.level
        s = self.state(self.pos, self.visited)
        q = self.qv(s)
        a = self.choose(q)
        dx, dy = ACTIONS[a]
        x, y = self.pos
        new = (x + dx, y + dy)
        self.total_steps += 1

        # удар о стену: робот сам этого не знал, получает штраф, попытка провалена
        if lv.is_wall(*new):
            self.crash = new
            q[a] += ALPHA * (CRASH_R - q[a])
            self.end_episode("crash")
            return

        r = STEP_R
        if new in self.visited:
            r += REVISIT_R
        gx, gy = lv.goal
        r += 0.1 * ((abs(gx - x) + abs(gy - y)) - (abs(gx - new[0]) + abs(gy - new[1])))
        done = new == lv.goal
        if done:
            r += GOAL_R

        self.pos = new
        self.visited.add(new)
        self.visit_count[new] = self.visit_count.get(new, 0) + 1
        self.steps += 1

        if done:
            target = r
        else:
            target = r + GAMMA * max(self.qv(self.state(new, self.visited)))
        q[a] += ALPHA * (target - q[a])

        if done:
            self.end_episode("win")
        elif self.steps >= self.max_steps:
            self.end_episode("timeout")

    def end_episode(self, result):
        self.total_eps += 1
        self.ep_on_level += 1
        self.result = result
        if result == "win":
            self.wins += 1
            self.streak += 1
            self.recent.append(1)
        else:
            self.streak = 0
            self.recent.append(0)
            if result == "crash":
                self.hits += 1
            else:
                self.timeouts += 1
        advance = self.auto_levels and self.streak >= WIN_STREAK
        self.pending = "next" if advance else "reset"
        self.hold = HOLD_FRAMES[self.speed_i]
        if self.hold == 0:
            self.apply_pending()

    def apply_pending(self):
        p, self.pending = self.pending, None
        if p == "next":
            self.say("Уровень пройден!")
            self.new_level(harder=True)
            self.save()
        elif p == "reset":
            self.reset_episode()

    def update(self, dt):
        if self.flash_t > 0:
            self.flash_t -= 1
        self.save_t += dt
        if self.save_t >= AUTOSAVE_SEC:
            self.save_t = 0
            self.save()
        if self.paused:
            return
        if self.hold > 0:  # показываем итог попытки (финиш/удар/время)
            self.hold -= 1
            if self.hold == 0:
                self.apply_pending()
            return
        self.acc += SPEEDS[self.speed_i] * dt
        n = 0
        while self.acc >= 1 and n < 4000:
            self.acc -= 1
            n += 1
            self.step()
            if self.hold > 0:
                self.acc = 0
                break
        if n >= 4000:
            self.acc = 0

    def success_rate(self):
        return 100 * sum(self.recent) / len(self.recent) if self.recent else 0


# ----------------------------------------------------------------- интерфейс
class UI:
    def __init__(self, game):
        self.g = game
        self.buttons = []
        self.grid_rect = pygame.Rect(0, 0, 10, 10)
        self.fonts = {}

    def font(self, size):
        size = max(10, int(size))
        if size not in self.fonts:
            self.fonts[size] = pygame.font.Font(None, size)
        return self.fonts[size]

    def text(self, surf, s, size, color, pos, center=False):
        img = self.font(size).render(s, True, color)
        r = img.get_rect()
        if center:
            r.center = pos
        else:
            r.topleft = pos
        surf.blit(img, r)

    def layout(self, W, H):
        m = max(8, int(W * 0.025))
        hud_h = int(H * 0.15)
        side = int(min(W - 2 * m, H * 0.46))
        self.grid_rect = pygame.Rect((W - side) // 2, hud_h + m // 2, side, side)
        top = self.grid_rect.bottom + m
        area_h = H - top - m
        rows, cols = 3, 2
        bw = (W - 2 * m - (cols - 1) * m) // cols
        bh = (area_h - (rows - 1) * m) // rows
        g = self.g
        defs = [
            (lambda: "Старт" if g.paused else "Пауза", self.a_pause,
             lambda: BTN_ON if g.paused else BTN),
            (lambda: "Скорость " + SPEED_NAMES[g.speed_i], self.a_speed, lambda: BTN),
            (lambda: "Новый уровень", self.a_new, lambda: BTN),
            (lambda: "Сброс ИИ", g.reset_ai, lambda: BTN_RED),
            (lambda: "Q-стрелки: " + ("вкл" if g.show_q else "выкл"), self.a_q,
             lambda: BTN_ON if g.show_q else BTN),
            (lambda: "Авто-уровни: " + ("вкл" if g.auto_levels else "выкл"), self.a_auto,
             lambda: BTN_ON if g.auto_levels else BTN),
        ]
        self.buttons = []
        for i, (label, act, col) in enumerate(defs):
            r, c = divmod(i, cols)
            rect = pygame.Rect(m + c * (bw + m), top + r * (bh + m), bw, bh)
            self.buttons.append((rect, label, act, col))
        self.m = m
        self.hud_h = hud_h

    # действия кнопок
    def a_pause(self):
        self.g.paused = not self.g.paused
        if self.g.paused:
            self.g.save()

    def a_speed(self):
        self.g.speed_i = (self.g.speed_i + 1) % len(SPEEDS)

    def a_new(self):
        self.g.new_level(harder=True)

    def a_q(self):
        self.g.show_q = not self.g.show_q

    def a_auto(self):
        self.g.auto_levels = not self.g.auto_levels

    def tap(self, pos):
        for rect, _, act, _ in self.buttons:
            if rect.collidepoint(pos):
                act()
                return
        if self.grid_rect.collidepoint(pos):
            lv = self.g.level
            cs = self.grid_rect.w / lv.n
            x = int((pos[0] - self.grid_rect.x) / cs)
            y = int((pos[1] - self.grid_rect.y) / cs)
            if 0 <= x < lv.n and 0 <= y < lv.n and lv.toggle(x, y):
                self.g.hold, self.g.pending, self.g.result = 0, None, None
                self.g.reset_episode()

    def draw(self, surf):
        W, H = surf.get_size()
        self.layout(W, H)
        g, lv = self.g, self.g.level
        surf.fill(BG)

        # --- HUD
        hud = pygame.Rect(self.m, self.m // 2, W - 2 * self.m, self.hud_h - self.m // 2)
        pygame.draw.rect(surf, PANEL, hud, border_radius=14)
        fs = H * 0.032
        x0, y0 = hud.x + 12, hud.y + 8
        self.text(surf, f"Уровень {g.level_no}   поле {lv.n}x{lv.n}", fs * 1.2, TEXT, (x0, y0))
        self.text(surf, f"Финиш {g.wins}  Удары {g.hits}  Время {g.timeouts}",
                  fs, DIM, (x0, y0 + fs * 1.3))
        self.text(surf, f"Случайность T={g.temp:.2f}   Чисто(20) {g.success_rate():.0f}%",
                  fs, DIM, (x0, y0 + fs * 2.4))
        self.text(surf, f"Подряд чисто {g.streak}/{WIN_STREAK}   Q: {len(g.Q)} сост.",
                  fs, DIM, (x0, y0 + fs * 3.5))

        # --- поле
        gr = self.grid_rect
        pygame.draw.rect(surf, GRID_BG, gr, border_radius=10)
        cs = gr.w / lv.n
        for y in range(lv.n):
            for x in range(lv.n):
                cell = pygame.Rect(gr.x + x * cs, gr.y + y * cs, cs, cs).inflate(-2, -2)
                if lv.walls[y][x]:
                    pygame.draw.rect(surf, WALL, cell, border_radius=5)
                elif (x, y) in g.visited:
                    pygame.draw.rect(surf, TRAIL, cell.inflate(-cs * 0.45, -cs * 0.45),
                                     border_radius=4)
                if g.show_q and not lv.walls[y][x] and (x, y) != lv.goal:
                    s = g.state((x, y), set())
                    q = g.Q.get(s)
                    if q and max(q) != min(q):
                        self.arrow(surf, cell.center, g.best(q), cs * 0.28, (255, 255, 255))
        # место удара (пока показываем итог попытки)
        if g.result == "crash" and g.hold > 0 and g.crash:
            cx, cy = g.crash
            if 0 <= cx < lv.n and 0 <= cy < lv.n:
                pygame.draw.rect(surf, BAD, pygame.Rect(gr.x + cx * cs, gr.y + cy * cs, cs, cs)
                                 .inflate(-2, -2), border_radius=5)
        # финиш
        gx, gy = lv.goal
        c = (gr.x + (gx + 0.5) * cs, gr.y + (gy + 0.5) * cs)
        pygame.draw.circle(surf, GOAL, c, cs * 0.36)
        self.text(surf, "F", cs * 0.6, BG, c, center=True)
        # старт
        sx, sy = lv.start
        pygame.draw.rect(surf, DIM, pygame.Rect(gr.x + sx * cs + 3, gr.y + sy * cs + 3,
                                                cs - 6, cs - 6), 2, border_radius=6)
        # робот
        rx, ry = g.pos
        c = (gr.x + (rx + 0.5) * cs, gr.y + (ry + 0.5) * cs)
        s = g.state(g.pos, g.visited)  # «зрение»: красные чёрточки у стен
        h = cs / 2 - 2
        for i, (dx, dy) in enumerate(ACTIONS):
            if (s[0] >> i) & 1:
                a = (c[0] + dx * h + dy * h * 0.6, c[1] + dy * h + dx * h * 0.6)
                b = (c[0] + dx * h - dy * h * 0.6, c[1] + dy * h - dx * h * 0.6)
                pygame.draw.line(surf, SENSE, a, b, max(2, int(cs * 0.08)))
        pygame.draw.circle(surf, ROBOT, c, cs * 0.30)
        for ex in (-0.1, 0.1):
            pygame.draw.circle(surf, BG, (c[0] + ex * cs, c[1] - 0.05 * cs), max(2, cs * 0.05))

        # итог попытки: рамка + крупная надпись, чтобы победу нельзя было принять за проигрыш
        if g.hold > 0 and g.result:
            label, col = {"win": ("ФИНИШ!", GOAL), "crash": ("УДАР О СТЕНУ", BAD),
                          "timeout": ("ВРЕМЯ ВЫШЛО", WARN)}[g.result]
            pygame.draw.rect(surf, col, gr, 5, border_radius=10)
            tw = self.font(H * 0.06).size(label)[0] + 30
            box = pygame.Rect(0, 0, tw, H * 0.08)
            box.center = (gr.centerx, gr.y + box.h)
            pygame.draw.rect(surf, BG, box, border_radius=12)
            pygame.draw.rect(surf, col, box, 3, border_radius=12)
            self.text(surf, label, H * 0.06, col, box.center, center=True)
        elif g.flash_t > 0:
            self.text(surf, g.flash, H * 0.05, GOAL, (W // 2, gr.centery), center=True)

        # --- кнопки
        for rect, label, act, col in self.buttons:
            pygame.draw.rect(surf, col(), rect, border_radius=16)
            size = min(rect.h * 0.34, rect.w * 0.12)
            self.text(surf, label(), size, TEXT, rect.center, center=True)

    def arrow(self, surf, center, a, size, color):
        dx, dy = ACTIONS[a]
        px, py = -dy, dx
        tip = (center[0] + dx * size, center[1] + dy * size)
        b1 = (center[0] - dx * size * 0.4 + px * size * 0.6,
              center[1] - dy * size * 0.4 + py * size * 0.6)
        b2 = (center[0] - dx * size * 0.4 - px * size * 0.6,
              center[1] - dy * size * 0.4 - py * size * 0.6)
        pygame.draw.polygon(surf, color, [tip, b1, b2])


# ----------------------------------------------------------------- main
def main():
    pygame.init()
    on_phone = "ANDROID_ARGUMENT" in os.environ or "ANDROID_ROOT" in os.environ
    if on_phone:
        screen = pygame.display.set_mode((0, 0))
    else:
        screen = pygame.display.set_mode((440, 880), pygame.RESIZABLE)
    pygame.display.set_caption("Q-робот")
    clock = pygame.time.Clock()
    game = Game()
    ui = UI(game)
    bg_events = tuple(getattr(pygame, n) for n in
                      ("APP_WILLENTERBACKGROUND", "APP_TERMINATING") if hasattr(pygame, n))

    running = True
    while running:
        dt = min(clock.tick(60) / 1000.0, 0.1)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type in bg_events:
                game.save()  # приложение уходит в фон — сохраняемся
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                ui.tap(e.pos)  # на телефоне тап приходит как мышь
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
        game.update(dt)
        ui.draw(screen)
        pygame.display.flip()

    game.save()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()