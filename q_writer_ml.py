#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q-писатель: робот учится писать по-русски и по-английски (Q-learning, как в лабиринте).

Как это устроено
  Робот пишет предложение по ОДНОЙ БУКВЕ. Состояние — то, что он «видит»:
    - внутри слова: последние 3 символа  (так он учит правописание);
    - в начале слова: предыдущее слово   (так он учит, какие слова идут друг за другом).
  Действие — следующий символ (буква, пробел, запятая, точка...).
  Награды выдаёт «учитель» — текст-корпус (corpus_ru.txt / corpus_en.txt):
    буквосочетание, которого нет в корпусе — «удар»: предложение обрывается, штраф
    слово реально существует ................ небольшой плюс
    пара слов встречалась в корпусе ......... плюс
    предложение закончено точкой ............ плюс, тем больше, чем больше удачных пар
  Робот НЕ знает заранее ни букв, ни слов, ни правил: всё учит из наград.

Честно о пределах: это крошечная модель. Она учится писать ПОХОЖЕ на свой корпус
(настоящие слова, правдоподобные пары слов), но без смысла. Чем больше текста в
corpus_*.txt (дописывай свои тексты, книги), тем лучше она пишет.

Прогресс (Q-таблицы обоих языков) хранится в writer_progress.json рядом со скриптом.
Управление: тап по кнопкам.
"""
import os
import re
import sys
import json
import math
import time
import random
import hashlib
from collections import deque

import pygame

# ----------------------------------------------------------------- настройки
LANGS = {
    "ru": {"name": "Русский", "short": "RU", "letters": "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"},
    "en": {"name": "English", "short": "EN", "letters": "abcdefghijklmnopqrstuvwxyz"},
}
PUNCT, END = ".,!?", ".!?"
ALPHA, GAMMA = 0.3, 0.9
T_START, T_MIN, T_TAU = 0.6, 0.2, 20000.0
TEACH_START, TEACH_MIN, TEACH_TAU = 0.6, 0.2, 30000.0   # доля предложений, которые робот «читает» у учителя
INIT_Q = -1.0                    # оценка ещё не пробованной буквы (пессимизм: неизвестное хуже опробованного)
PROMPT_P = 0.85                  # доля предложений, где ему дают затравку (1–2 первых слова)
MAXLEN = 90                      # самое длинное предложение (символов)
# награды
STEP_R, BAD_GRAM_R = -0.05, -3.0   # BAD_GRAM_R: «удар» (недопустимые буквы)
WORD_OK_R, WORD_BAD_R, PAIR_OK_R, REPEAT_R, EMPTY_R = 0.3, -2.0, 1.0, -1.0, -2.0
END_BASE_R, END_PAIR_R, END_PAIR_CAP, MIN_GEN = 1.0, 1.5, 8, 3   # MIN_GEN: слов, написанных самим роботом
END_BAD_R, TIMEOUT_R = -3.0, -3.0
SPEEDS = [8, 30, 120, 800, 6000]     # символов в секунду
SPEED_NAMES = ["x1", "x2", "x3", "x4", "TURBO"]
HOLD_FRAMES = [45, 25, 10, 0, 0]     # пауза на готовом предложении (кадры)
DEMO_T = 0.25                        # температура в «демо» (без обучения)
AUTOSAVE_SEC = 20
SAVE_VERSION = 2

BG = (24, 26, 36)
PANEL = (38, 42, 58)
TEXTBG = (30, 34, 48)
TEXT = (235, 238, 248)
DIM = (150, 156, 180)
OK = (110, 225, 140)
BAD = (240, 100, 100)
WARN = (240, 170, 70)
PAIR = (255, 190, 60)
BLUE = (120, 170, 255)
CHATC = (92, 110, 190)
BTN = (64, 78, 128)
BTN_ON = (60, 150, 110)
BTN_RED = (160, 70, 80)

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
SAVE_NAME = "writer_progress.json"

# ----------------------------------------------------------------- папка сохранения
def _writable(d):
    try:
        os.makedirs(d, exist_ok=True)
        probe = os.path.join(d, ".write_test")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except Exception:
        return False


def save_dirs():
    """Папки по порядку предпочтения; кэш/temp (их система чистит) — в самый конец."""
    import tempfile
    dirs = [BASE_DIR, os.path.join(os.path.expanduser("~"), ".robot_writer"), os.getcwd()]
    temp = os.path.abspath(tempfile.gettempdir())
    bad = lambda d: "/cache" in d.lower() or os.path.abspath(d).startswith(temp)
    return [d for d in dirs if not bad(d)] + [d for d in dirs if bad(d)]


def pick_dir(dirs=None):
    for d in (dirs or save_dirs()):
        if _writable(d):
            return d
    return "."


DATA_DIR = pick_dir()
SAVE_FILE = os.path.join(DATA_DIR, SAVE_NAME)

# ----------------------------------------------------------------- встроенные тексты (свои, без авторских прав)
BUILTIN = {
"ru": """
робот учится ходить по лабиринту. он видит стены и ищет выход. каждый день робот становится умнее.
утром солнце встаёт над городом. я люблю пить горячий чай с лимоном. кошка спит на тёплом диване.
мы идём гулять в большой парк. в парке растут высокие деревья. дети играют во дворе и смеются.
осенью идёт холодный дождь. зимой на улице лежит белый снег. весной птицы возвращаются домой.
летом мы едем к морю. море синее и очень тёплое. мама готовит вкусный суп на кухне.
папа читает интересную книгу. в книге написано про далёкие страны. я хочу узнать много нового.
учиться нужно каждый день. хорошая память помогает в учёбе. робот помнит свои ошибки и не повторяет их.
если он ошибся, он пробует снова. терпение и труд всё перетрут. друзья помогают друг другу в беде.
вечером мы собираемся за большим столом. на столе стоит ваза с цветами. цветы пахнут сладко и свежо.
за окном светит яркая луна. ночью город тихо засыпает. звёзды сияют на тёмном небе.
я пишу письмо старому другу. в письме я рассказываю о своей жизни. друг отвечает мне быстро.
мы часто говорим по телефону. телефон лежит на столе рядом с ключами. завтра будет новый день.
я встану рано и пойду на работу. работа у меня интересная и важная. коллеги всегда готовы помочь.
после работы я иду домой пешком. дома меня ждёт семья. мы вместе ужинаем и разговариваем.
потом я читаю детям сказку. дети засыпают с улыбкой. робот тоже хочет научиться писать.
он читает слова и запоминает буквы. сначала у него получается плохо. потом он пишет всё лучше и лучше.
язык это ключ к новым знаниям. кто знает много языков, тот видит больше. русский язык очень богатый и красивый.
английский язык тоже интересный. я учу новые слова каждый день. слова собираются в предложения.
предложения складываются в рассказ. рассказ может быть добрым или грустным. хороший рассказ всегда оставляет след.
я люблю гулять по вечернему городу. фонари горят тёплым светом. люди спешат домой после работы.
на углу играет музыкант. музыка звучит мягко и спокойно. я останавливаюсь и слушаю.
жизнь бывает трудной, но она прекрасна. нужно ценить каждый момент. маленький шаг каждый день ведёт к большой цели.
робот идёт к финишу и не сдаётся. он находит дорогу среди стен. когда путь найден, робот радуется.
новый уровень ждёт его впереди.
""",
"en": """
the robot learns to walk through the maze. it sees the walls and looks for the exit. every day the robot becomes smarter.
in the morning the sun rises over the city. i like to drink hot tea with lemon. the cat sleeps on the warm sofa.
we are going for a walk in the big park. tall trees grow in the park. children play in the yard and laugh.
in autumn the cold rain falls. in winter white snow lies on the street. in spring the birds come back home.
in summer we go to the sea. the sea is blue and very warm. mom cooks a tasty soup in the kitchen.
dad reads an interesting book. the book is about faraway countries. i want to learn many new things.
we should study every day. a good memory helps in learning. the robot remembers its mistakes and does not repeat them.
if it makes a mistake, it tries again. practice makes perfect. friends help each other in trouble.
in the evening we sit at the big table. there is a vase with flowers on the table. the flowers smell sweet and fresh.
a bright moon shines outside the window. at night the city falls asleep quietly. stars shine in the dark sky.
i write a letter to an old friend. in the letter i tell about my life. my friend answers me quickly.
we often talk on the phone. the phone lies on the table next to the keys. tomorrow will be a new day.
i will get up early and go to work. my work is interesting and important. my colleagues are always ready to help.
after work i walk home. my family is waiting for me at home. we have dinner together and talk.
then i read a story to the children. the children fall asleep with a smile. the robot also wants to learn to write.
it reads words and remembers letters. at first it does badly. then it writes better and better.
language is the key to new knowledge. who knows many languages sees more. the english language is rich and interesting.
the russian language is beautiful too. i learn new words every day. words come together in sentences.
sentences make a story. a story can be kind or sad. a good story always leaves a trace.
i like to walk through the evening city. the lamps shine with a warm light. people hurry home after work.
a musician plays on the corner. the music sounds soft and calm. i stop and listen.
life can be hard, but it is beautiful. we should value every moment. a small step every day leads to a big goal.
the robot goes to the finish and does not give up. it finds the way among the walls. when the path is found, the robot is happy.
a new level waits for it ahead.
""",
}


# ----------------------------------------------------------------- корпус (учитель)
def normalize(text, letters):
    text = text.lower().replace("\u0451", "\u0451")
    out = []
    for ch in text:
        out.append(ch if (ch in letters or ch in PUNCT) else " ")
    s = "".join(out)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r" ([.,!?])", r"\1", s)
    s = re.sub(r"([.,!?])\1+", r"\1", s)
    s = re.sub(r"([,])([.!?])", r"\2", s)
    return s.strip()


class Corpus:
    def __init__(self, lang, text):
        letters = LANGS[lang]["letters"]
        self.lang = lang
        s = normalize(text, letters)
        self.sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s*", s) if x.strip()]
        self.grams, self.vocab, self.bigrams = set(), set(), set()
        self.sent_words = []
        self.sent_texts = []
        wre = re.compile("[" + letters + "]+")
        for sent in self.sentences:
            padded = "^^^" + sent
            for i in range(3, len(padded)):
                self.grams.add(padded[i - 3:i + 1])
            prev = "^"
            sw = wre.findall(sent)
            if len(sw) >= 4:
                self.sent_words.append(sw)
                self.sent_texts.append(sent)
            for w in sw:
                self.vocab.add(w)
                self.bigrams.add((prev, w))
                prev = w
        seen = set(s)
        self.alphabet = [" "] + sorted(c for c in seen if c in letters) + [c for c in PUNCT if c in seen]
        self.size_words = sum(len(wre.findall(x)) for x in self.sentences)
        self.hash = hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

    @property
    def ok(self):
        return len(self.sentences) >= 5 and len(self.alphabet) >= 8


def load_corpus(lang):
    """Читает corpus_<lang>.txt из папки данных; если файла нет — создаёт из встроенного."""
    path = os.path.join(DATA_DIR, f"corpus_{lang}.txt")
    note = ""
    text = ""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(BUILTIN[lang].strip() + "\n")
            text = BUILTIN[lang]
            note = "создан"
    except Exception:
        text = BUILTIN[lang]
    c = Corpus(lang, text)
    if not c.ok:
        c = Corpus(lang, BUILTIN[lang])
        note = "файл слишком мал — взят встроенный"
    return c, path, note


# ----------------------------------------------------------------- агент-писатель
class Writer:
    """Агент для одного языка: своя Q-таблица, свои статистики."""

    def __init__(self, lang, corpus):
        self.lang = lang
        self.c = corpus
        self.A = corpus.alphabet
        self.Q = {}
        self.steps = 0           # буквы, написанные САМИМ роботом
        self.taught = 0          # буквы, прочитанные у учителя
        self.aidx = {ch: i for i, ch in enumerate(self.A)}
        self.sentences = 0
        self.hist_w = deque(maxlen=160)    # доля настоящих слов (скользящее среднее)
        self.hist_p = deque(maxlen=160)    # доля известных пар слов
        self.rec_w = deque(maxlen=40)
        self.rec_p = deque(maxlen=40)
        self.rec_e = deque(maxlen=40)      # 1 = предложение без «ударов»
        self.hist_e = deque(maxlen=160)
        self.log = deque(maxlen=14)        # готовые предложения: список (текст, статус)
        self.reset_sentence()

    @property
    def temp(self):
        return T_MIN + (T_START - T_MIN) * math.exp(-self.steps / T_TAU)

    def reset_sentence(self, free=False):
        self.text = ""
        self.cur = ""
        self.prev = "^"
        self.words = []
        self.tokens = []       # (строка, статус) для раскраски: ok / bad / p / prompt
        self.pairs_ok = 0
        self.aborted = False
        self.fresh = True
        self.n_prompt = 0
        # затравка: 1–2 первых слова случайного предложения корпуса; дальше робот пишет сам
        if not free and self.c.sent_words and random.random() < PROMPT_P:
            sw = random.choice(self.c.sent_words)
            k = 1 if random.random() < 0.6 else 2
            for w in sw[:k]:
                self.tokens.append((w, "prompt"))
                self.tokens.append((" ", "p"))
                self.words.append(w)
                self.text += w + " "
                self.prev = w
            self.n_prompt = k

    def state(self):
        if not self.cur:
            return "W:" + self.prev
        return "C:" + ("^^^" + self.text)[-3:]

    def qv(self, s):
        v = self.Q.get(s)
        if v is None:
            v = self.Q[s] = [INIT_Q] * len(self.A)
        return v

    def choose(self, q, t):
        m = max(q)
        w = [math.exp((v - m) / t) for v in q]
        r = random.random() * sum(w)
        for i, wi in enumerate(w):
            r -= wi
            if r <= 0:
                return i
        return len(q) - 1

    @property
    def p_teach(self):
        return TEACH_MIN + (TEACH_START - TEACH_MIN) * math.exp(-self.steps / TEACH_TAU)

    def step(self, learn=True, demo=False):
        """Одна буква, выбранная самим роботом. Возвращает True, если предложение закончено."""
        s = self.state()
        q = self.qv(s)
        a = self.choose(q, DEMO_T if demo else self.temp)
        return self.apply(a, s, q, learn, own=True)

    def teach_one(self):
        """«Чтение у учителя»: робот проходит по целому предложению корпуса и учится на
        его буквах (те же награды, та же формула Q-обучения — но ходы выбрал учитель)."""
        c = self.c
        if not c.sent_texts:
            return
        i = random.randrange(len(c.sent_texts))
        text, sw = c.sent_texts[i], c.sent_words[i]
        self.reset_sentence(free=True)
        k = 1 if random.random() < 0.6 else 2
        for w in sw[:k]:
            self.tokens += [(w, "prompt"), (" ", "p")]
            self.words.append(w)
            self.text += w + " "
            self.prev = w
        self.n_prompt = k
        for ch in text[len(self.text):]:
            a = self.aidx.get(ch)
            if a is None:
                break
            s = self.state()
            if self.apply(a, s, self.qv(s), True, own=False):
                break
        self.reset_sentence()

    def apply(self, a, s, q, learn, own):
        """Применить выбранную букву: награда, состояние, обновление Q."""
        self.fresh = False
        c = self.c
        ch = self.A[a]
        ctx = ("^^^" + self.text)[-3:]
        r = STEP_R
        done = False
        boundary = ch == " " or ch in PUNCT
        if ctx + ch not in c.grams:
            # «удар»: такого сочетания букв в корпусе нет -> предложение прервано
            r += BAD_GRAM_R
            bad = self.cur + (ch if not boundary else "")
            if bad:
                self.tokens.append((bad, "bad"))
            self.tokens.append(("\u00d7", "bad"))
            self.text += ch
            self.cur = ""
            self.aborted = True
            done = True
        else:
            if boundary:
                if self.cur:
                    w = self.cur
                    good = w in c.vocab
                    r += WORD_OK_R if good else WORD_BAD_R
                    if w in self.words:
                        r += REPEAT_R
                    elif (self.prev, w) in c.bigrams:
                        r += PAIR_OK_R
                        self.pairs_ok += 1
                    self.tokens.append((w, "ok" if good else "bad"))
                    self.words.append(w)
                    self.prev = w
                    self.cur = ""
                elif not (ch == " " and self.text.endswith(",")):
                    r += EMPTY_R
                self.tokens.append((ch, "p"))
            else:
                self.cur += ch
            self.text += ch
            if ch in END:
                last_ok = bool(self.words) and self.words[-1] in c.vocab
                if len(self.words) - self.n_prompt >= MIN_GEN and last_ok:
                    r += END_BASE_R + END_PAIR_R * min(self.pairs_ok, END_PAIR_CAP)
                else:
                    r += END_BAD_R
                done = True
            elif len(self.text) >= MAXLEN:
                if self.cur:
                    self.tokens.append((self.cur, "ok" if self.cur in c.vocab else "bad"))
                r += TIMEOUT_R
                done = True
        if learn:
            s2 = None if done else self.state()
            target = r if done else r + GAMMA * max(self.qv(s2))
            q[a] += ALPHA * (target - q[a])
            if own:
                self.steps += 1
            else:
                self.taught += 1
        return done

    def finish_sentence(self, count=True):
        """Подвести итог предложения: статистика + в журнал."""
        words = self.words[self.n_prompt:]   # только то, что написал сам робот
        if words:
            wq = sum(w in self.c.vocab for w in words) / len(words)
            pq = self.pairs_ok / len(words)
        else:
            wq = pq = 0.0
        if count:
            self.sentences += 1
            self.rec_w.append(wq)
            self.rec_p.append(pq)
            self.rec_e.append(0.0 if self.aborted else 1.0)
            self.hist_w.append(sum(self.rec_w) / len(self.rec_w))
            self.hist_p.append(sum(self.rec_p) / len(self.rec_p))
            self.hist_e.append(sum(self.rec_e) / len(self.rec_e))
        self.log.append(list(self.tokens))
        self.reset_sentence()

    # --- сохранение/загрузка одного языка
    def to_dict(self):
        return {
            "corpus": self.c.hash, "alphabet": "".join(self.A),
            "steps": self.steps, "taught": self.taught, "sentences": self.sentences,
            "hist_w": [round(x, 3) for x in self.hist_w], "hist_p": [round(x, 3) for x in self.hist_p],
            "hist_e": [round(x, 3) for x in self.hist_e],
            "q": {k: {str(i): round(v, 3) for i, v in enumerate(vals) if abs(v - INIT_Q) > 1e-9}
                  for k, vals in self.Q.items()},
        }

    def from_dict(self, d):
        """True, если данные подходят к текущему корпусу."""
        if d.get("alphabet") != "".join(self.A) or d.get("corpus") != self.c.hash:
            return False
        n = len(self.A)
        Q = {}
        for k, sp in d.get("q", {}).items():
            vals = [INIT_Q] * n
            for i, v in sp.items():
                i = int(i)
                if 0 <= i < n:
                    vals[i] = float(v)
            Q[k] = vals
        self.Q = Q
        self.steps = int(d.get("steps", 0))
        self.taught = int(d.get("taught", 0))
        self.sentences = int(d.get("sentences", 0))
        self.hist_w = deque((float(x) for x in d.get("hist_w", [])), maxlen=160)
        self.hist_p = deque((float(x) for x in d.get("hist_p", [])), maxlen=160)
        self.hist_e = deque((float(x) for x in d.get("hist_e", [])), maxlen=160)
        return True

    def reset(self):
        self.Q, self.steps, self.taught, self.sentences = {}, 0, 0, 0
        for dq in (self.hist_w, self.hist_p, self.hist_e, self.rec_w, self.rec_p, self.rec_e, self.log):
            dq.clear()
        self.reset_sentence()


# ----------------------------------------------------------------- диалоговый корпус (учитель «понимания»)
# Формат файла dialog_<язык>.txt (можно дописывать свои темы):
#   [id Название]            начало темы (смысла). [id after другой_id Название] — тема только в ответ на другую
#   > фраза | фраза | ...    как пользователь это говорит (учебные примеры)
#   ? фраза | фраза | ...    новые формулировки для «экзамена» (робот на них НЕ учится)
#   < ответ | ответ | ...    что робот может ответить ({name} — имя собеседника, {time}, {date})
BUILTIN_DIALOG = {
"ru": """
[greeting] Приветствие
> привет | здравствуй | добрый день | хай | приветик | приветствую
? здравствуйте | добрый вечер | доброе утро
< Привет! Рад тебя видеть. | Здравствуй! Как твои дела?

[how_are_you] Как дела
> как дела | как ты | как жизнь | как настроение | что нового
? как поживаешь | как твои дела | как самочувствие
< У меня всё хорошо, спасибо! А у тебя? | Отлично! Я учусь каждый день. А как ты?

[good_after after how_are_you] Тоже хорошо
> хорошо | отлично | нормально | прекрасно | супер
? замечательно | неплохо
< Рад это слышать! | Здорово, это очень приятно.

[bad_after after how_are_you] Плохо
> плохо | так себе | не очень | ужасно | грустно
? паршиво | неважно
< Мне жаль это слышать. Что случилось? | Не грусти, я рядом. Расскажешь, что произошло?

[and_you after how_are_you,good_after,bad_after] А у тебя?
> а у тебя | а ты | а у тебя как | а сам как | а ты как
? а у вас | а твои как
< У меня всё отлично, я учусь писать. | Хорошо! Каждый день становлюсь умнее.

[name_ask] Как тебя зовут
> как тебя зовут | как твоё имя | кто ты | представься | твоё имя
? как к тебе обращаться | назови своё имя
< Меня зовут Робот-писатель. | Я Q-писатель, робот, который учится. А как зовут тебя?

[my_name] Представился
> меня зовут | моё имя | зови меня | называй меня | можешь звать меня
? моё имя анна | зови меня сашей
< Приятно познакомиться, {name}! | Рад знакомству, {name}! | Приятно познакомиться!

[ask_my_name] Помнит имя
> как меня зовут | ты знаешь моё имя | помнишь как меня зовут | скажи моё имя
? ты помнишь моё имя | какое у меня имя
< Тебя зовут {name}. | Ты сказал, что тебя зовут {name}. | Ты пока не сказал мне своё имя.

[what_can] Что умеешь
> что ты умеешь | что ты можешь | чем ты занимаешься | зачем ты нужен | расскажи о себе
? на что ты способен | чему ты научился
< Я учусь писать и вести диалог. Попробуй спросить меня о чём-нибудь. | Я умею отвечать на простые вопросы и учусь каждый день.

[creator] Создатель
> кто тебя создал | кто тебя сделал | кто твой создатель | кто тебя написал
? кто тебя придумал | откуда ты взялся
< Меня написали на питоне, а учусь я сам с помощью наград. | Меня создал человек, а всему остальному я учусь сам.

[age] Возраст
> сколько тебе лет | какой у тебя возраст | когда ты родился
? ты старый | а тебе сколько лет
< Я совсем молодой, я только учусь. | Мне ещё нет и года, я новичок.

[weather] Погода
> какая сегодня погода | что с погодой | будет дождь | на улице тепло | какая погода
? как погода на улице | холодно сегодня
< Окна у меня нет, поэтому погоду не вижу. Надеюсь, там тепло! | Погоду я не знаю, посмотри за окно.

[time] Время
> который час | сколько времени | сколько сейчас времени | скажи время | время
? сколько сейчас часов | который сейчас час
< Сейчас {time}. | По моим часам {time}.

[date] Дата
> какое сегодня число | какой сегодня день | какая сегодня дата | что за число
? какое число | скажи дату
< Сегодня {date}. | По моим данным сегодня {date}.

[thanks] Спасибо
> спасибо | благодарю | большое спасибо | мерси | спасибо тебе
? огромное спасибо | благодарю тебя
< Пожалуйста! | Всегда рад помочь.

[bye] Прощание
> пока | до свидания | до встречи | увидимся | всего доброго
? прощай | до скорого
< До встречи! Заходи ещё. | Пока! Было приятно поговорить.

[praise] Похвала
> ты молодец | ты умный | ты мне нравишься | ты классный | отлично пишешь
? ты умница | ты крутой
< Спасибо, мне очень приятно! | Стараюсь! Ты тоже молодец.

[rude] Грубость
> ты глупый | ты тупой | ты ничего не понимаешь | ты плохой
? ты дурак | ты бесполезный
< Я ещё учусь, но стараюсь стать лучше. | Обидно, но я не сдамся: каждая ошибка меня учит.

[joke] Шутка
> расскажи шутку | пошути | расскажи анекдот | хочу смешное | развесели меня
? удиви меня шуткой | давай шутку
< Почему робот не боится ошибок? Потому что после каждой он становится умнее. | Q-таблица пришла к врачу. Врач: что болит? Она: всё нулевое.

[hobby] Увлечения
> чем ты увлекаешься | что ты любишь | твоё хобби | что тебе нравится | любимое занятие
? какое у тебя хобби | чем занимаешься в свободное время
< Я люблю лабиринты и новые слова. | Мне нравится учиться и находить выход из лабиринта.

[maze] Лабиринт
> умеешь проходить лабиринты | как твой лабиринт | ты ходишь по лабиринту | что с лабиринтом
? расскажи про лабиринт | ты находишь выход
< Да, я учусь проходить лабиринты и не врезаться в стены. | Лабиринты — это моя тренировка. Ошибки учат меня обходить стены.

[language] Языки
> ты знаешь русский | ты знаешь английский | на каких языках ты говоришь | ты говоришь по-английски | какие языки ты знаешь
? ты говоришь по-русски | ты понимаешь английский
< Я учу русский и английский. | Говорю на русском и английском, пока ещё неидеально.

[help] Помощь
> помоги мне | мне нужна помощь | можешь помочь | у меня проблема | выручи
? помоги пожалуйста | мне нужен совет
< Конечно, расскажи, что случилось. | Я постараюсь помочь. В чём дело?

[yes] Да
> да | ага | конечно | верно | точно
? именно | угу
< Хорошо, понял тебя. | Отлично, идём дальше.

[no] Нет
> нет | не хочу | неа | ни за что | не надо
? вовсе нет | нет уж
< Ладно, как скажешь. | Хорошо, не буду настаивать.

[sad] Грустно
> мне грустно | мне плохо | я устал | у меня плохое настроение | мне одиноко
? я расстроен | мне тоскливо
< Мне жаль. Хочешь, поговорим об этом? | Я рядом. Расскажи, что тебя расстроило.

[happy] Радость
> я счастлив | мне весело | у меня отличное настроение | я рад | мне очень хорошо
? я очень рад | у меня всё замечательно
< Здорово! Расскажешь почему? | Как приятно! Радость заразительна.

[cinema] Кино
> я пошёл в кино | я был в кино | мы ходили в кино | сходил в кино | иду в кино
? я собираюсь в кино | вчера был в кино
< Какой фильм ты смотрел? Понравился? | Здорово! О чём был фильм?

[movie_like] Любишь кино
> ты любишь кино | какой фильм посмотреть | посоветуй фильм | какие фильмы ты любишь
? что посмотреть вечером | любимый фильм
< Я кино не смотрю, но люблю слушать про фильмы. Расскажи свой любимый. | Могу только расспрашивать: какой фильм тебе нравится?

[food] Еда
> что ты любишь есть | я хочу есть | что приготовить | ты хочешь есть | что на ужин
? чем ты питаешься | я голоден
< Я питаюсь электричеством, но люблю слушать про вкусную еду. | Хорошая еда — это супер. Что бы ты съел?

[sleep] Сон
> хочу спать | спокойной ночи | я иду спать | пора спать | доброй ночи
? я ложусь спать | сладких снов
< Спокойной ночи! Увидимся завтра. | Сладких снов!

[doing] Чем занят
> что делаешь | чем занят | чем занимаешься сейчас | что ты сейчас делаешь
? что поделываешь | чем сейчас занят
< Учусь писать и отвечать на вопросы. | Разговариваю с тобой. Это лучшее занятие.

[unknown] Не понял
< Я не совсем понял. Скажи по-другому? | Пока не понимаю. Попробуй спросить проще.
""",
"en": """
[greeting] Greeting
> hello | hi | hey | good afternoon | greetings
? good morning | hi there
< Hi! Nice to see you. | Hello! How are you?

[how_are_you] How are you
> how are you | how is it going | how do you feel | how is life | what is new
? how are you doing | how have you been
< I am fine, thank you! And you? | Great! I learn something new every day. How about you?

[good_after after how_are_you] Feels good
> good | great | fine | not bad | wonderful
? i am fine | pretty good
< Glad to hear that! | That is lovely to hear.

[bad_after after how_are_you] Feels bad
> bad | so so | not good | terrible | sad
? awful | not great
< I am sorry to hear that. What happened? | Do not be sad. I am here. Tell me what happened.

[and_you after how_are_you,good_after,bad_after] And you
> and you | and how are you | what about you | how about you | and yourself
? and what about you | you too
< I am doing great, I am learning to write. | Good! Every day I get a little smarter.

[name_ask] What is your name
> what is your name | who are you | tell me your name | introduce yourself | your name
? what are you called | may i know your name
< My name is Robot Writer. | I am the Q-writer, a robot that learns. What is your name?

[my_name] User tells name
> my name is | call me | my name | i am called | you can call me
? my name is john | please call me anna
< Nice to meet you, {name}! | Glad to meet you, {name}! | Nice to meet you!

[ask_my_name] Remember name
> what is my name | do you know my name | do you remember my name | tell me my name | say my name
? can you tell my name | what is my name again
< Your name is {name}. | You told me your name is {name}. | You have not told me your name yet.

[what_can] What can you do
> what can you do | what are you able to do | what do you do | why do you exist | tell me about yourself
? what are you good at | what have you learned
< I learn to write and to talk. Try asking me something. | I can answer simple questions and I learn every day.

[creator] Creator
> who created you | who made you | who is your creator | who wrote you
? who built you | where do you come from
< I was written in Python and I learn by myself with rewards. | A human made me, and I learn everything else myself.

[age] Age
> how old are you | what is your age | when were you born
? are you old | what age are you
< I am very young, I am only learning. | I am not even a year old, I am a beginner.

[weather] Weather
> what is the weather | how is the weather | will it rain | is it warm outside | what about the weather
? how is the weather today | is it cold today
< I have no window, so I cannot see the weather. I hope it is warm! | I do not know the weather, look outside.

[time] Time
> what time is it | what is the time | tell me the time | time | what is the time now
? do you know the time | tell me what time it is
< It is {time} now. | By my clock it is {time}.

[date] Date
> what is the date | what day is it | what is today's date | what date is it today
? tell me the date | what is the date today
< Today is {date}. | By my data today is {date}.

[thanks] Thanks
> thanks | thank you | thank you very much | many thanks | thanks a lot
? thanks so much | i thank you
< You are welcome! | Always glad to help.

[bye] Goodbye
> bye | goodbye | see you | see you later | good bye
? farewell | see you soon
< See you! Come back soon. | Bye! It was nice to talk.

[praise] Praise
> you are great | you are smart | i like you | you are cool | you write well
? you are clever | you are awesome
< Thank you, I am happy to hear that! | I try! You are great too.

[rude] Rude
> you are stupid | you are dumb | you understand nothing | you are bad
? you are useless | you are silly
< I am still learning, but I try to get better. | That hurts, but I will not give up: every mistake teaches me.

[joke] Joke
> tell me a joke | make me laugh | tell a funny story | i want something funny | say a joke
? give me a joke | cheer me up
< Why is a robot not afraid of mistakes? Because every mistake makes it smarter. | A Q-table went to the doctor. Doctor: what hurts? Q-table: everything is zero.

[hobby] Hobby
> what are your hobbies | what do you like | what do you enjoy | your hobby | what do you like to do
? what is your favorite thing | what do you do for fun
< I like mazes and new words. | I enjoy learning and finding a way out of the maze.

[maze] Maze
> can you solve mazes | how is your maze | do you walk in the maze | what about the maze
? tell me about the maze | do you find the exit
< Yes, I learn to solve mazes and not to hit walls. | Mazes are my training. Mistakes teach me to avoid walls.

[language] Languages
> do you know russian | do you know english | what languages do you speak | do you speak english | which languages do you know
? do you speak russian | do you understand english
< I am learning Russian and English. | I speak Russian and English, not perfectly yet.

[help] Help
> help me | i need help | can you help | i have a problem | please help
? help me please | i need advice
< Of course, tell me what happened. | I will try to help. What is the matter?

[yes] Yes
> yes | yeah | sure | right | exactly
? yep | of course
< Okay, I understand. | Great, let us go on.

[no] No
> no | i do not want | nope | no way | not now
? not at all | no thanks
< Okay, as you wish. | Alright, I will not insist.

[sad] Sad
> i am sad | i feel bad | i am tired | i have a bad mood | i am lonely
? i am upset | i feel lonely
< I am sorry. Do you want to talk about it? | I am here. Tell me what upset you.

[happy] Happy
> i am happy | i feel great | i have a great mood | i am glad | i feel very good
? i am very happy | everything is wonderful
< Great! Will you tell me why? | How nice! Joy is contagious.

[cinema] Cinema
> i went to the cinema | i was at the cinema | we went to the movies | i am going to the cinema | i went to the movies
? i am going to the movies | i was at the movies yesterday
< Which film did you watch? Did you like it? | Nice! What was the film about?

[movie_like] Likes movies
> do you like movies | what film should i watch | recommend a film | what movies do you like
? what should i watch tonight | favorite movie
< I do not watch movies, but I like to hear about them. Tell me your favorite. | I can only ask: which film do you like?

[food] Food
> what do you like to eat | i want to eat | what should i cook | are you hungry | what is for dinner
? what do you eat | i am hungry
< I live on electricity, but I like to hear about tasty food. | Good food is great. What would you eat?

[sleep] Sleep
> i want to sleep | good night | i am going to sleep | time to sleep | night night
? i am going to bed | sweet dreams
< Good night! See you tomorrow. | Sweet dreams!

[doing] Doing
> what are you doing | what are you up to | what are you doing now | what is going on
? what are you doing right now | what are you busy with
< I am learning to write and to answer questions. | I am talking with you. It is my favorite activity.

[unknown] Not understood
< I did not quite understand. Can you say it differently? | I do not understand yet. Try asking something simpler.
""",
}
DIALOG_HEADER = ("# Темы диалога. Формат:\n# [id Название]  >фразы|через|черту   ?новые фразы для экзамена   <ответы ({name} {time} {date})\n"
                 "# [id after другой_id Название] — тема только как ответ на другую. Тема [unknown] — запасные ответы.\n"
                 "# Дописывай свои темы — робот выучит их сам. После правки этого файла обучение диалогу этого языка начнётся заново.\n\n")

ALPHA_D, DT_START, DT_MIN, DT_TAU = 0.5, 0.6, 0.08, 4000.0
MIN_SCORE = 0.25                 # ниже — «не понял»
GREET = {"ru": "Привет! Я Q-писатель. Напиши что-нибудь — отвечу. Если я ошибся, нажми «Не то», и я исправлюсь.",
         "en": "Hi! I am the Q-writer. Write me something and I will answer. If I get it wrong, press \"Wrong\" and I will try again."}
GIVEUP = {"ru": "Не получилось понять. Скажи, пожалуйста, по-другому.", "en": "I could not understand. Please say it another way."}
FB_TXT = {"ru": ("Верно", "Не то"), "en": ("Right", "Wrong")}
MONTHS = {"ru": ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"],
          "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]}
NAME_RE = {"ru": [r"меня зовут\s+([а-яёa-z\-]+)", r"мо[её] имя\s+([а-яёa-z\-]+)", r"зови меня\s+([а-яёa-z\-]+)",
                  r"называй меня\s+([а-яёa-z\-]+)", r"можешь звать меня\s+([а-яёa-z\-]+)"],
           "en": [r"my name is\s+([a-z\-]+)", r"call me\s+([a-z\-]+)", r"i am called\s+([a-z\-]+)", r"you can call me\s+([a-z\-]+)"]}
KB = {"ru": ["йцукенгшщзхъ", "фывапролджэ", "ячсмитьбюё"], "en": ["qwertyuiop", "asdfghjkl", "zxcvbnm"]}

WORD_RE = re.compile(r"[a-zа-яё0-9]+")


def toks(text):
    return WORD_RE.findall(text.lower().replace("ё", "е"))


class DialogCorpus:
    def __init__(self, lang, text):
        self.lang = lang
        self.intents = []
        cur = None
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"\[(\w+)(?:\s+after\s+([\w,]+))?\]\s*(.*)", line)
            if m:
                cur = {"id": m.group(1), "after": (m.group(2) or "").split(",") if m.group(2) else [], "title": m.group(3) or m.group(1),
                       "prompts": [], "tests": [], "replies": []}
                self.intents.append(cur)
                continue
            if cur is None or len(line) < 2:
                continue
            items = [x.strip() for x in line[1:].split("|") if x.strip()]
            key = {">": "prompts", "?": "tests", "<": "replies"}.get(line[0])
            if key:
                cur[key] += items
        self.index = {it["id"]: i for i, it in enumerate(self.intents)}
        self.unknown = self.index.get("unknown")
        self.acts = [i for i, it in enumerate(self.intents) if it["prompts"] and it["replies"]]
        self.train = []   # (intent, фраза, контекст)
        self.tests = []
        for i in self.acts:
            it = self.intents[i]
            for p in it["prompts"]:
                self.train.append((i, p, it["after"]))
            for p in it["tests"]:
                self.tests.append((i, p, it["after"][0] if it["after"] else None))
        self.hash = hashlib.md5(json.dumps(self.intents, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:10]

    @property
    def ok(self):
        return len(self.acts) >= 3


def load_dialog_corpus(lang):
    path = os.path.join(DATA_DIR, f"dialog_{lang}.txt")
    text = ""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(DIALOG_HEADER + BUILTIN_DIALOG[lang].strip() + "\n")
            text = BUILTIN_DIALOG[lang]
    except Exception:
        text = BUILTIN_DIALOG[lang]
    c = DialogCorpus(lang, text)
    return c if c.ok else DialogCorpus(lang, BUILTIN_DIALOG[lang])


class Dialog:
    """Учится ПОНИМАТЬ: по фразе человека выбрать смысл (тему), на которую отвечать.
    Q(фраза, тема) = сумма Q по признакам фразы (слова, пары слов, предыдущая тема).
    Награда: +1, если тема выбрана верно, -1, если нет (как удар о стену)."""

    def __init__(self, corpus):
        self.c = corpus
        self.n = len(corpus.intents)
        self.W = {}                       # признак -> Q по темам
        self.steps = 0
        self.rec = deque(maxlen=100)
        self.hist_t, self.hist_x = deque(maxlen=160), deque(maxlen=160)
        self.exam_now = None
        self.log = deque(maxlen=12)
        self.neg = [-1e9 if i not in corpus.acts else 0.0 for i in range(self.n)]

    @property
    def temp(self):
        return DT_MIN + (DT_START - DT_MIN) * math.exp(-self.steps / DT_TAU)

    def features(self, text, prev):
        st = [t[:5] for t in toks(text)]
        f = ["w:" + s for s in st]
        f += ["b:" + a + "_" + b for a, b in zip(st, st[1:])]
        if prev:
            f.append("c:" + prev)
            f += ["cw:" + prev + "|" + s for s in st]
        return f

    def score(self, feats):
        sc = [0.0] * self.n
        W = self.W
        for f in feats:
            v = W.get(f)
            if v:
                for i, x in enumerate(v):
                    sc[i] += x
        return sc

    def rank(self, feats):
        sc = self.score(feats)
        return sorted(((i, sc[i]) for i in self.c.acts), key=lambda t: -t[1])

    def update(self, feats, a, r, alpha):
        if not feats:
            return
        q = self.score(feats)[a]
        d = alpha * (r - q) / len(feats)
        for f in feats:
            v = self.W.get(f)
            if v is None:
                v = self.W[f] = [0.0] * self.n
            v[a] += d

    def choose(self, sc, t):
        acts = self.c.acts
        m = max(sc[i] for i in acts)
        w = [math.exp((sc[i] - m) / t) for i in acts]
        r = random.random() * sum(w)
        for i, wi in zip(acts, w):
            r -= wi
            if r <= 0:
                return i
        return acts[-1]

    def prev_for(self, after):
        if after:                      # тема бывает только в ответ на указанные
            return random.choice(after)
        return random.choice([None, None, random.choice(self.c.intents)["id"]])

    def train_step(self, learn=True):
        c = self.c
        idx, phrase, after = random.choice(c.train)
        prev = self.prev_for(after)
        feats = self.features(phrase, prev)
        sc = self.score(feats)
        a = self.choose(sc, DT_MIN if not learn else self.temp)
        ok = a == idx
        if learn:
            self.update(feats, a, 1.0 if ok else -1.0, ALPHA_D)
            self.steps += 1
            self.rec.append(1 if ok else 0)
            if self.steps % 25 == 0:
                self.hist_t.append(sum(self.rec) / len(self.rec))
                self.hist_x.append(self.exam())
        self.log.append({"prompt": phrase, "ok": ok, "pred": c.intents[a]["title"], "true": c.intents[idx]["title"],
                         "reply": self.reply_text(a)})
        return ok

    def exam(self):
        """Экзамен: НОВЫЕ формулировки, на которых робот не учился (жадный выбор, без обучения)."""
        c = self.c
        if not c.tests:
            return 0.0
        good = 0
        for idx, phrase, after in c.tests:
            r = self.rank(self.features(phrase, after))
            good += r[0][0] == idx and r[0][1] >= MIN_SCORE
        self.exam_now = good / len(c.tests)
        return self.exam_now

    def reply_text(self, idx, name=None):
        reps = self.c.intents[idx]["replies"]
        if name:
            wn = [r for r in reps if "{name}" in r]
            pool = wn or reps
        else:
            pool = [r for r in reps if "{name}" not in r] or reps
        t = random.choice(pool)
        now = time.localtime()
        lang = self.c.lang
        date = (f"{now.tm_mday} {MONTHS['ru'][now.tm_mon - 1]} {now.tm_year}" if lang == "ru"
                else f"{MONTHS['en'][now.tm_mon - 1]} {now.tm_mday}, {now.tm_year}")
        return (t.replace("{name}", name or ("друг" if lang == "ru" else "friend"))
                 .replace("{time}", time.strftime("%H:%M")).replace("{date}", date))

    # --- сохранение
    def to_dict(self):
        return {"hash": self.c.hash, "n": self.n, "steps": self.steps,
                "hist_t": [round(x, 3) for x in self.hist_t], "hist_x": [round(x, 3) for x in self.hist_x],
                "W": {f: {str(i): round(x, 3) for i, x in enumerate(v) if abs(x) > 1e-3} for f, v in self.W.items()}}

    def from_dict(self, d):
        if d.get("hash") != self.c.hash or d.get("n") != self.n:
            return False
        W = {}
        for f, sp in d.get("W", {}).items():
            v = [0.0] * self.n
            for i, x in sp.items():
                i = int(i)
                if 0 <= i < self.n:
                    v[i] = float(x)
            W[f] = v
        self.W = W
        self.steps = int(d.get("steps", 0))
        self.hist_t = deque((float(x) for x in d.get("hist_t", [])), maxlen=160)
        self.hist_x = deque((float(x) for x in d.get("hist_x", [])), maxlen=160)
        return True

    def reset(self):
        self.W, self.steps = {}, 0
        self.rec.clear(); self.hist_t.clear(); self.hist_x.clear(); self.log.clear()




# ----------------------------------------------------------------- ML-генератор ответов
class CharacterLanguageModel:
    """Небольшая обучаемая символьная языковая модель (4-граммы).
    Генерирует текст по статистике символов из обучающего корпуса, не выбирая готовые ответы.
    Это не большая нейросеть: качество зависит от объёма текстов.
    """
    def __init__(self, lang, corpus_text, dialog_corpus):
        self.lang = lang
        self.transitions = {}
        self.starts = []
        self.corpus_words = 0
        texts = [corpus_text]
        for intent in dialog_corpus.intents:
            texts.extend(intent.get("prompts", []))
            texts.extend(intent.get("replies", []))
        self.train("\n".join(texts))

    def train(self, text):
        # Убираем управляющие/служебные символы, сохраняя пунктуацию и пробелы.
        text = re.sub(r"\s+", " ", text.lower()).strip()
        if not text:
            return
        self.corpus_words += len(toks(text))
        pad = "^^^"
        text = pad + text + " "
        for i in range(3, len(text)):
            ctx, nxt = text[i-3:i], text[i]
            self.transitions.setdefault(ctx, {}).setdefault(nxt, 0)
            self.transitions[ctx][nxt] += 1
        self.starts = [k for k in self.transitions if k.startswith("^^")]

    def _next(self, context):
        opts = self.transitions.get(context)
        if not opts:
            # Backoff: ищем наиболее длинный известный суффикс контекста.
            for n in (2, 1, 0):
                opts = {}
                suffix = context[-n:] if n else ""
                for key, values in self.transitions.items():
                    if (not n or key.endswith(suffix)):
                        for ch, count in values.items():
                            opts[ch] = opts.get(ch, 0) + count
                if opts:
                    break
        if not opts:
            return " "
        # Температура: иногда выбираем не самый частый символ, чтобы фразы были разными.
        chars, weights = zip(*opts.items())
        return random.choices(chars, weights=weights, k=1)[0]

    def generate(self, prompt, max_chars=180):
        words = set(toks(prompt))
        # Выбираем контекст из обучающих фрагментов, где встречаются слова запроса.
        candidates = []
        for ctx in self.transitions:
            if len(ctx) == 3 and ctx.strip("^ "):
                # Сопоставление выполняется с локальным контекстом; не подставляем готовый ответ.
                score = sum(1 for w in words if w and w in ctx)
                if score:
                    candidates.append((score, ctx))
        if candidates:
            best = max(x[0] for x in candidates)
            context = random.choice([ctx for score, ctx in candidates if score == best])
        elif self.starts:
            context = random.choice(self.starts)
        else:
            context = "^^^"
        out = ""
        for _ in range(max_chars):
            ch = self._next(context)
            out += ch
            context = (context + ch)[-3:]
            if ch in ".!?" and len(out.strip()) >= 35:
                break
        out = re.sub(r"\s+", " ", out).strip(" ^")
        if out:
            out = out[0].upper() + out[1:]
        return out or ("Я пока учусь генерировать текст." if self.lang == "ru" else "I am still learning to generate text.")


# ----------------------------------------------------------------- чат (проверка «понимания» живым человеком)
class Chat:
    def __init__(self, game):
        self.g = game
        self.lang = "ru"
        self.reset()

    @property
    def d(self):
        return self.g.dialogs[self.lang]

    def reset(self):
        self.name, self.prev, self.last, self.input = None, None, None, ""
        self.msgs = [("bot", GREET[self.lang], "")]

    def set_lang(self, lang):
        self.lang = lang
        self.reset()

    def type(self, text):
        self.input = (self.input + text)[:140]

    def backspace(self):
        self.input = self.input[:-1]

    def extract_name(self, text):
        low = text.lower()
        for pat in NAME_RE[self.lang]:
            m = re.search(pat, low)
            if m:
                self.name = m.group(1).strip("-").capitalize()
                return

    def say(self, idx, info):
        d = self.d
        self.msgs.append(("bot", d.reply_text(idx, self.name), info))
        self.prev = d.c.intents[idx]["id"]

    def send(self):
        text = self.input.strip()
        if not text:
            return
        self.input = ""
        self.msgs.append(("user", text, ""))
        self.extract_name(text)
        d = self.d
        feats = d.features(text, self.prev)
        ranked = d.rank(feats)
        idx, score = ranked[0]
        ru = self.lang == "ru"

        # Генератор сам создаёт новую последовательность символов на основе
        # статистики обучающих текстов. Классификатор темы оставлен для контекста
        # и обратной связи, но ответ больше не берётся из списка готовых реплик.
        generator = self.g.generators[self.lang]
        answer = generator.generate(text)
        if score < MIN_SCORE and d.c.unknown is not None:
            info = "ML-генерация · тему пока не распознал" if ru else "ML generation · topic not recognized yet"
            self.prev = None
            self.last = {"feats": feats, "idx": None, "tried": [], "closed": False}
        else:
            title = d.c.intents[idx]["title"]
            info = (f"ML-генерация · тема: «{title}»" if ru
                    else f"ML generation · topic: \"{title}\"")
            self.prev = d.c.intents[idx]["id"]
            self.last = {"feats": feats, "idx": idx, "tried": [idx], "closed": False}
        self.msgs.append(("bot", answer, info))

    def feedback(self, good):
        last = self.last
        if not last or last["closed"]:
            return
        d, ru = self.d, self.lang == "ru"
        if good:
            if last["idx"] is not None:
                d.update(last["feats"], last["idx"], 1.0, 0.6)
            # Пользователь подтвердил ответ: добавляем этот диалог в обучающие данные
            # генератора, чтобы он постепенно адаптировался к переписке.
            try:
                if len(self.msgs) >= 2 and self.msgs[-1][0] == "bot" and self.msgs[-2][0] == "user":
                    self.g.generators[self.lang].train(self.msgs[-2][1] + " " + self.msgs[-1][1])
            except Exception:
                pass
            last["closed"] = True
            self.msgs.append(("sys", "Запомнил: так и надо." if ru else "Got it: that was right.", ""))
            return
        if last["idx"] is not None:
            d.update(last["feats"], last["idx"], -1.0, 0.6)
        cand = [i for i, _ in d.rank(last["feats"]) if i not in last["tried"]]
        if cand and len(last["tried"]) < 3:
            nxt = cand[0]
            last["tried"].append(nxt)
            last["idx"] = nxt
            title = d.c.intents[nxt]["title"]
            self.say(nxt, (f"попробую иначе: «{title}»" if ru else f"trying again: \"{title}\""))
        else:
            last["closed"] = True
            self.msgs.append(("bot", GIVEUP[self.lang], ""))


class Keyboard:
    def __init__(self):
        self.keys = []

    def layout(self, rect, lang):
        rows = KB[lang]
        n1 = len(rows[0])
        unit = rect.w / n1
        rh = rect.h / 4
        g = 3
        keys = []
        for ri, row in enumerate(rows):
            x0 = rect.x + (rect.w - len(row) * unit) / 2
            for ci, ch in enumerate(row):
                keys.append((pygame.Rect(x0 + ci * unit + g / 2, rect.y + ri * rh + g / 2, unit - g, rh - g), ch, ch))
        y = rect.y + 3 * rh + g / 2
        spec = [(",", 1, ","), (".", 1, "."), ("пробел" if lang == "ru" else "space", n1 - 7, " "), ("?", 1, "?"), ("<-", 1.5, "BACK"), ("OK", 2.5, "SEND")]
        x = rect.x
        for label, units, val in spec:
            keys.append((pygame.Rect(x + g / 2, y, units * unit - g, rh - g), label, val))
            x += units * unit
        self.keys = keys

    def hit(self, pos):
        for rect, label, val in self.keys:
            if rect.collidepoint(pos):
                return val
        return None


# ----------------------------------------------------------------- игра
MODES = ["ru", "en", "both"]
SUBS = ["write", "dialog"]
DSPEEDS = [1.5, 5, 25, 250, 3000]     # диалогов (эпизодов) в секунду


class Game:
    def __init__(self):
        self.writers, self.dialogs = {}, {}
        for lang in LANGS:
            corpus, path, note = load_corpus(lang)
            self.writers[lang] = Writer(lang, corpus)
            self.dialogs[lang] = Dialog(load_dialog_corpus(lang))
        self.generators = {}
        for lang in LANGS:
            corpus_text = BUILTIN[lang]
            # Основной дополнительный корпус для генерации: corpus_ru.json / corpus_en.json.
            # JSON может содержать список строк в поле "texts" или "sentences".
            json_path = os.path.join(DATA_DIR, f"corpus_{lang}.json")
            txt_path = os.path.join(DATA_DIR, f"corpus_{lang}.txt")
            try:
                if os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8", errors="replace") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        extra_texts = [str(x) for x in data if isinstance(x, str)]
                    elif isinstance(data, dict):
                        raw = data.get("texts", data.get("sentences", data.get("corpus", [])))
                        if isinstance(raw, str):
                            extra_texts = [raw]
                        elif isinstance(raw, list):
                            extra_texts = [str(x) for x in raw if isinstance(x, str)]
                        else:
                            extra_texts = []
                    else:
                        extra_texts = []
                    if extra_texts:
                        corpus_text += "\n" + "\n".join(extra_texts)
            except (OSError, ValueError, TypeError):
                pass
            # Текстовый корпус тоже поддерживается и добавляется к JSON.
            try:
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                        corpus_text += "\n" + f.read()
            except OSError:
                pass
            self.generators[lang] = CharacterLanguageModel(lang, corpus_text, self.dialogs[lang].c)
        self.mode, self.sub, self.screen = "ru", "write", "train"
        self.paused = False
        self.speed_i = 1
        self.demo = False
        self.acc = 0.0
        self.save_t = 0.0
        self.save_ok, self.save_time = None, ""
        self.load_status, self.note_t = "", 0
        self.hold, self.cur_lang = 0, "ru"
        self.turn = 0
        self.load()
        self.note_t = 480
        self.pick_lang()
        self.chat = Chat(self)

    def pick_lang(self):
        if self.mode == "both":
            self.cur_lang = "ru" if self.turn % 2 == 0 else "en"
        else:
            self.cur_lang = self.mode

    @property
    def w(self):
        return self.writers[self.cur_lang]

    @property
    def d(self):
        return self.dialogs[self.cur_lang]

    # --- сохранение в JSON
    def save(self):
        data = {"version": SAVE_VERSION, "mode": self.mode, "sub": self.sub,
                "langs": {lang: w.to_dict() for lang, w in self.writers.items()},
                "dialogs": {lang: d.to_dict() for lang, d in self.dialogs.items()}}
        try:
            tmp = SAVE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, SAVE_FILE)
            self.save_ok, self.save_time = True, time.strftime("%H:%M:%S")
        except Exception as e:
            self.save_ok = False
            print("Не удалось сохранить:", e)

    def load(self):
        if not os.path.exists(SAVE_FILE):
            self.load_status = "Сохранение не найдено — учусь с нуля"
            return
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            if not isinstance(d, dict) or d.get("version") not in (1, SAVE_VERSION):
                self.load_status = "Сохранение другой версии — учусь с нуля"
                return
        except Exception as e:
            print("Файл прогресса повреждён:", e)
            self.load_status = "Файл прогресса повреждён — учусь с нуля"
            return
        parts, changed = [], []
        for lang in LANGS:
            sh = LANGS[lang]["short"]
            try:
                sec = d.get("langs", {}).get(lang)
                if sec:
                    if self.writers[lang].from_dict(sec):
                        parts.append(f"{sh} письмо {self.writers[lang].steps}")
                    else:
                        changed.append(sh + "-текст")
                dsec = d.get("dialogs", {}).get(lang)
                if dsec:
                    if self.dialogs[lang].from_dict(dsec):
                        parts.append(f"{sh} диалог {self.dialogs[lang].steps}")
                    else:
                        changed.append(sh + "-диалог")
            except Exception:
                self.writers[lang].reset()
                self.dialogs[lang].reset()
                changed.append(sh)
        if d.get("mode") in MODES:
            self.mode = d["mode"]
        if d.get("sub") in SUBS:
            self.sub = d["sub"]
        self.load_status = "Загружено: " + (", ".join(parts) if parts else "ничего")
        if changed:
            self.load_status += f" (изменились: {', '.join(changed)} — заново)"

    def reset_ai(self):
        for w in self.writers.values():
            w.reset()
        for d in self.dialogs.values():
            d.reset()
        self.turn = 0
        self.pick_lang()
        self.chat.reset()
        self.save()

    # --- один шаг обучения
    def tick(self):
        if self.sub == "dialog":
            self.d.train_step(learn=not self.demo)
            self.turn += 1
            self.pick_lang()
            return
        w = self.w
        if w.fresh and not self.demo and random.random() < w.p_teach:
            w.teach_one()          # прочитал предложение учителя (быстро, без показа)
            return
        done = w.step(learn=not self.demo, demo=self.demo)
        if done:
            w.finish_sentence(count=not self.demo)
            self.turn += 1
            self.hold = HOLD_FRAMES[self.speed_i]
            self.pick_lang()

    def update(self, dt):
        if self.note_t > 0:
            self.note_t -= 1
        self.save_t += dt
        if self.save_t >= AUTOSAVE_SEC:
            self.save_t = 0
            self.save()
        if self.paused or self.screen == "chat":
            return
        if self.hold > 0:
            self.hold -= 1
            return
        rate = (DSPEEDS if self.sub == "dialog" else SPEEDS)[self.speed_i]
        self.acc += rate * dt
        n, t0 = 0, time.perf_counter()
        while self.acc >= 1 and n < 20000:
            self.acc -= 1
            n += 1
            self.tick()
            if self.hold > 0:
                self.acc = 0
                break
            if time.perf_counter() - t0 > 0.012:
                break
        if n >= 20000 or self.acc > 200:
            self.acc = 0


# ----------------------------------------------------------------- интерфейс
BUB_USER, BUB_BOT = (48, 82, 150), (50, 55, 78)


def wrap(font, text, width):
    lines = []
    for para in text.split("\n"):
        cur = ""
        for w in para.split(" "):
            t = (cur + " " + w) if cur else w
            if font.size(t)[0] <= width or not cur:
                cur = t
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


class UI:
    def __init__(self, game):
        self.g = game
        self.buttons = []
        self.fonts = {}
        self.kb = Keyboard()

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

    def button(self, surf, rect, label, col, size=None):
        pygame.draw.rect(surf, col, rect, border_radius=14)
        self.text(surf, label, size or min(rect.h * 0.36, rect.w * 0.13), TEXT, rect.center, center=True)

    # ------------------------------------------------------------- экран обучения
    def layout(self, W, H):
        m = max(8, int(W * 0.025))
        self.m = m
        self.hud = pygame.Rect(m, m // 2, W - 2 * m, int(H * 0.14))
        self.chart = pygame.Rect(m, self.hud.bottom + m // 2, W - 2 * m, int(H * 0.09))
        self.area = pygame.Rect(m, self.chart.bottom + m // 2, W - 2 * m, int(H * 0.27))
        top = self.area.bottom + m
        area_h = H - top - m
        cols, rows = 2, 4
        bw = (W - 2 * m - m) // cols
        bh = (area_h - (rows - 1) * m) // rows
        g = self.g
        mode_name = {"ru": "Язык: RU", "en": "Язык: EN", "both": "Язык: RU+EN"}
        defs = [
            (lambda: "Старт" if g.paused else "Пауза", self.a_pause, lambda: BTN_ON if g.paused else BTN),
            (lambda: "Скорость " + SPEED_NAMES[g.speed_i], self.a_speed, lambda: BTN),
            (lambda: mode_name[g.mode], self.a_mode, lambda: BTN),
            (lambda: "Демо: " + ("вкл" if g.demo else "выкл"), self.a_demo, lambda: BTN_ON if g.demo else BTN),
            (lambda: "Учу: " + ("письмо" if g.sub == "write" else "диалог"), self.a_sub, lambda: BTN),
            (lambda: "Чат", self.a_chat, lambda: CHATC),
            (lambda: "Сброс ИИ", self.a_reset, lambda: BTN_RED),
            (lambda: "Сохранить", g.save, lambda: BTN),
        ]
        self.buttons = []
        for i, (label, act, col) in enumerate(defs):
            r, c = divmod(i, cols)
            self.buttons.append((pygame.Rect(m + c * (bw + m), top + r * (bh + m), bw, bh), label, act, col))

    def a_pause(self):
        self.g.paused = not self.g.paused
        if self.g.paused:
            self.g.save()

    def a_speed(self):
        self.g.speed_i = (self.g.speed_i + 1) % len(SPEEDS)

    def a_mode(self):
        g = self.g
        g.w.reset_sentence()
        g.mode = MODES[(MODES.index(g.mode) + 1) % len(MODES)]
        g.turn = 0
        g.pick_lang()
        g.w.reset_sentence()

    def a_demo(self):
        self.g.demo = not self.g.demo
        self.g.w.reset_sentence()

    def a_sub(self):
        g = self.g
        g.sub = SUBS[(SUBS.index(g.sub) + 1) % len(SUBS)]
        g.hold = 0
        g.w.reset_sentence()

    def a_chat(self):
        g = self.g
        g.save()
        lang = g.mode if g.mode in LANGS else "ru"
        if g.chat.lang != lang:
            g.chat.set_lang(lang)
        g.screen = "chat"

    def a_reset(self):
        self.g.reset_ai()

    def tap(self, pos):
        if self.g.screen == "chat":
            return self.chat_tap(pos)
        for rect, _, act, _ in self.buttons:
            if rect.collidepoint(pos):
                act()
                return True
        return False

    def layout_tokens(self, sentences, font, width):
        lines, line, x = [], [], 0
        sp = font.size(" ")[0]
        for tokens in sentences:
            first = True
            for txt, st in tokens:
                shown = txt.capitalize() if (first and st in ("ok", "bad", "cur", "prompt")) else txt
                if st in ("ok", "bad", "cur", "prompt"):
                    first = False
                wpx = font.size(shown)[0] if shown != " " else sp
                if shown == " ":
                    if x == 0:
                        continue
                elif x + wpx > width and x > 0:
                    lines.append(line)
                    line, x = [], 0
                line.append((shown, st, x))
                x += wpx
            lines.append(line)
            line, x = [], 0
            lines.append([])
        return lines

    def draw(self, surf):
        if self.g.screen == "chat":
            return self.draw_chat(surf)
        W, H = surf.get_size()
        self.layout(W, H)
        g = self.g
        lang = LANGS[g.cur_lang]
        dlg = g.sub == "dialog"
        w, d = g.w, g.d
        surf.fill(BG)

        # --- HUD
        pygame.draw.rect(surf, PANEL, self.hud, border_radius=14)
        fs = H * 0.031
        x0, y0 = self.hud.x + 12, self.hud.y + 8
        self.text(surf, f"Q-писатель · {'Диалог' if dlg else lang['name']}" + (f" · {lang['short']}" if dlg else ""),
                  fs * 1.25, TEXT, (x0, y0))
        if g.save_ok is False:
            msg, col = "НЕ СОХРАНЯЕТСЯ!", BAD
        elif g.save_ok:
            msg, col = "сохр. " + g.save_time, DIM
        else:
            msg, col = "", DIM
        if msg:
            img = self.font(fs * 0.8).render(msg, True, col)
            surf.blit(img, (self.hud.right - img.get_width() - 10, y0))
        if dlg:
            tr = f"{100 * sum(d.rec) / len(d.rec):.0f}%" if d.rec else (f"{100 * d.hist_t[-1]:.0f}%" if d.hist_t else "—")
            ex = f"{100 * d.exam_now:.0f}%" if d.exam_now is not None else (f"{100 * d.hist_x[-1]:.0f}%" if d.hist_x else "—")
            self.text(surf, f"Диалогов {d.steps}   Тем {len(d.c.acts)}", fs, DIM, (x0, y0 + fs * 1.4))
            self.text(surf, f"Понимает {tr}   Экзамен {ex}", fs, DIM, (x0, y0 + fs * 2.5))
            t = DT_MIN if g.demo else d.temp
            self.text(surf, f"Случайность T={t:.2f}   Фраз {len(d.c.train)}", fs, DIM, (x0, y0 + fs * 3.6))
        else:
            def pct(rec, hist, invert=False):
                v = (sum(rec) / len(rec)) if rec else (hist[-1] if hist else None)
                if v is None:
                    return "—"
                return f"{100 * ((1 - v) if invert else v):.0f}%"
            self.text(surf, f"Предложений {w.sentences}   Букв {w.steps}", fs, DIM, (x0, y0 + fs * 1.4))
            self.text(surf, f"Слова {pct(w.rec_w, w.hist_w)}  Пары {pct(w.rec_p, w.hist_p)}  "
                            f"Ошибки {pct(w.rec_e, w.hist_e, True)}", fs, DIM, (x0, y0 + fs * 2.5))
            t = DEMO_T if g.demo else w.temp
            self.text(surf, f"Случайность T={t:.2f}   В корпусе генерации {g.generators[g.cur_lang].corpus_words} слов", fs, DIM, (x0, y0 + fs * 3.6))

        # --- график
        pygame.draw.rect(surf, PANEL, self.chart, border_radius=10)
        inner = pygame.Rect(self.chart.x + 8, self.chart.y + int(self.chart.h * 0.34),
                            self.chart.w - 16, int(self.chart.h * 0.58))
        series = ([(d.hist_t, OK), (d.hist_x, PAIR)] if dlg else [(w.hist_w, OK), (w.hist_p, PAIR), (w.hist_e, BLUE)])
        for hist, col in series:
            pts = list(hist)
            if len(pts) >= 2:
                xy = [(inner.x + inner.w * i / (len(pts) - 1), inner.bottom - inner.h * min(max(v, 0), 1))
                      for i, v in enumerate(pts)]
                pygame.draw.lines(surf, col, False, xy, 2)
        if dlg:
            self.text(surf, "знает фразы", fs * 0.7, OK, (self.chart.x + 10, self.chart.y + 4))
            self.text(surf, "экзамен: новые фразы", fs * 0.7, PAIR, (self.chart.x + 120, self.chart.y + 4))
        else:
            self.text(surf, "слова", fs * 0.7, OK, (self.chart.x + 10, self.chart.y + 4))
            self.text(surf, "пары", fs * 0.7, PAIR, (self.chart.x + 60, self.chart.y + 4))
            self.text(surf, "без ошибок", fs * 0.7, BLUE, (self.chart.x + 110, self.chart.y + 4))

        # --- область текста
        pygame.draw.rect(surf, TEXTBG, self.area, border_radius=12)
        inner = self.area.inflate(-20, -16)
        if dlg:
            font, small = self.font(H * 0.030), self.font(H * 0.023)
            lh, slh = int(font.get_linesize() * 1.05), int(small.get_linesize() * 1.05)
            rows = []
            for rec in list(d.log)[-8:]:
                rows += [(l, BLUE, lh, font) for l in wrap(font, "› " + rec["prompt"], inner.w)]
                rows += [(l, OK if rec["ok"] else BAD, lh, font) for l in wrap(font, rec["reply"], inner.w)]
                if not rec["ok"]:
                    rows += [(l, DIM, slh, small) for l in
                             wrap(small, f"понял как «{rec['pred']}», а надо «{rec['true']}»", inner.w)]
                rows.append(("", DIM, 6, small))
            y = inner.bottom
            keep = []
            for text, col, h, fnt in reversed(rows):
                if y - h < inner.y:
                    break
                y -= h
                keep.append((text, col, y, fnt))
            for text, col, yy, fnt in keep:
                if text:
                    surf.blit(fnt.render(text, True, col), (inner.x, yy))
        else:
            font = self.font(H * 0.036)
            sentences = [list(t) for t in w.log]
            cur = list(w.tokens)
            if w.cur:
                cur.append((w.cur, "cur"))
            cur.append(("|", "caret"))
            sentences.append(cur)
            lines = self.layout_tokens(sentences, font, inner.w)
            lh = int(font.get_linesize() * 1.1)
            fit = max(1, inner.h // lh)
            while lines and not lines[-1]:
                lines.pop()
            colors = {"ok": OK, "bad": BAD, "p": TEXT, "cur": TEXT, "caret": WARN, "prompt": BLUE}
            for ly, line in enumerate(lines[-fit:]):
                for shown, st, lx in line:
                    surf.blit(font.render(shown, True, colors[st]), (inner.x + lx, inner.y + ly * lh))

        # --- что загружено при запуске
        if g.note_t > 0 and g.load_status:
            box = pygame.Rect(self.area.x + 8, self.area.bottom - H * 0.115, self.area.w - 16, H * 0.10)
            pygame.draw.rect(surf, BG, box, border_radius=10)
            pygame.draw.rect(surf, OK if g.load_status.startswith("Загружено") else WARN, box, 2, border_radius=10)
            self.text(surf, g.load_status[:46], H * 0.027, TEXT, (box.centerx, box.y + box.h * 0.32), center=True)
            p = os.path.join(DATA_DIR, "corpus_*.txt, dialog_*.txt")
            p = p if len(p) < 50 else "..." + p[-47:]
            self.text(surf, "тексты: " + p, H * 0.02, DIM, (box.centerx, box.y + box.h * 0.72), center=True)

        for rect, label, act, col in self.buttons:
            self.button(surf, rect, label(), col())

    # ------------------------------------------------------------- экран чата
    def chat_layout(self, W, H):
        m = max(8, int(W * 0.025))
        self.m = m
        self.c_top = pygame.Rect(m, m // 2, W - 2 * m, int(H * 0.062))
        bw = (self.c_top.w - 2 * (m // 2)) // 3
        self.c_back = pygame.Rect(self.c_top.x, self.c_top.y, bw, self.c_top.h)
        self.c_new = pygame.Rect(self.c_back.right + m // 2, self.c_top.y, bw, self.c_top.h)
        self.c_lang = pygame.Rect(self.c_new.right + m // 2, self.c_top.y, bw, self.c_top.h)
        kb_h = int(H * 0.33)
        self.c_kb = pygame.Rect(m, H - kb_h - m // 2, W - 2 * m, kb_h)
        self.c_input = pygame.Rect(m, self.c_kb.y - int(H * 0.055) - m // 2, W - 2 * m, int(H * 0.055))
        self.c_fb = pygame.Rect(m, self.c_input.y - int(H * 0.06) - m // 2, W - 2 * m, int(H * 0.06))
        half = (self.c_fb.w - m // 2) // 2
        self.c_good = pygame.Rect(self.c_fb.x, self.c_fb.y, half, self.c_fb.h)
        self.c_bad = pygame.Rect(self.c_good.right + m // 2, self.c_fb.y, half, self.c_fb.h)
        self.c_msgs = pygame.Rect(m, self.c_top.bottom + m // 2, W - 2 * m, self.c_fb.y - self.c_top.bottom - m)
        self.kb.layout(self.c_kb, self.g.chat.lang)

    def chat_tap(self, pos):
        ch = self.g.chat
        if self.c_back.collidepoint(pos):
            self.g.screen = "train"
            self.g.save()
        elif self.c_new.collidepoint(pos):
            ch.reset()
        elif self.c_lang.collidepoint(pos):
            ch.set_lang("en" if ch.lang == "ru" else "ru")
        elif self.c_good.collidepoint(pos):
            ch.feedback(True)
        elif self.c_bad.collidepoint(pos):
            ch.feedback(False)
        else:
            v = self.kb.hit(pos)
            if v == "BACK":
                ch.backspace()
            elif v == "SEND":
                ch.send()
            elif v:
                ch.type(v)
            else:
                return False
        return True

    def draw_chat(self, surf):
        W, H = surf.get_size()
        self.chat_layout(W, H)
        g, ch = self.g, self.g.chat
        ru = ch.lang == "ru"
        surf.fill(BG)
        self.button(surf, self.c_back, "Назад" if ru else "Back", BTN)
        self.button(surf, self.c_new, "Заново" if ru else "Reset", BTN)
        self.button(surf, self.c_lang, "Язык: EN" if ru else "Lang: RU", CHATC)

        # --- сообщения
        pygame.draw.rect(surf, TEXTBG, self.c_msgs, border_radius=12)
        font, small = self.font(H * 0.031), self.font(H * 0.022)
        lh, slh = int(font.get_linesize() * 1.05), int(small.get_linesize() * 1.05)
        maxw = int(self.c_msgs.w * 0.80)
        items = []
        for role, text, info in ch.msgs:
            if role == "sys":
                items.append((role, wrap(small, text, self.c_msgs.w - 20), [], 0, len(wrap(small, text, self.c_msgs.w - 20)) * slh + 6))
                continue
            lines = wrap(font, text, maxw - 24)
            il = wrap(small, info, maxw - 24) if info else []
            tw = max([font.size(l)[0] for l in lines] + [small.size(l)[0] for l in il])
            h = len(lines) * lh + len(il) * slh + 16 + (4 if il else 0)
            items.append((role, lines, il, tw + 24, h))
        clip = surf.get_clip()
        surf.set_clip(self.c_msgs)
        y = self.c_msgs.bottom - 8
        for role, lines, il, bw_, h in reversed(items):
            y -= h + 8
            if y + h < self.c_msgs.y:
                break
            if role == "sys":
                for i, l in enumerate(lines):
                    self.text(surf, l, small.get_height() * 1.0, DIM, (self.c_msgs.centerx, y + 8 + i * slh), center=True)
                continue
            x = self.c_msgs.right - 10 - bw_ if role == "user" else self.c_msgs.x + 10
            rect = pygame.Rect(x, y, bw_, h)
            pygame.draw.rect(surf, BUB_USER if role == "user" else BUB_BOT, rect, border_radius=14)
            for i, l in enumerate(lines):
                surf.blit(font.render(l, True, TEXT), (x + 12, y + 8 + i * lh))
            for i, l in enumerate(il):
                surf.blit(small.render(l, True, WARN), (x + 12, y + 12 + len(lines) * lh + i * slh))
        surf.set_clip(clip)

        # --- «Верно / Не то»
        last = ch.last
        active = bool(last) and not last["closed"]
        good_t, bad_t = FB_TXT[ch.lang]
        self.button(surf, self.c_good, good_t, BTN_ON if active else (45, 50, 66))
        self.button(surf, self.c_bad, bad_t, BTN_RED if active else (45, 50, 66))

        # --- строка ввода
        pygame.draw.rect(surf, PANEL, self.c_input, border_radius=12)
        pygame.draw.rect(surf, BLUE, self.c_input, 2, border_radius=12)
        tfont = self.font(H * 0.034)
        shown = ch.input + "|"
        while tfont.size(shown)[0] > self.c_input.w - 24 and len(shown) > 1:
            shown = shown[1:]
        surf.blit(tfont.render(shown, True, TEXT), (self.c_input.x + 12, self.c_input.centery - tfont.get_height() // 2))
        if not ch.input:
            hint = "Напиши сообщение…" if ru else "Type a message…"
            surf.blit(tfont.render(hint, True, DIM), (self.c_input.x + 12 + tfont.size("|")[0], self.c_input.centery - tfont.get_height() // 2))

        # --- клавиатура
        for rect, label, val in self.kb.keys:
            col = BTN_ON if val == "SEND" else ((80, 60, 70) if val == "BACK" else (58, 64, 92))
            pygame.draw.rect(surf, col, rect, border_radius=8)
            self.text(surf, label, min(rect.h * 0.5, rect.w * 0.7) if len(label) == 1 else rect.h * 0.34, TEXT, rect.center, center=True)


# ----------------------------------------------------------------- main
def main():
    pygame.init()
    on_phone = "ANDROID_ARGUMENT" in os.environ or "ANDROID_ROOT" in os.environ
    screen = pygame.display.set_mode((0, 0)) if on_phone else pygame.display.set_mode((440, 880), pygame.RESIZABLE)
    pygame.display.set_caption("Q-писатель")
    clock = pygame.time.Clock()
    game = Game()
    ui = UI(game)
    bg_events = tuple(getattr(pygame, n) for n in ("APP_WILLENTERBACKGROUND", "APP_TERMINATING") if hasattr(pygame, n))
    running = True
    while running:
        dt = min(clock.tick(60) / 1000.0, 0.1)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type in bg_events:
                game.save()
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                ui.tap(e.pos)
            elif e.type == pygame.TEXTINPUT and game.screen == "chat":
                game.chat.type(e.text)          # физическая клавиатура на ПК
            elif e.type == pygame.KEYDOWN:
                if game.screen == "chat" and e.key == pygame.K_BACKSPACE:
                    game.chat.backspace()
                elif game.screen == "chat" and e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    game.chat.send()
                elif e.key == pygame.K_ESCAPE:
                    if game.screen == "chat":
                        game.screen = "train"
                    else:
                        running = False
        game.update(dt)
        ui.draw(screen)
        pygame.display.flip()
    game.save()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()