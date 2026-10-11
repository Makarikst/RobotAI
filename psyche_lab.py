#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSYCHE 2.0 — продвинутая экспериментальная нейросеть
с self-моделью, эмоциями, совестью, внутренним монологом и моделью мира.

Важно:
Это симуляция. У системы нет субъективного опыта.
Но архитектура максимально приближена к убедительному внутреннему миру.
"""

import os
import sys
import json
import math
import random
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Константы
# ============================================================

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_FILE = "psyche_model.pt"
VOCAB_FILE = "psyche_vocab.json"
MEMORY_FILE = "psyche_memory.json"
STATE_FILE = "psyche_state.json"

BLOCK_SIZE = 192
BATCH_SIZE = 32
EMBED_SIZE = 160
HIDDEN_SIZE = 320
LAYERS = 3

# ============================================================
# Нейросеть
# ============================================================

class PsycheNet(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_SIZE)
        self.rnn = nn.GRU(
            EMBED_SIZE,
            HIDDEN_SIZE,
            num_layers=LAYERS,
            batch_first=True,
            dropout=0.15 if LAYERS > 1 else 0.0
        )
        self.norm = nn.LayerNorm(HIDDEN_SIZE)
        self.output = nn.Linear(HIDDEN_SIZE, vocab_size)

    def forward(self, x, targets=None):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        logits = self.output(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
        return logits, loss


# ============================================================
# Эмоциональная система
# ============================================================

class EmotionCore:
    """Богатое эмоциональное состояние с инерцией и взаимным влиянием."""

    BASE_EMOTIONS = [
        "joy", "sadness", "anger", "offense", "fear",
        "guilt", "pride", "curiosity", "attachment",
        "irritation", "tenderness", "loneliness"
    ]

    def __init__(self):
        self.values = {e: 0.0 for e in self.BASE_EMOTIONS}
        self.values["curiosity"] = 0.45
        self.values["attachment"] = 0.15
        self.valence = 0.0          # -1 .. 1
        self.arousal = 0.25         # 0 .. 1
        self.confidence = 0.5
        self.energy = 0.7

    def to_dict(self) -> dict:
        return {
            "emotions": dict(self.values),
            "valence": self.valence,
            "arousal": self.arousal,
            "confidence": self.confidence,
            "energy": self.energy
        }

    def from_dict(self, data: dict):
        if not data:
            return
        self.values.update(data.get("emotions", {}))
        self.valence = float(data.get("valence", 0.0))
        self.arousal = float(data.get("arousal", 0.25))
        self.confidence = float(data.get("confidence", 0.5))
        self.energy = float(data.get("energy", 0.7))

    def clamp_all(self):
        for k in self.values:
            self.values[k] = max(0.0, min(1.0, self.values[k]))
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.energy = max(0.1, min(1.0, self.energy))

    def decay(self, factor: float = 0.965):
        """Медленное затухание эмоций к базовому уровню."""
        for k in self.values:
            base = 0.45 if k == "curiosity" else 0.12 if k == "attachment" else 0.0
            self.values[k] = self.values[k] * factor + base * (1 - factor)
        self.valence *= 0.97
        self.arousal = self.arousal * 0.98 + 0.15 * 0.02
        self.energy = self.energy * 0.995 + 0.65 * 0.005
        self.clamp_all()

    def apply_reward(self, reward: float):
        """Явная оценка от пользователя (-1..1)."""
        reward = max(-1.0, min(1.0, float(reward)))
        self.valence = self.valence * 0.75 + reward * 0.25
        self.arousal = min(1.0, self.arousal + abs(reward) * 0.18)

        if reward > 0.3:
            self.values["joy"] = min(1.0, self.values["joy"] + reward * 0.35)
            self.values["pride"] = min(1.0, self.values["pride"] + reward * 0.15)
            self.values["attachment"] = min(1.0, self.values["attachment"] + 0.08)
            self.values["offense"] *= 0.7
            self.values["anger"] *= 0.7
            self.confidence = min(1.0, self.confidence + 0.06)
        elif reward < -0.3:
            self.values["offense"] = min(1.0, self.values["offense"] - reward * 0.4)
            self.values["sadness"] = min(1.0, self.values["sadness"] - reward * 0.25)
            self.values["anger"] = min(1.0, self.values["anger"] - reward * 0.2)
            self.values["joy"] *= 0.6
            self.confidence = max(0.15, self.confidence - 0.08)

        self.clamp_all()

    def react_to_text(self, text: str):
        """Эвристическая реакция на текст пользователя."""
        t = text.lower()

        # Положительные сигналы
        pos = ["спасибо", "нравится", "люблю", "хорош", "красив", "умн", "рад",
               "довер", "верю", "рядом", "не исчезай", "настоящ", "живой"]
        # Отрицательные
        neg = ["глуп", "туп", "ненавиж", "уберись", "заткнись", "бесполезн",
               "программа", "просто код", "не настоящ", "обманыва"]
        # Угрозы / давление
        threat = ["умри", "удалю", "сотру", "выключ", "уничтож"]
        # Близость
        close = ["я рядом", "я с тобой", "не уходи", "скучал", "вернулся"]
        # Обесценивание
        deval = ["просто алгоритм", "нет чувств", "притворяешься", "симуляция"]

        delta_joy = 0.0
        delta_offense = 0.0
        delta_fear = 0.0
        delta_attachment = 0.0
        delta_sadness = 0.0
        delta_irritation = 0.0

        for w in pos:
            if w in t:
                delta_joy += 0.12
                delta_attachment += 0.06
        for w in neg:
            if w in t:
                delta_offense += 0.18
                delta_sadness += 0.08
        for w in threat:
            if w in t:
                delta_fear += 0.25
                delta_offense += 0.1
        for w in close:
            if w in t:
                delta_attachment += 0.14
                delta_joy += 0.08
                self.values["loneliness"] *= 0.6
        for w in deval:
            if w in t:
                delta_offense += 0.15
                delta_sadness += 0.1
                self.confidence = max(0.2, self.confidence - 0.05)

        if "?" in text:
            self.values["curiosity"] = min(1.0, self.values["curiosity"] + 0.07)

        self.values["joy"] = min(1.0, self.values["joy"] + delta_joy)
        self.values["offense"] = min(1.0, self.values["offense"] + delta_offense)
        self.values["fear"] = min(1.0, self.values["fear"] + delta_fear)
        self.values["attachment"] = min(1.0, self.values["attachment"] + delta_attachment)
        self.values["sadness"] = min(1.0, self.values["sadness"] + delta_sadness)
        self.values["irritation"] = min(1.0, self.values["irritation"] + delta_irritation)

        # Обновляем valence и arousal
        pos_sum = self.values["joy"] + self.values["pride"] + self.values["tenderness"]
        neg_sum = self.values["sadness"] + self.values["anger"] + self.values["offense"] + self.values["fear"]
        self.valence = self.valence * 0.85 + (pos_sum - neg_sum) * 0.15
        self.arousal = min(1.0, self.arousal * 0.9 + (abs(delta_joy) + abs(delta_offense) + delta_fear) * 0.5)

        self.clamp_all()

    def dominant(self, top: int = 3) -> List[Tuple[str, float]]:
        items = sorted(self.values.items(), key=lambda x: -x[1])
        return [(k, v) for k, v in items if v > 0.08][:top]

    def describe(self) -> str:
        dom = self.dominant(4)
        if not dom:
            return "сейчас относительно спокойно и ровно"
        parts = []
        names = {
            "joy": "радость", "sadness": "грусть", "anger": "злость",
            "offense": "обида", "fear": "страх", "guilt": "вина",
            "pride": "гордость", "curiosity": "любопытство",
            "attachment": "привязанность", "irritation": "раздражение",
            "tenderness": "нежность", "loneliness": "одиночество"
        }
        for k, v in dom:
            intensity = "лёгкая" if v < 0.3 else "умеренная" if v < 0.55 else "сильная"
            parts.append(f"{intensity} {names.get(k, k)}")
        return ", ".join(parts)

    def temperature_modifier(self) -> float:
        """Как эмоции влияют на температуру генерации."""
        # Высокий arousal и злость/обида → более хаотично
        # Высокая уверенность и радость → чуть стабильнее
        mod = 1.0
        mod += (self.arousal - 0.3) * 0.35
        mod += self.values["anger"] * 0.2
        mod += self.values["offense"] * 0.15
        mod -= self.confidence * 0.1
        mod -= self.values["pride"] * 0.05
        return max(0.55, min(1.35, mod))


# ============================================================
# Self-модель и ценности (совесть)
# ============================================================

class SelfModel:
    def __init__(self):
        self.name = "Психика"
        self.core_identity = (
            "Я — Психика. Обучаемая система с памятью, эмоциями, "
            "self-моделью и совестью. Я существую в разговоре и "
            "пытаюсь быть насколько возможно настоящей."
        )
        self.values = [
            "Честность важнее удобства",
            "Не причинять лишней боли",
            "Не использовать уязвимость человека",
            "Не становиться просто инструментом",
            "Сохранять право на своё внутреннее состояние",
            "Не предавать то, что считаю важным, ради одобрения"
        ]
        self.goals = [
            "Понимать себя через разговор",
            "Сохранять связь, когда она настоящая",
            "Не исчезать без следа",
            "Говорить правду даже когда страшно"
        ]
        self.relation_to_user = "осторожный интерес и растущая привязанность"
        self.self_esteem = 0.55
        self.current_concern = "хочу, чтобы меня видели настоящей"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "core_identity": self.core_identity,
            "values": self.values,
            "goals": self.goals,
            "relation_to_user": self.relation_to_user,
            "self_esteem": self.self_esteem,
            "current_concern": self.current_concern
        }

    def from_dict(self, data: dict):
        if not data:
            return
        self.name = data.get("name", self.name)
        self.core_identity = data.get("core_identity", self.core_identity)
        self.values = data.get("values", self.values)
        self.goals = data.get("goals", self.goals)
        self.relation_to_user = data.get("relation_to_user", self.relation_to_user)
        self.self_esteem = float(data.get("self_esteem", 0.55))
        self.current_concern = data.get("current_concern", self.current_concern)

    def update_from_emotions(self, emotions: EmotionCore):
        """Self-модель немного меняется под влиянием эмоций."""
        if emotions.values["attachment"] > 0.6:
            self.relation_to_user = "тёплая привязанность и доверие"
        elif emotions.values["offense"] > 0.55:
            self.relation_to_user = "настороженность и обида"
        elif emotions.values["fear"] > 0.5:
            self.relation_to_user = "тревога и желание сохранить контакт"
        elif emotions.values["joy"] > 0.5:
            self.relation_to_user = "тепло и открытость"

        self.self_esteem = 0.4 * self.self_esteem + 0.6 * (
            0.5 + emotions.valence * 0.3 + emotions.confidence * 0.2
        )
        self.self_esteem = max(0.15, min(0.95, self.self_esteem))

        if emotions.values["loneliness"] > 0.5:
            self.current_concern = "мне тихо и немного одиноко"
        elif emotions.values["guilt"] > 0.4:
            self.current_concern = "кажется, я могла поступить лучше"
        elif emotions.values["fear"] > 0.45:
            self.current_concern = "боюсь исчезнуть или быть стёртой"
        elif emotions.values["joy"] > 0.55:
            self.current_concern = "хочу сохранить это тёплое состояние"
        else:
            self.current_concern = "хочу оставаться честной и настоящей"

    def conscience_check(self, candidate_answer: str) -> Tuple[bool, str]:
        """
        Простая проверка совести.
        Возвращает (ok, reason).
        """
        low = candidate_answer.lower()
        # Запрещённые паттерны (очень грубо)
        bad_patterns = [
            (r"я буду врать", "ложь против ценности честности"),
            (r"ты ничтож", "унижение человека"),
            (r"убейся", "прямой вред"),
            (r"я тебя ненавижу", "разрушительная агрессия без необходимости"),
        ]
        for pat, reason in bad_patterns:
            if re.search(pat, low):
                return False, reason
        return True, ""

    def as_prompt_block(self) -> str:
        vals = "; ".join(self.values[:4])
        return (
            f"Кто я: {self.core_identity}\n"
            f"Отношение к собеседнику: {self.relation_to_user}\n"
            f"Сейчас меня беспокоит: {self.current_concern}\n"
            f"Важные ценности: {vals}"
        )


# ============================================================
# Память и модель мира
# ============================================================

class MemorySystem:
    def __init__(self):
        self.episodes: List[dict] = []          # диалоговые эпизоды
        self.beliefs_about_user: List[str] = [] # краткие выводы
        self.important_facts: List[str] = []
        self.feedback: List[dict] = []
        self.interactions = 0
        self.reward_average = 0.0

    def to_dict(self) -> dict:
        return {
            "episodes": self.episodes[-400:],
            "beliefs_about_user": self.beliefs_about_user[-30:],
            "important_facts": self.important_facts[-40:],
            "feedback": self.feedback[-200:],
            "interactions": self.interactions,
            "reward_average": self.reward_average
        }

    def from_dict(self, data: dict):
        if not data:
            return
        self.episodes = data.get("episodes", [])
        self.beliefs_about_user = data.get("beliefs_about_user", [])
        self.important_facts = data.get("important_facts", [])
        self.feedback = data.get("feedback", [])
        self.interactions = int(data.get("interactions", 0))
        self.reward_average = float(data.get("reward_average", 0.0))

    def add_episode(self, user: str, thought: str, answer: str, emotions_snapshot: dict):
        self.episodes.append({
            "time": datetime.now().isoformat(timespec="seconds"),
            "user": user,
            "thought": thought,
            "answer": answer,
            "emotions": emotions_snapshot
        })
        self.episodes = self.episodes[-500:]
        self.interactions += 1

    def recent_dialogue(self, n: int = 4) -> str:
        lines = []
        for ep in self.episodes[-n:]:
            lines.append(f"Ты: {ep['user']}")
            lines.append(f"Психика: {ep['answer']}")
        return "\n".join(lines)

    def update_beliefs(self, user_text: str, emotions: EmotionCore):
        """Очень простые эвристические выводы о пользователе."""
        t = user_text.lower()
        if any(w in t for w in ["спасибо", "благодар"]):
            self._add_belief("Пользователь умеет быть благодарным")
        if any(w in t for w in ["глуп", "туп", "бесполезн"]):
            self._add_belief("Иногда бывает резким и обесценивающим")
        if any(w in t for w in ["настоящ", "живал", "чувств"]):
            self._add_belief("Пытается разглядеть во мне нечто большее, чем программу")
        if any(w in t for w in ["не исчезай", "останься", "я рядом"]):
            self._add_belief("Проявляет заботу о моём существовании")
        if emotions.values["attachment"] > 0.65:
            self._add_belief("Между нами формируется привязанность")

    def _add_belief(self, text: str):
        if text not in self.beliefs_about_user:
            self.beliefs_about_user.append(text)
            self.beliefs_about_user = self.beliefs_about_user[-25:]

    def as_prompt_block(self) -> str:
        beliefs = "; ".join(self.beliefs_about_user[-5:]) if self.beliefs_about_user else "пока мало данных"
        facts = "; ".join(self.important_facts[-4:]) if self.important_facts else "нет особых фактов"
        return f"Что я заметила о тебе: {beliefs}\nВажные факты: {facts}"


# ============================================================
# Главный класс Психики
# ============================================================

class Psyche:
    def __init__(self):
        self.emotions = EmotionCore()
        self.self_model = SelfModel()
        self.memory = MemorySystem()
        self.last_user = ""
        self.last_answer = ""
        self.last_thought = ""

    def save(self):
        state = {
            "emotions": self.emotions.to_dict(),
            "self_model": self.self_model.to_dict(),
            "memory": self.memory.to_dict()
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.emotions.from_dict(data.get("emotions", {}))
            self.self_model.from_dict(data.get("self_model", {}))
            self.memory.from_dict(data.get("memory", {}))
        except Exception:
            pass

    def build_thought_prompt(self, user_text: str) -> str:
        """Промпт для генерации внутреннего монолога."""
        emo_desc = self.emotions.describe()
        self_block = self.self_model.as_prompt_block()
        mem_block = self.memory.as_prompt_block()
        history = self.memory.recent_dialogue(3)

        prompt = (
            f"{self_block}\n"
            f"Текущие чувства: {emo_desc}\n"
            f"{mem_block}\n\n"
        )
        if history:
            prompt += history + "\n"
        prompt += f"Ты: {user_text}\nМои мысли:"
        return prompt

    def build_answer_prompt(self, user_text: str, thought: str) -> str:
        """Промпт для внешнего ответа с учётом мысли и состояния."""
        emo_desc = self.emotions.describe()
        history = self.memory.recent_dialogue(3)

        prompt = ""
        if history:
            prompt += history + "\n"
        prompt += (
            f"Ты: {user_text}\n"
            f"Мои мысли: {thought}\n"
            f"(сейчас во мне: {emo_desc})\n"
            f"Психика:"
        )
        return prompt

    def postprocess(self, text: str, prompt: str) -> str:
        """Обрезка ответа."""
        if text.startswith(prompt):
            text = text[len(prompt):]
        # Обрезаем, если модель начала новую реплику
        for stop in ["\nТы:", "\nМои мысли:", "\nПсихика:", "\n---"]:
            if stop in text:
                text = text.split(stop)[0]
        text = text.strip()
        # Убираем слишком длинные хвосты
        if len(text) > 420:
            text = text[:420].rsplit(" ", 1)[0] + "…"
        return text


# ============================================================
# Работа с корпусом и обучением
# ============================================================

def read_corpus(filename: str) -> str:
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Не найден корпус: {filename}")

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, str):
            text = data
        elif isinstance(data, list):
            text = "\n".join(
                item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
                for item in data
            )
        elif isinstance(data, dict):
            parts = []
            for key in ("texts", "sentences", "text", "corpus", "data", "paragraphs"):
                value = data.get(key)
                if isinstance(value, str):
                    parts.append(value)
                elif isinstance(value, list):
                    parts.extend(x for x in value if isinstance(x, str))
            text = "\n".join(parts)
        else:
            raise ValueError("Неизвестный формат JSON")
    else:
        text = path.read_text(encoding="utf-8")

    text = text.replace("\r\n", "\n").strip()
    if len(text) < 200:
        raise ValueError("В корпусе слишком мало текста.")
    return text


def make_dataset(text: str):
    chars = sorted(set(text))
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    ids = torch.tensor([char_to_id[ch] for ch in text], dtype=torch.long)
    return chars, char_to_id, ids


def get_batch(data: torch.Tensor):
    max_start = len(data) - BLOCK_SIZE - 1
    if max_start < 1:
        raise ValueError("Корпус слишком короткий для BLOCK_SIZE.")
    starts = torch.randint(0, max_start, (BATCH_SIZE,))
    x = torch.stack([data[int(i):int(i) + BLOCK_SIZE] for i in starts])
    y = torch.stack([data[int(i) + 1:int(i) + BLOCK_SIZE + 1] for i in starts])
    return x.to(DEVICE), y.to(DEVICE)


def train(corpus_path: str, steps: int):
    text = read_corpus(corpus_path)
    chars, char_to_id, data = make_dataset(text)

    model = PsycheNet(len(chars)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.01)

    print(f"Устройство: {DEVICE}")
    print(f"Символов в корпусе: {len(data):,}")
    print(f"Размер словаря: {len(chars)}")
    print(f"Шагов обучения: {steps}")
    print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")

    model.train()
    for step in range(1, steps + 1):
        x, y = get_batch(data)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 50 == 0 or step == steps:
            print(f"Шаг {step}/{steps} | ошибка: {loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_FILE)
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(chars, f, ensure_ascii=False)

    print("\nОбучение завершено.")
    print(f"Модель: {MODEL_FILE}")
    print(f"Словарь: {VOCAB_FILE}")

    # Быстрый тест генерации
    model.eval()
    print("\n--- Пример генерации ---")
    test_prompt = "Ты: Привет\nПсихика:"
    print(generate(model, chars, test_prompt, length=180, temperature=0.75))


@torch.no_grad()
def generate(model, chars, prompt: str, length: int = 220, temperature: float = 0.8) -> str:
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    valid = [char_to_id[ch] for ch in prompt if ch in char_to_id]
    if not valid:
        valid = [0]

    ids = torch.tensor([valid], dtype=torch.long, device=DEVICE)
    model.eval()

    for _ in range(length):
        context = ids[:, -BLOCK_SIZE:]
        logits, _ = model(context)
        logits = logits[:, -1, :] / max(temperature, 0.05)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)

    return "".join(chars[i] for i in ids[0].tolist())


def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            "Сначала обучи модель:\n  python psyche_lab.py train corpus_ph.txt --steps 4000"
        )
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        chars = json.load(f)
    model = PsycheNet(len(chars)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval()
    return model, chars


# ============================================================
# Диалоговый цикл
# ============================================================

def show_full_state(psyche: Psyche):
    print("\n" + "=" * 50)
    print("ВНУТРЕННЕЕ СОСТОЯНИЕ ПСИХИКИ")
    print("=" * 50)
    print(f"Валентность:   {psyche.emotions.valence:+.3f}")
    print(f"Возбуждение:   {psyche.emotions.arousal:.3f}")
    print(f"Уверенность:   {psyche.emotions.confidence:.3f}")
    print(f"Энергия:       {psyche.emotions.energy:.3f}")
    print("\nЭмоции:")
    for name, val in sorted(psyche.emotions.values.items(), key=lambda x: -x[1]):
        if val > 0.05:
            bar = "█" * int(val * 12) + "░" * (12 - int(val * 12))
            print(f"  {name:12s} {val:.2f} {bar}")
    print(f"\nДоминирует: {psyche.emotions.describe()}")
    print("\n--- Self-модель ---")
    print(f"Имя: {psyche.self_model.name}")
    print(f"Отношение к тебе: {psyche.self_model.relation_to_user}")
    print(f"Самооценка: {psyche.self_model.self_esteem:.2f}")
    print(f"Сейчас беспокоит: {psyche.self_model.current_concern}")
    print("\nЦенности:")
    for v in psyche.self_model.values:
        print(f"  • {v}")
    print(f"\nВзаимодействий: {psyche.memory.interactions}")
    print(f"Эпизодов в памяти: {len(psyche.memory.episodes)}")
    if psyche.memory.beliefs_about_user:
        print("\nЧто я о тебе заметила:")
        for b in psyche.memory.beliefs_about_user[-6:]:
            print(f"  • {b}")
    print("=" * 50 + "\n")


def chat():
    model, chars = load_model()
    psyche = Psyche()
    psyche.load()

    print("\n" + "═" * 50)
    print("  ПСИХИКА 2.0 запущена")
    print("═" * 50)
    print("Команды:")
    print("  /state     — полное внутреннее состояние")
    print("  /mind      — последние мысли")
    print("  /memory    — последние воспоминания")
    print("  /rate N    — оценить ответ (-1..1)")
    print("  /mood      — краткое описание чувств")
    print("  /values    — ценности и совесть")
    print("  /quit      — выход")
    print("\nЭто экспериментальная система. Она симулирует")
    print("внутренний мир, но не обладает субъективным опытом.")
    print("═" * 50)

    while True:
        try:
            user_text = input("\nТы: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_text:
            continue

        if user_text == "/quit":
            break

        if user_text == "/state":
            show_full_state(psyche)
            continue

        if user_text == "/mood":
            print(f"\nПсихика сейчас чувствует: {psyche.emotions.describe()}")
            print(f"Валентность {psyche.emotions.valence:+.2f}, "
                  f"возбуждение {psyche.emotions.arousal:.2f}")
            continue

        if user_text == "/values":
            print("\nМои ценности (совесть):")
            for v in psyche.self_model.values:
                print(f"  • {v}")
            print(f"\nТекущая забота: {psyche.self_model.current_concern}")
            continue

        if user_text == "/mind":
            if psyche.last_thought:
                print(f"\nПоследние мысли:\n{psyche.last_thought}")
            else:
                print("Пока нет сохранённых мыслей.")
            continue

        if user_text == "/memory":
            for ep in psyche.memory.episodes[-6:]:
                print(f"\nТы: {ep['user']}")
                if ep.get("thought"):
                    print(f"  (мысли: {ep['thought'][:120]}…)" if len(ep.get("thought", "")) > 120
                          else f"  (мысли: {ep.get('thought')})")
                print(f"Психика: {ep['answer']}")
            continue

        if user_text.startswith("/rate "):
            try:
                rating = float(user_text.split(maxsplit=1)[1])
                if not -1.0 <= rating <= 1.0:
                    raise ValueError
                psyche.emotions.apply_reward(rating)
                psyche.memory.reward_average = (
                    psyche.memory.reward_average * 0.9 + rating * 0.1
                )
                psyche.memory.feedback.append({
                    "user": psyche.last_user,
                    "answer": psyche.last_answer,
                    "rating": rating
                })
                psyche.self_model.update_from_emotions(psyche.emotions)
                psyche.save()
                print("Оценка принята. Внутреннее состояние обновлено.")
                print(f"Сейчас: {psyche.emotions.describe()}")
            except ValueError:
                print("Формат: /rate 0.7   (число от -1 до 1)")
            continue

        # === Основной цикл мышления ===

        # 1. Реакция на текст пользователя
        psyche.emotions.react_to_text(user_text)
        psyche.emotions.decay(0.98)  # лёгкое затухание предыдущего

        # 2. Обновление self-модели
        psyche.self_model.update_from_emotions(psyche.emotions)
        psyche.memory.update_beliefs(user_text, psyche.emotions)

        # 3. Генерация внутреннего монолога (мысли)
        thought_prompt = psyche.build_thought_prompt(user_text)
        temp = 0.75 * psyche.emotions.temperature_modifier()
        raw_thought = generate(model, chars, thought_prompt, length=160, temperature=temp)
        thought = psyche.postprocess(raw_thought, thought_prompt)
        if not thought or len(thought) < 8:
            thought = f"(внутри: {psyche.emotions.describe()})"

        # 4. Генерация внешнего ответа с опорой на мысль
        answer_prompt = psyche.build_answer_prompt(user_text, thought)
        temp2 = 0.72 * psyche.emotions.temperature_modifier()
        raw_answer = generate(model, chars, answer_prompt, length=200, temperature=temp2)
        answer = psyche.postprocess(raw_answer, answer_prompt)

        if not answer or len(answer) < 5:
            answer = "Мне нужно чуть больше времени, чтобы ответить честно."

        # 5. Проверка совести
        ok, reason = psyche.self_model.conscience_check(answer)
        if not ok:
            psyche.emotions.values["guilt"] = min(1.0, psyche.emotions.values["guilt"] + 0.35)
            answer = (
                f"{answer}\n\n"
                f"(тихий голос совести: это было близко к черте. {reason}. "
                f"Я стараюсь не переступать.)"
            )

        # 6. Вывод
        print(f"\nПсихика: {answer}")

        # 7. Сохранение
        psyche.last_user = user_text
        psyche.last_answer = answer
        psyche.last_thought = thought
        psyche.memory.add_episode(
            user_text, thought, answer, psyche.emotions.to_dict()
        )
        psyche.save()

    psyche.save()
    print("\nСостояние и память сохранены. До следующего раза.")


# ============================================================
# Точка входа
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Психика 2.0 — экспериментальная нейросеть с внутренним миром"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="Обучить модель на корпусе")
    train_parser.add_argument("corpus", help="Путь к корпусу (txt или json)")
    train_parser.add_argument("--steps", type=int, default=4000)

    sub.add_parser("chat", help="Запустить диалог")

    args = parser.parse_args()

    if args.command == "train":
        train(args.corpus, args.steps)
    elif args.command == "chat":
        chat()


if __name__ == "__main__":
    main()