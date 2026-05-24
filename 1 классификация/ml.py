"""
Задание 1. Многоклассовая классификация по датасету DASS-42.

Скрипт обучает модель, которая выдаёт не просто два состояния, а конкретный
класс психологического дистресса:
0 - Норма, 1 - Легкий, 2 - Умеренный, 3 - Тяжелый, 4 - Крайне тяжелый.

На выходе создаются:
- figures/accuracy_metrics.png          — график качества модели;
- figures/confusion_matrix.png          — матрица ошибок;
- figures/feature_importance.png        — важность признаков;
- classification_report.csv             — таблица precision/recall/f1-score;
- metrics.txt                           — краткий текстовый отчет;
- ../models/dass_multiclass_model.pkl   — сохраненная модель.
"""

import os
# Ограничиваем число потоков, чтобы sklearn стабильно работал на обычном ноутбуке.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
CLASS_NAMES = ["Норма", "Легкий", "Умеренный", "Тяжелый", "Крайне тяжелый"]


def find_data_file() -> Path:
    """Ищет датасет рядом с проектом, чтобы скрипт запускался из разных папок."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "data" / "dass_dataset.csv",
        script_dir.parent / "data" / "dass_dataset.csv",
        script_dir.parent / "2 кластеризация" / "материалы" / "data" / "dass_dataset.csv",
        script_dir.parent / "2 кластеризация" / "data" / "dass_dataset.csv",
        Path.cwd() / "data" / "dass_dataset.csv",
        Path.cwd() / "dass_dataset.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Не найден файл dass_dataset.csv. Поместите его в папку data рядом со скриптом.")


def read_dataset(path: Path) -> pd.DataFrame:
    """Читает CSV. В архиве встречается табуляция, поэтому используем автоопределение."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if df.shape[1] == 1:
        df = pd.read_csv(path, low_memory=False)
    return df


def calculate_dass_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Считает шкалы DASS-42 по официальной логике: ответы 1-4 переводятся в 0-3, сумма умножается на 2."""
    depression_items = [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42]
    anxiety_items = [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41]
    stress_items = [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]

    def scale_sum(items):
        total = pd.Series(0, index=df.index, dtype="float64")
        for item in items:
            col = f"Q{item}A"
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce").fillna(1) - 1
                values = values.clip(0, 3)
                total += values
        return total * 2

    scores = pd.DataFrame(index=df.index)
    scores["depression_score"] = scale_sum(depression_items)
    scores["anxiety_score"] = scale_sum(anxiety_items)
    scores["stress_score"] = scale_sum(stress_items)
    scores["total_score"] = scores.sum(axis=1)
    return scores


def make_target(total_score: pd.Series) -> pd.Series:
    """Создает 5 классов общего уровня дистресса."""
    def categorize(score: float) -> int:
        if score <= 30:
            return 0
        if score <= 43:
            return 1
        if score <= 58:
            return 2
        if score <= 80:
            return 3
        return 4

    return total_score.apply(categorize).astype(int)


def calculate_tipi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создает признаки Big Five по TIPI."""
    out = pd.DataFrame(index=df.index)
    for col in [f"TIPI{i}" for i in range(1, 11)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    pairs = {
        "tipi_extraversion": ("TIPI1", "TIPI6"),
        "tipi_agreeableness": ("TIPI7", "TIPI2"),
        "tipi_conscientiousness": ("TIPI3", "TIPI8"),
        "tipi_neuroticism": ("TIPI4", "TIPI9"),
        "tipi_openness": ("TIPI5", "TIPI10"),
    }
    for name, (direct, reverse) in pairs.items():
        if direct in df.columns and reverse in df.columns:
            out[name] = (df[direct] + (8 - df[reverse])) / 2
    return out


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Готовит числовые признаки для модели."""
    features = pd.DataFrame(index=df.index)

    # Ответы на 42 вопроса DASS — главные признаки.
    for i in range(1, 43):
        col = f"Q{i}A"
        if col in df.columns:
            features[col] = pd.to_numeric(df[col], errors="coerce")

    # Временные признаки прохождения теста.
    for col in ["introelapse", "testelapse", "surveyelapse"]:
        if col in df.columns:
            features[col] = pd.to_numeric(df[col], errors="coerce")

    # Демографические признаки.
    for col in ["age", "gender", "education", "urban", "engnat", "hand", "religion", "orientation", "race", "voted", "married", "familysize"]:
        if col in df.columns:
            features[col] = pd.to_numeric(df[col], errors="coerce")

    # Личностные признаки TIPI.
    tipi = calculate_tipi_features(df.copy())
    features = pd.concat([features, tipi], axis=1)

    # Чистка выбросов по возрасту.
    if "age" in features.columns:
        features.loc[(features["age"] < 13) | (features["age"] > 100), "age"] = np.nan

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True))
    return features


def plot_metrics(metrics: dict, out_dir: Path) -> None:
    names = list(metrics.keys())
    values = list(metrics.values())
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values)
    plt.ylim(0, 1.05)
    plt.ylabel("Значение")
    plt.title("Метрики качества многоклассовой классификации")
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_metrics.png", dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, out_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Матрица ошибок")
    plt.colorbar()
    ticks = np.arange(len(CLASS_NAMES))
    plt.xticks(ticks, CLASS_NAMES, rotation=35, ha="right")
    plt.yticks(ticks, CLASS_NAMES)
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")

    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > threshold else "black")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=300)
    plt.close()


def plot_feature_importance(model: Pipeline, feature_names: list[str], out_dir: Path) -> pd.DataFrame:
    rf = model.named_steps["model"]
    importance = pd.DataFrame({"feature": feature_names, "importance": rf.feature_importances_})
    importance = importance.sort_values("importance", ascending=False)

    top = importance.head(15).sort_values("importance")
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Важность признака")
    plt.title("Топ-15 признаков модели")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=300)
    plt.close()
    return importance


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "figures"
    models_dir = script_dir.parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    data_path = find_data_file()
    print(f"Датасет: {data_path}")
    df = read_dataset(data_path)
    scores = calculate_dass_scores(df)
    y = make_target(scores["total_score"])
    X = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=180,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_test, y_pred),
        "F1 macro": f1_score(y_test, y_pred, average="macro"),
        "F1 weighted": f1_score(y_test, y_pred, average="weighted"),
    }

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(script_dir / "classification_report.csv", encoding="utf-8-sig")

    plot_metrics(metrics, out_dir)
    plot_confusion_matrix(y_test, y_pred, out_dir)
    importance = plot_feature_importance(model, list(X.columns), out_dir)
    importance.to_csv(script_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    joblib.dump(model, models_dir / "dass_multiclass_model.pkl")

    class_distribution = y.value_counts().sort_index()
    with open(script_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write("Результаты многоклассовой классификации DASS-42\n")
        f.write("=" * 60 + "\n")
        f.write(f"Всего объектов: {len(df)}\n")
        f.write(f"Количество признаков: {X.shape[1]}\n")
        f.write("Классы: " + ", ".join(CLASS_NAMES) + "\n\n")
        f.write("Распределение классов:\n")
        for class_id, count in class_distribution.items():
            f.write(f"- {CLASS_NAMES[class_id]}: {count} ({count / len(y) * 100:.2f}%)\n")
        f.write("\nМетрики:\n")
        for name, value in metrics.items():
            f.write(f"- {name}: {value:.4f}\n")
        f.write("\nТоп-10 важных признаков:\n")
        for _, row in importance.head(10).iterrows():
            f.write(f"- {row['feature']}: {row['importance']:.5f}\n")

    print("Готово. Файлы сохранены в папке 1 классификация и figures.")
    print(metrics)


if __name__ == "__main__":
    main()
