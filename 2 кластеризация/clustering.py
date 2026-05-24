"""
Задание 2. Кластеризация K-Means по датасету DASS-42.

Скрипт убирает метку класса, созданную по суммарному DASS-баллу, и запускает
K-Means только на признаках. После этого найденные кластеры сравниваются
с реальными классами дистресса.

На выходе создаются:
- figures/cluster_pca.png               — визуализация кластеров в PCA-проекции;
- figures/cluster_distribution.png      — распределение объектов по кластерам;
- figures/cluster_vs_real.png           — сравнение кластеров с реальными классами;
- figures/elbow_silhouette.png          — подбор числа кластеров;
- clustering_metrics.txt                — краткий отчет по кластеризации;
- clustering_table.csv                  — таблица соответствия классов и кластеров;
- ../models/kmeans_model.pkl            — сохраненная модель K-Means.
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

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
CLASS_NAMES = ["Норма", "Легкий", "Умеренный", "Тяжелый", "Крайне тяжелый"]


def find_data_file() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "data" / "dass_dataset.csv",
        script_dir.parent / "data" / "dass_dataset.csv",
        script_dir / "материалы" / "data" / "dass_dataset.csv",
        script_dir.parent / "2 кластеризация" / "материалы" / "data" / "dass_dataset.csv",
        Path.cwd() / "data" / "dass_dataset.csv",
        Path.cwd() / "dass_dataset.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Не найден файл dass_dataset.csv. Поместите его в папку data рядом со скриптом.")


def read_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if df.shape[1] == 1:
        df = pd.read_csv(path, low_memory=False)
    return df


def calculate_dass_scores(df: pd.DataFrame) -> pd.DataFrame:
    depression_items = [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42]
    anxiety_items = [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41]
    stress_items = [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]

    def scale_sum(items):
        total = pd.Series(0, index=df.index, dtype="float64")
        for item in items:
            col = f"Q{item}A"
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce").fillna(1) - 1
                total += values.clip(0, 3)
        return total * 2

    scores = pd.DataFrame(index=df.index)
    scores["depression_score"] = scale_sum(depression_items)
    scores["anxiety_score"] = scale_sum(anxiety_items)
    scores["stress_score"] = scale_sum(stress_items)
    scores["total_score"] = scores.sum(axis=1)
    return scores


def make_target(total_score: pd.Series) -> pd.Series:
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
    features = pd.DataFrame(index=df.index)

    # В признаки добавляем сами ответы и дополнительные характеристики.
    # Метку класса сюда не добавляем — это важно для честной кластеризации.
    for i in range(1, 43):
        col = f"Q{i}A"
        if col in df.columns:
            features[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["introelapse", "testelapse", "surveyelapse", "age", "gender", "education", "urban", "engnat", "hand", "religion", "orientation", "race", "voted", "married", "familysize"]:
        if col in df.columns:
            features[col] = pd.to_numeric(df[col], errors="coerce")

    features = pd.concat([features, calculate_tipi_features(df.copy())], axis=1)

    if "age" in features.columns:
        features.loc[(features["age"] < 13) | (features["age"] > 100), "age"] = np.nan

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True))
    return features


def purity_score(y_true: np.ndarray, clusters: np.ndarray) -> float:
    """Считает долю объектов, которые попали в доминирующий реальный класс своего кластера."""
    table = pd.crosstab(clusters, y_true)
    return table.max(axis=1).sum() / table.values.sum()


def plot_elbow(X_scaled: np.ndarray, out_dir: Path) -> tuple[int, list[float], list[float]]:
    k_values = list(range(2, 8))
    inertias = []
    silhouettes = []

    # Silhouette на полной выборке дорогой, поэтому берем фиксированную подвыборку.
    rng = np.random.default_rng(RANDOM_STATE)
    if len(X_scaled) > 1000:
        idx = rng.choice(len(X_scaled), size=1000, replace=False)
        X_for_sil = X_scaled[idx]
    else:
        X_for_sil = X_scaled

    for k in k_values:
        model = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=1, batch_size=4096, max_iter=50)
        labels_for_sil = model.fit_predict(X_for_sil)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(X_for_sil, labels_for_sil))

    best_k = k_values[int(np.argmax(silhouettes))]

    plt.figure(figsize=(9, 5))
    plt.plot(k_values, inertias, marker="o", label="Inertia")
    plt.xlabel("Количество кластеров k")
    plt.ylabel("Inertia")
    plt.title("Метод локтя для K-Means")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "elbow.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(k_values, silhouettes, marker="o", label="Silhouette")
    plt.xlabel("Количество кластеров k")
    plt.ylabel("Silhouette score")
    plt.title("Подбор числа кластеров по silhouette score")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "elbow_silhouette.png", dpi=300)
    plt.close()

    return best_k, inertias, silhouettes


def plot_cluster_distribution(labels: np.ndarray, out_dir: Path) -> None:
    counts = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    bars = plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("Кластер")
    plt.ylabel("Количество объектов")
    plt.title("Распределение объектов по кластерам")
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 100, str(value), ha="center")
    plt.tight_layout()
    plt.savefig(out_dir / "cluster_distribution.png", dpi=300)
    plt.close()


def plot_pca(X_scaled: np.ndarray, labels: np.ndarray, out_dir: Path) -> None:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    # Чтобы картинка не была перегружена, рисуем не более 10000 точек.
    rng = np.random.default_rng(RANDOM_STATE)
    if len(X_pca) > 10000:
        idx = rng.choice(len(X_pca), size=10000, replace=False)
        X_plot = X_pca[idx]
        labels_plot = labels[idx]
    else:
        X_plot = X_pca
        labels_plot = labels

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=labels_plot, s=8, alpha=0.7)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.title("PCA-проекция кластеров K-Means")
    plt.colorbar(scatter, label="Кластер")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "cluster_pca.png", dpi=300)
    plt.close()


def plot_cluster_vs_real(y_true: pd.Series, labels: np.ndarray, out_dir: Path) -> pd.DataFrame:
    table = pd.crosstab(pd.Series(labels, name="Кластер"), pd.Series(y_true, name="Реальный класс"))
    table_named = table.rename(columns={i: CLASS_NAMES[i] for i in range(len(CLASS_NAMES))})

    plt.figure(figsize=(10, 6))
    plt.imshow(table_named.values, aspect="auto")
    plt.title("Сравнение найденных кластеров с реальными классами")
    plt.xlabel("Реальный класс")
    plt.ylabel("Кластер K-Means")
    plt.colorbar(label="Количество объектов")
    plt.xticks(np.arange(len(table_named.columns)), table_named.columns, rotation=35, ha="right")
    plt.yticks(np.arange(len(table_named.index)), table_named.index)

    threshold = table_named.values.max() / 2 if table_named.values.max() else 0
    for i in range(table_named.shape[0]):
        for j in range(table_named.shape[1]):
            value = table_named.values[i, j]
            plt.text(j, i, str(value), ha="center", va="center",
                     color="white" if value > threshold else "black")
    plt.tight_layout()
    plt.savefig(out_dir / "cluster_vs_real.png", dpi=300)
    plt.close()
    return table_named


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
    y_real = make_target(scores["total_score"])
    X = prepare_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, inertias, silhouettes = plot_elbow(X_scaled, out_dir)

    # Для сопоставления с 5 реальными уровнями используем k=5.
    # best_k сохраняем в отчете как результат автоматического подбора.
    k_for_comparison = 5
    kmeans = MiniBatchKMeans(n_clusters=k_for_comparison, random_state=RANDOM_STATE, n_init=5, batch_size=4096, max_iter=100)
    labels = kmeans.fit_predict(X_scaled)

    ari = adjusted_rand_score(y_real, labels)
    nmi = normalized_mutual_info_score(y_real, labels)
    purity = purity_score(y_real.to_numpy(), labels)

    plot_cluster_distribution(labels, out_dir)
    plot_pca(X_scaled, labels, out_dir)
    table = plot_cluster_vs_real(y_real, labels, out_dir)
    table.to_csv(script_dir / "clustering_table.csv", encoding="utf-8-sig")

    joblib.dump(kmeans, models_dir / "kmeans_model.pkl")
    joblib.dump(scaler, models_dir / "kmeans_scaler.pkl")

    with open(script_dir / "clustering_metrics.txt", "w", encoding="utf-8") as f:
        f.write("Результаты кластеризации K-Means по датасету DASS-42\n")
        f.write("=" * 60 + "\n")
        f.write(f"Всего объектов: {len(df)}\n")
        f.write(f"Количество признаков без метки: {X.shape[1]}\n")
        f.write(f"Лучшее k по silhouette: {best_k}\n")
        f.write(f"Для сравнения с реальными 5 классами использовано k: {k_for_comparison}\n\n")
        f.write("Метрики совпадения кластеров с реальными классами:\n")
        f.write(f"- Adjusted Rand Index: {ari:.4f}\n")
        f.write(f"- Normalized Mutual Information: {nmi:.4f}\n")
        f.write(f"- Purity: {purity:.4f}\n\n")
        f.write("Silhouette score по k:\n")
        for k, value in zip(range(2, 8), silhouettes):
            f.write(f"- k={k}: {value:.4f}\n")
        f.write("\nТаблица соответствия кластеров и реальных классов сохранена в clustering_table.csv\n")

    print("Готово. Файлы сохранены в папке 2 кластеризация и figures.")
    print({"best_k": best_k, "ARI": ari, "NMI": nmi, "Purity": purity})


if __name__ == "__main__":
    main()
