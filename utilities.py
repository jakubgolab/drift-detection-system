import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
import torch
import re
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import logging
import pickle


logger = logging.getLogger("drift_detection")


def load_arff_data(filepath):
    """
    Ładuje dane z pliku .arff i konwertuje je do formatu pandas DataFrame.
    """
    try:
        data, meta = arff.loadarff(filepath)
        df = pd.DataFrame(data)

        if b"class" in df.columns or "class" in df.columns:
            df.drop(columns=["class"], inplace=True, errors="ignore")

        feature_columns = [col for col in df.columns if col != "class"]
        df.rename(
            columns={col: f"feature_{i}" for i, col in enumerate(feature_columns)},
            inplace=True,
        )

        return df
    except Exception as e:
        logger.info(f"Błąd podczas wczytywania pliku ARFF: {str(e)}")
        raise


def experiment_on_stream(
    stream_path,
    detectors,
    result_folder,
    window_size=1000,
    drift_points=None,
    plot_results=True,
):
    """
    Przeprowadza eksperyment porównujący detektory dryfu na strumieniu danych.

    Parameters
    ----------
    stream_path : str
        Ścieżka do pliku ze strumieniem danych w formacie .arff
    detectors : dict
        Słownik z nazwami detektorów i ich instancjami
    window_size : int, default=300
        Rozmiar okna do obliczania średnich wyników
    drift_points : list, default=None
        Lista punktów, w których występują rzeczywiste dryfy (jeśli znane)
    plot_results : bool, default=True
        Czy generować wykresy z wynikami

    Returns
    -------
    dict
        Słownik zawierający wyniki dla każdego detektora
    """
    logger.info(f"Rozpoczynam eksperyment dla strumienia: {os.path.basename(stream_path)}")
    result_folder = Path(result_folder) / os.path.basename(stream_path).replace(
        ".arff", ""
    )

    df = load_arff_data(stream_path)

    results = {
        "detections": {name: [] for name in detectors},
        "detection_times": {name: [] for name in detectors},
        "reconstruction_errors": {name: [] for name in detectors if "VAE" in name},
    }

    error_windows = {name: [] for name in detectors if "VAE" in name}

    n_samples = len(df)

    logger.info(f"Liczba próbek: {n_samples}")
    logger.info(f"Liczba cech: {df.shape[1]}")

    start_time = time.time()

    for i in range(n_samples):
        # Pobierz aktualny obiekt
        current_features = df.iloc[i].values

        if i % 1000 == 0:
            logger.info(f"Przetwarzam próbkę {i+1}/{n_samples}")

        # Sprawdź każdy detektor
        for name, detector in detectors.items():
            detection_start = time.time()

            if "VAE" in name and hasattr(detector, "update"):
                # VAE detektor bezpośrednio
                detector.update(current_features)
                is_drift = detector.drift_detected()

                # Zapisz błędy rekonstrukcji jeśli dostępne
                if (
                    hasattr(detector, "test_data")
                    and len(detector.test_data) > 0
                    and detector.is_trained
                ):
                    with torch.no_grad():
                        last_sample = torch.FloatTensor([detector.test_data[-1]]).to(
                            detector.device
                        )
                        recon, mu, logvar = detector.model(last_sample)
                        recon_error = torch.sum((recon - last_sample) ** 2).item()
                        error_windows[name].append(recon_error)

                        # Utrzymuj okno o stałej wielkości
                        if len(error_windows[name]) > window_size:
                            error_windows[name].pop(0)

                        # Śledź średnie błędy
                        results["reconstruction_errors"][name].append(
                            np.mean(error_windows[name]) if error_windows[name] else 0
                        )
            else:
                # Standardowy detektor River
                scalar_value = np.linalg.norm(
                    current_features
                )  # Użycie normy Euklidesowej
                detector.update(scalar_value)
                is_drift = detector.drift_detected

            detection_time = time.time() - detection_start

            # Zapisz wyniki detekcji
            if is_drift:
                results["detections"][name].append(i)
            results["detection_times"][name].append(detection_time)

    total_time = time.time() - start_time
    logger.info(f"Czas wykonania eksperymentu: {total_time:.2f} sekund")

    # Podsumowanie wyników
    logger.info("\nWyniki detekcji dryfów:")
    for name in detectors:
        detections = results["detections"][name]
        logger.info(
            f"{name}: wykryte dryfy: {len(detections)}, "
            f"średni czas detekcji: {np.mean(results['detection_times'][name]):.6f} sekund"
        )
        if detections:
            logger.info(f"  Pozycje wykrytych dryfów: {detections}")

    # Generowanie wykresów
    if plot_results:
        plot_experiment_results(
            results,
            drift_points,
            result_folder,
            os.path.basename(stream_path),
            window_size,
        )

    return results


def plot_experiment_results(
    results, drift_points, result_folder, stream_name, window_size, samples=200000
):
    """
    Generuje wykresy z wynikami eksperymentu.
    """
    plt.figure(figsize=(15, 10))

    # 1. Wykres błędów rekonstrukcji dla detektorów VAE
    if "reconstruction_errors" in results and any(results["reconstruction_errors"]):
        plt.subplot(2, 1, 1)
        for name, errors in results["reconstruction_errors"].items():
            if errors:
                # plt.plot(errors, label=f"{name} (średnie błędy w oknie {window_size})")
                plt.plot(errors, label=f"{name} (mean window-block error {window_size})")

        # Dodaj zacieniowane obszary dla przedziałów dryfu
        if drift_points:
            y_min, y_max = plt.ylim()
            for start, end in drift_points:
                plt.axvspan(
                    start,
                    end,
                    alpha=0.2,
                    color="red",
                    label=(
                        f"Real drift {start}-{end}"
                        if (start, end) == drift_points[0]
                        else ""
                    ),
                )

                # Dodaj pionowe linie na początku i końcu przedziału dla lepszej widoczności
                plt.axvline(x=start, color="r", linestyle="--", alpha=0.5)
                plt.axvline(x=end, color="r", linestyle="--", alpha=0.5)

        plt.title(f"Reconstruction errors - {stream_name}")
        plt.xlabel("Sample")
        plt.ylabel("Mean reconstruction error")
        plt.legend()
        plt.grid(True, alpha=0.3)
        xlim_1 = plt.xlim()

    # 2. Wykres punktów wykrycia dryfu
    plt.subplot(2, 1, 2)
    detector_names = sorted(results["detections"].keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(detector_names)))

    for i, name in enumerate(detector_names):
        detections = results["detections"][name]
        if detections:
            plt.scatter(
                [d for d in detections],
                [i] * len(detections),
                color=colors[i],
                label=name,
                s=100,
                marker="o",
            )

    if drift_points:
        for start, end in drift_points:
            plt.axvspan(
                start,
                end,
                alpha=0.2,
                color="red",
                label=(
                    f"Drift interval {start}-{end}"
                    if (start, end) == drift_points[0]
                    else ""
                ),
            )

            # Dodaj pionowe linie na początku i końcu przedziału dla lepszej widoczności
            plt.axvline(x=start, color="r", linestyle="--", alpha=0.5)
            plt.axvline(x=end, color="r", linestyle="--", alpha=0.5)

    plt.yticks(range(len(detector_names)), detector_names)
    plt.xlim(xlim_1)
    plt.title(f"Detected drift points - {stream_name}")
    plt.xlabel("Sample")
    plt.ylabel("Detector")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    result_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Wyniki będą zapisywane w: {result_folder}")

    plt.savefig(result_folder / f"{stream_name}_drift_detection_results.png", dpi=300)

    logger.info(
        f"Wykresy zapisane w: {result_folder}"
    )


def run_comparison_experiment(detectors, stream_paths, drift_points, results_path):
    """
    Przeprowadza porównanie różnych detektorów dryfu na wielu strumieniach danych.

    Parameters
    ----------
    stream_paths : list
        Lista ścieżek do plików ze strumieniami danych
    drift_points : dict, default=None
        Słownik mapujący nazwy strumieni na listy punktów dryfu

    Returns
    -------
    dict
        Słownik zawierający wyniki dla każdego strumienia i detektora
    """
    if os.path.exists(results_path / 'results.pkl'):
        with open(results_path / 'results.pkl', 'rb') as f:
            all_results = pickle.load(f)
        logger.info("Wczytano wyniki z pliku.")
    else:
        # Inicjalizacja wyników
        all_results = {}

    # Przeprowadzenie eksperymentów na każdym strumieniu
    for idx, stream_path in enumerate(stream_paths):
        logger.info(f"\nRozpoczynam eksperyment dla strumienia numer {idx}/{len(stream_paths)}")

        stream_name = os.path.basename(stream_path)

        # Pobierz punkty dryfu dla danego strumienia (jeśli dostępne)
        stream_drift_points = (
            drift_points.get(stream_name, None) if drift_points else None
        )

        # Przeprowadź eksperyment
        results = experiment_on_stream(
            stream_path=stream_path,
            detectors=detectors,
            drift_points=stream_drift_points,
            plot_results=True,
            window_size=1000,
            result_folder=results_path,
        )

        all_results[stream_name] = results

        # Zapisz wyniki do pliku
        with open(results_path / 'results.pkl', 'wb') as f:
            pickle.dump(all_results, f)
        logger.info(f"Wyniki zapisane w: {results_path / 'results.pkl'}")

    return all_results


def evaluate_detectors(all_results, drift_points, delta=10000, bucket_size=10000, data_len=200000):
    """
    Ocenia detektory na podstawie wyników eksperymentów.

    Parameters
    ----------
    all_results : dict
        Słownik zawierający wyniki dla każdego strumienia i detektora
    drift_points : dict
        Słownik mapujący nazwy strumieni na listy punktów dryfu
    """

    evals = {}

    # Ewaluacja eksperymentu
    for stream_name, results in all_results.items():
        if stream_name not in evals:
            evals[stream_name] = {}
        for detector_name, detections in results["detections"].items():
            y_true, y_pred = evaluate_drift_detection(data_len, drift_points[stream_name],
                                                      delta,
                                                      bucket_size,
                                                      detections)
            evals[stream_name][detector_name] = {
                "y_true": y_true,
                "y_pred": y_pred,
                "detection_times": results["detection_times"][detector_name],
            }

    return evals


def are_windows_overlapping(start1, end1, start2, end2):
    return end1 > start2 and start1 < end2


def evaluate_drift_detection(data_len, drift_points, delta, bucket_size, detections):
    indices = np.arange(0, data_len, bucket_size)
    sections = []

    for i in range(len(indices)):
        start = indices[i]
        if i == len(indices) - 1:
            end = np.inf
        else:
            end = indices[i + 1]
        sections.append((start, end))

    drifts = {}

    for window in sections:
        drifts[window] = 0
        for drift_point in drift_points:
            if are_windows_overlapping(
                window[0], window[1], drift_point[0], drift_point[1] + delta
            ):
                drifts[window] = 1
                break

    detector_result = {}

    for window in sections:
        detector_result[window] = 0
        for detection_idx in detections:
            if window[0] <= detection_idx < window[1]:
                detector_result[window] = 1

    y_true = list(drifts.values())
    y_pred = list(detector_result.values())
    return y_true, y_pred


def get_drift_points(file_content, stream_name, min_threshold=10000):
    # Wzorzec do znalezienia bloku dotyczącego konkretnego strumienia
    stream_pattern = (
        r"#\s*\d+\s*\$\(STRM_DIR\)/" + re.escape(stream_name) + r":[^\']*\'([^\']*)\'"
    )

    # Szukamy bloku dla określonego strumienia
    stream_match = re.search(stream_pattern, file_content)

    if not stream_match:
        return []

    # Pobieramy część zawierającą definicje dryfów
    drift_definitions = stream_match.group(1)

    # Wzorzec do wyodrębnienia parametrów start i end
    drift_pattern = r"(?:start=(\d+),end=(\d+))"

    # Znajdujemy wszystkie dopasowania
    drift_points = re.findall(drift_pattern, drift_definitions)

    # Konwertujemy wyniki na listę krotek z liczbami całkowitymi
    # Filtrujemy tylko te punkty dryfu, które mają wartości >= min_threshold
    result = []
    for start, end in drift_points:
        start_int = int(start)
        end_int = int(end)

        # Pomijamy punkty dryfu z wartościami poniżej progu
        if start_int >= min_threshold or end_int >= min_threshold:
            result.append((start_int, end_int))

    return result


def calculate_metrics(results_evaluated):
    metrics = {}

    detectors = list(results_evaluated[list(results_evaluated.keys())[0]].keys())
    for detector in detectors:
        metrics[detector] = {
            "TP": [],
            "FP": [],
            "TN": [],
            "FN": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "mean_detection_time": [],
            "std_detection_time": [],
            "g_mean": [],
        }

    for stream_name, results in results_evaluated.items():
        for detector_name, single_result in results.items():
            y_true = single_result["y_true"]
            y_pred = single_result["y_pred"]
            detection_times = single_result["detection_times"]

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            g_mean = np.sqrt(precision * recall)
            mean_detection_time = np.mean(detection_times)
            std_detection_time = np.std(detection_times)
            metrics[detector_name]["TP"].append(tp)
            metrics[detector_name]["FP"].append(fp)
            metrics[detector_name]["TN"].append(tn)
            metrics[detector_name]["FN"].append(fn)
            metrics[detector_name]["accuracy"].append(accuracy)
            metrics[detector_name]["precision"].append(precision)
            metrics[detector_name]["recall"].append(recall)
            metrics[detector_name]["f1_score"].append(f1)
            metrics[detector_name]["g_mean"].append(g_mean)
            metrics[detector_name]["mean_detection_time"].append(mean_detection_time)
            metrics[detector_name]["std_detection_time"].append(std_detection_time)

    return metrics


def display_metrics(metrics):
    for detector_name, detector_metrics in metrics.items():
        logger.info(f"\nDetektor: {detector_name}")
        logger.info("=" * 50)
        logger.info(f"TP: {np.mean(detector_metrics['TP'])}")
        logger.info(f"FP: {np.mean(detector_metrics['FP'])}")
        logger.info(f"TN: {np.mean(detector_metrics['TN'])}")
        logger.info(f"FN: {np.mean(detector_metrics['FN'])}")
        logger.info(f"Accuracy: {np.mean(detector_metrics['accuracy'])}")
        logger.info(f"Precision: {np.mean(detector_metrics['precision'])}")
        logger.info(f"Recall: {np.mean(detector_metrics['recall'])}")
        logger.info(f"F1 Score: {np.mean(detector_metrics['f1_score'])}")
        logger.info(f"G-Mean: {np.mean(detector_metrics['g_mean'])}")
        logger.info(f"Mean Detection Time: {np.mean(detector_metrics['mean_detection_time'])}")
        logger.info(f"Std Detection Time: {np.std(detector_metrics['std_detection_time'])}")
