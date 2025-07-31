import os
import numpy as np
from utilities import run_comparison_experiment, get_drift_points, evaluate_detectors, calculate_metrics
from pathlib import Path
import pickle
from river import drift
from models.batch_vae_drift_detector import BatchVAEDriftDetector as VAEDriftDetector
import logging
import argparse


def main():
    argparser = argparse.ArgumentParser(description="Run drift detection experiment on selected streams.")
    argparser.add_argument(
        "stream_group",
        type=str,
    )
    args = argparser.parse_args()
    stream_group = args.stream_group

    # Ścieżka do katalogu z danymi
    data_dir = "new_streams"

    # Ścieżka do pliku z definicją strumieni
    stream_definition_file = "new_streams/streams.txt"

    # Ścieżka do folderu z wynikami eksperymentu
    results_path = Path("results") / stream_group

    # Lista strumieni do analizy
    with open(f'stream_groups/{stream_group}.txt', 'r') as f:
        streams_to_analyze = [line.strip() for line in f.readlines()]

    # Tworzenie katalogu na wyniki, jeśli nie istnieje
    results_path.mkdir(parents=True, exist_ok=True)

    existing_files = os.listdir(results_path)

    logger = logging.getLogger("drift_detection")
    logging.basicConfig(filename=results_path/"drift_detection.log", level=logging.INFO)

    logger.info(f"Zapis wyników do katalogu: {results_path}")

    # Znajdź wszystkie pliki .arff w katalogu
    stream_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".arff"):
                if file not in existing_files:
                    if file in streams_to_analyze:
                        stream_paths.append(os.path.join(root, file))

    logger.info(f"Znaleziono {len(stream_paths)} strumieni do analizy.")

    # Ograniczenie liczby strumieni do analizy
    sample_size = 300
    if len(stream_paths) > sample_size:
        stream_paths = np.random.choice(stream_paths, sample_size, replace=False)
        logger.info(f"Losowo wybrano {sample_size} strumieni do analizy:")
        for path in stream_paths:
            logger.info(f" - {os.path.basename(path)}")

    with open(stream_definition_file, "r") as f:
        stream_definition_file_content = f.read()

    streams = [os.path.basename(path) for path in stream_paths]

    drift_points = {}
    for stream in streams:
        drift_points[stream] = get_drift_points(stream_definition_file_content, stream)

    # Uruchom eksperyment
    input_dim = 5

    detectors = {
        "ADWIN": drift.ADWIN(delta=0.05, min_window_length=5000, clock=1000),
        "KSWIN": drift.KSWIN(alpha=0.05, window_size=5000, stat_size=1000),
        "PageHinkley": drift.PageHinkley(min_instances=5000),
        "VAE (latent_dim=2)": VAEDriftDetector(
            input_dim=input_dim,
            hidden_dim=32,
            latent_dim=2,
            p_threshold=0.05,
            epochs=50,
            retrain_on_drift=True,
            window_size=5000,
        ),
        "VAE (latent_dim=5)": VAEDriftDetector(
            input_dim=input_dim,
            hidden_dim=32,
            latent_dim=5,
            p_threshold=0.05,
            epochs=50,
            retrain_on_drift=True,
            window_size=5000,
        ),
    }

    predictions = run_comparison_experiment(detectors, stream_paths, drift_points, results_path)

    results_evaluated = evaluate_detectors(predictions, drift_points, delta=5000, bucket_size=5000)

    metrics = calculate_metrics(results_evaluated)

    with open(results_path / "metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    logger.info(f"Wyniki eksperymentu zapisano w {results_path / 'metrics.pkl'}")


if __name__ == "__main__":
    main()
