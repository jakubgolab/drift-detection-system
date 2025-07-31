from river import base


class VAEDriftAdaptivePredictor(base.Estimator):
    """Wrapper łączący dowolny estymator River z detektorem dryfu VAE.

    Parameters
    ----------
    estimator : river.base.Estimator
        Bazowy estymator River, który ma być adaptowany.
    detector : VAEDriftDetector
        Instancja detektora dryfu VAE.
    """

    def __init__(self, estimator, detector):
        self.estimator = estimator
        self.detector = detector
        self._is_classification = hasattr(estimator, "predict_proba")

    def learn_one(self, x, y=None, **kwargs):
        """Aktualizuje detektor dryfu i model."""
        # Aktualizuj detektor dryfu
        self.detector.update(list(x.values()))

        # Ucz model na bieżących danych
        self.estimator.learn_one(x, y, **kwargs)

        # Jeśli wykryto dryf i model obsługuje metodę reset, resetuj model
        if self.detector.drift_detected() and hasattr(self.estimator, "reset"):
            self.estimator.reset()

        return self

    def predict_one(self, x, **kwargs):
        """Wykonuje predykcję używając aktualnego modelu."""
        return self.estimator.predict_one(x, **kwargs)

    def predict_proba_one(self, x, **kwargs):
        """Wykonuje predykcję prawdopodobieństwa używając aktualnego modelu."""
        if self._is_classification:
            return self.estimator.predict_proba_one(x, **kwargs)
        raise NotImplementedError("Bazowy estymator nie implementuje predict_proba_one")
