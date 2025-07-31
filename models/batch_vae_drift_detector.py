from river import base
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats
from models.vae_model import VAEModel


class BatchVAEDriftDetector(base.DriftDetector):
    """Detektor dryfu koncepcji używający Wariacyjnego Autoenkodera z przetwarzaniem wsadowym.

    Implementacja przetwarza dane w oknach (batchach) i porównuje kolejne okna z oknem referencyjnym.
    Domyślnie model nie jest trenowany ponownie po wykryciu dryfu, zachowując oryginalny punkt odniesienia.

    Parameters
    ----------
    input_dim : int
        Liczba wymiarów danych wejściowych.
    window_size : int, default=1000
        Rozmiar okna dla przetwarzania wsadowego.
    retrain_on_drift : bool, default=False
        Czy trenować model ponownie po wykryciu dryfu (True) czy zachować oryginalny model (False).
    hidden_dim : int, default=32
        Liczba jednostek w warstwach ukrytych.
    latent_dim : int, default=2
        Liczba wymiarów przestrzeni ukrytej.
    p_threshold : float, default=0.05
        Próg p-wartości dla testu Kołmogorowa-Smirnowa.
    learning_rate : float, default=1e-3
        Współczynnik uczenia dla optymalizatora.
    batch_size : int, default=128
        Rozmiar mini-batcha podczas treningu.
    epochs : int, default=50
        Liczba epok treningu.
    beta : float, default=1.0
        Parametr bilansujący wagę regularyzacji KL w funkcji straty.
    device : str, default='cpu'
        Urządzenie na którym wykonywane będą obliczenia ('cpu' lub 'cuda').
    """

    def __init__(
        self,
        input_dim,
        window_size=1000,
        retrain_on_drift=False,
        hidden_dim=32,
        latent_dim=2,
        p_threshold=0.05,
        learning_rate=1e-3,
        batch_size=128,
        epochs=50,
        beta=1.0,
        device="cpu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.window_size = window_size
        self.retrain_on_drift = retrain_on_drift
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.p_threshold = p_threshold
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.beta = beta
        self.device = device

        # Inicjalizacja modelu VAE
        self.model = VAEModel(input_dim, hidden_dim, latent_dim)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Bufory danych
        self.reference_window = []
        self.current_window = []
        self.reference_errors = None

        # Dodajemy atrybut test_data, który będzie zawierał ostatnie próbki do obliczenia błędów rekonstrukcji
        self.test_data = []
        self.reconstruction_errors = []

        # Flagi stanu
        self.is_trained = False
        self.is_reference_set = False
        self.is_drift_detected = False

        # Historia detekcji dryfu
        self.drift_history = []

    def _vae_loss(self, x_recon, x, mu, log_var):
        """Funkcja straty VAE: błąd rekonstrukcji + regularyzacja KL."""
        recon_loss = F.mse_loss(x_recon, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + self.beta * kl_loss

    def _train_model(self, data):
        """Trenuje model VAE na danych."""
        if len(data) < self.batch_size:
            return False

        # Konwertuj dane do tensora
        data_tensor = torch.FloatTensor(np.array(data)).to(self.device)

        # Utwórz DataLoader
        dataset = torch.utils.data.TensorDataset(data_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        # Trenuj model
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_idx, (batch,) in enumerate(dataloader):
                self.optimizer.zero_grad()
                recon_batch, mu, log_var = self.model(batch)
                loss = self._vae_loss(recon_batch, batch, mu, log_var)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

        self.is_trained = True
        return True

    def _compute_reconstruction_errors(self, data):
        """Oblicza błędy rekonstrukcji dla zbioru danych."""
        self.model.eval()
        data_tensor = torch.FloatTensor(np.array(data)).to(self.device)
        with torch.no_grad():
            recon, _, _ = self.model(data_tensor)
            errors = torch.sum((recon - data_tensor) ** 2, dim=1).cpu().numpy()
        return errors

    def _compute_and_store_reconstruction_error(self, sample):
        """Oblicza i zapisuje błąd rekonstrukcji dla pojedynczej próbki."""
        if not self.is_trained:
            return 0.0

        self.model.eval()
        sample_tensor = torch.FloatTensor([sample]).to(self.device)
        with torch.no_grad():
            recon, _, _ = self.model(sample_tensor)
            error = torch.sum((recon - sample_tensor) ** 2).item()
            self.reconstruction_errors.append(error)
            return error

    def _detect_drift(self, current_window):
        """Wykrywa dryf porównując rozkłady błędów rekonstrukcji."""
        if not self.is_trained or not self.is_reference_set:
            return False, 1.0

        # Oblicz błędy rekonstrukcji dla aktualnego okna
        current_errors = self._compute_reconstruction_errors(current_window)

        # Wykonaj test Kołmogorowa-Smirnowa
        ks_stat, p_value = stats.ks_2samp(self.reference_errors, current_errors)

        # Jeśli p-wartość jest mniejsza od progu, wykryto dryf
        drift_detected = p_value < self.p_threshold

        return drift_detected, p_value

    def add_sample(self, x):
        """Dodaje pojedynczą próbkę do bufora aktualnego okna.

        Parameters
        ----------
        x : array-like
            Nowa obserwacja.

        Returns
        -------
        self : BatchVAEDriftDetector
            Ten obiekt.
        """
        # Konwersja x do formatu numpy
        x_array = np.asarray(x).reshape(1, -1)
        sample = x_array[0]

        # Dodaj próbkę do test_data dla kompatybilności z experiment_on_stream
        self.test_data.append(sample.copy())
        # Ogranicz rozmiar test_data do ostatnich window_size próbek
        if len(self.test_data) > self.window_size:
            self.test_data.pop(0)

        # Oblicz błąd rekonstrukcji dla tej próbki, jeśli model jest już wytrenowany
        if self.is_trained:
            self._compute_and_store_reconstruction_error(sample)

        # Dodaj próbkę do aktualnego okna
        self.current_window.append(sample)

        # Jeśli okno osiągnęło pełen rozmiar, przetwórz je
        if len(self.current_window) >= self.window_size:
            self._process_window()

        return self

    def add_batch(self, batch):
        """Dodaje cały batch danych.

        Parameters
        ----------
        batch : array-like
            Batch obserwacji.

        Returns
        -------
        self : BatchVAEDriftDetector
            Ten obiekt.
        """
        # Konwersja batcha do formatu numpy
        batch_array = np.asarray(batch)

        # Dodaj batch do aktualnego okna
        for sample in batch_array:
            self.add_sample(sample)  # Używamy add_sample dla spójności przetwarzania

        return self

    def _process_window(self):
        """Przetwarza aktualne okno - trenuje model lub wykrywa dryf."""
        # Jeśli nie mamy jeszcze okna referencyjnego, ustaw obecne okno jako referencyjne
        if not self.is_reference_set:
            self.reference_window = self.current_window.copy()

            # Trenuj model na oknie referencyjnym
            success = self._train_model(self.reference_window)
            if success:
                self.is_reference_set = True
                self.reference_errors = self._compute_reconstruction_errors(
                    self.reference_window
                )

                # Oblicz błędy rekonstrukcji dla próbek referencyjnych
                for sample in self.reference_window:
                    self._compute_and_store_reconstruction_error(sample)

            # Wyczyść aktualne okno
            self.current_window = []
            return

        # Wykryj dryf w aktualnym oknie
        drift_detected, p_value = self._detect_drift(self.current_window)
        self.is_drift_detected = drift_detected

        # Zapisz wynik detekcji
        self.drift_history.append(
            {"drift_detected": drift_detected, "p_value": p_value}
        )

        # Jeśli wykryto dryf i wymagane jest retrenowanie, zaktualizuj model
        if drift_detected and self.retrain_on_drift:
            print(f"Wykryto dryf (p-value: {p_value:.6f}). Trenowanie nowego modelu...")
            self.reference_window = self.current_window.copy()
            self._train_model(self.reference_window)
            self.reference_errors = self._compute_reconstruction_errors(
                self.reference_window
            )
        elif drift_detected:
            print(f"Wykryto dryf (p-value: {p_value:.6f}). Model pozostaje bez zmian.")

        # Wyczyść aktualne okno
        self.current_window = []

    def update(self, x):
        """Kompatybilność z interfejsem River - dodaje pojedynczą próbkę.

        Parameters
        ----------
        x : array-like
            Nowa obserwacja.

        Returns
        -------
        self : BatchVAEDriftDetector
            Ten obiekt.
        """
        return self.add_sample(x)

    def drift_detected(self):
        """Zwraca True, jeśli wykryto dryf, w przeciwnym razie False."""
        if self.is_drift_detected:
            # Resetuj flagę detekcji dryfu po jej użyciu
            self.is_drift_detected = False
            return True
        else:
            # Jeśli dryf nie został wykryty, nie resetuj flagi
            return False

    def get_latest_reconstruction_error(self):
        """Zwraca ostatni błąd rekonstrukcji, jeśli jest dostępny."""
        if self.reconstruction_errors:
            return self.reconstruction_errors[-1]
        return None

    def get_all_reconstruction_errors(self):
        """Zwraca wszystkie zapisane błędy rekonstrukcji."""
        return self.reconstruction_errors

    def process_stream(self, data_stream, reset=True):
        """Przetwarza cały strumień danych.

        Parameters
        ----------
        data_stream : array-like
            Strumień danych do przetworzenia.
        reset : bool, default=True
            Czy resetować detektor przed przetwarzaniem.

        Returns
        -------
        list
            Historia detekcji dryfu.
        """
        if reset:
            # Resetuj stan detektora
            self.reference_window = []
            self.current_window = []
            self.test_data = []
            self.reconstruction_errors = []
            self.is_trained = False
            self.is_reference_set = False
            self.is_drift_detected = False
            self.drift_history = []

        # Przetwarzaj dane w oknach o rozmiarze window_size
        data_array = np.asarray(data_stream)
        num_samples = len(data_array)

        for i in range(0, num_samples, self.window_size):
            end_idx = min(i + self.window_size, num_samples)
            batch = data_array[i:end_idx]

            # Jeśli batch jest pełnego rozmiaru, przetwórz go bezpośrednio
            if len(batch) == self.window_size:
                if not self.is_reference_set:
                    self.reference_window = batch
                    self._train_model(self.reference_window)
                    self.is_reference_set = True
                    self.reference_errors = self._compute_reconstruction_errors(
                        self.reference_window
                    )

                    # Dodaj próbki do test_data i oblicz błędy rekonstrukcji
                    for sample in batch:
                        self.test_data.append(sample.copy())
                        self._compute_and_store_reconstruction_error(sample)
                else:
                    drift_detected, p_value = self._detect_drift(batch)
                    self.is_drift_detected = drift_detected

                    self.drift_history.append(
                        {
                            "window_idx": len(self.drift_history),
                            "sample_range": f"{i}-{end_idx-1}",
                            "drift_detected": drift_detected,
                            "p_value": p_value,
                        }
                    )

                    # Dodaj próbki do test_data i oblicz błędy rekonstrukcji
                    for sample in batch:
                        self.test_data.append(sample.copy())
                        # Ogranicz rozmiar test_data
                        if len(self.test_data) > self.window_size:
                            self.test_data.pop(0)
                        self._compute_and_store_reconstruction_error(sample)

                    if drift_detected and self.retrain_on_drift:
                        print(
                            f"Wykryto dryf w oknie {len(self.drift_history)-1} (p-value: {p_value:.6f}). Trenowanie nowego modelu..."
                        )
                        self.reference_window = batch
                        self._train_model(self.reference_window)
                        self.reference_errors = self._compute_reconstruction_errors(
                            self.reference_window
                        )
                    elif drift_detected:
                        print(
                            f"Wykryto dryf w oknie {len(self.drift_history)-1} (p-value: {p_value:.6f}). Model pozostaje bez zmian."
                        )
            else:
                # Dla niepełnego batcha dodaj próbki pojedynczo
                for sample in batch:
                    self.add_sample(sample)

        return self.drift_history

    def get_drift_history(self):
        """Zwraca historię detekcji dryfu."""
        return self.drift_history
