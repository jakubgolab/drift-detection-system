from river import base
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats
from collections import deque
from models.vae_model import VAEModel


class VAEDriftDetector(base.DriftDetector):
    """Detektor dryfu koncepcji używający Wariacyjnego Autoenkodera.

    Śledzi błędy rekonstrukcji VAE i używa testu statystycznego do wykrywania
    istotnych zmian rozkładu błędów, co wskazuje na dryf.

    Parameters
    ----------
    input_dim : int
        Liczba wymiarów danych wejściowych.
    hidden_dim : int, default=32
        Liczba jednostek w warstwach ukrytych.
    latent_dim : int, default=2
        Liczba wymiarów przestrzeni ukrytej.
    reference_size : int, default=300
        Liczba obserwacji używanych do trenowania początkowego modelu i jako referencja.
    test_size : int, default=100
        Liczba obserwacji używanych w oknie testowym do wykrywania dryfu.
    p_threshold : float, default=0.01
        Próg p-wartości dla testu Kołmogorowa-Smirnowa.
    burn_in : int, default=0
        Liczba obserwacji ignorowanych przed rozpoczęciem detekcji (opcjonalnie).
    learning_rate : float, default=1e-3
        Współczynnik uczenia dla optymalizatora.
    batch_size : int, default=32
        Rozmiar batcha podczas treningu.
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
        hidden_dim=32,
        latent_dim=2,
        reference_size=300,
        test_size=100,
        p_threshold=0.01,
        burn_in=0,
        learning_rate=1e-3,
        batch_size=32,
        epochs=50,
        beta=1.0,
        device="cpu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.reference_size = reference_size
        self.test_size = test_size
        self.p_threshold = p_threshold
        self.burn_in = burn_in
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.beta = beta
        self.device = device

        # Inicjalizacja modelu VAE
        self.model = VAEModel(input_dim, hidden_dim, latent_dim)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Liczniki i bufory
        self.n_samples = 0
        self.reference_data = deque(maxlen=reference_size)
        self.test_data = deque(maxlen=test_size)
        self.reference_errors = None
        self.is_trained = False
        self.is_drift_detected = False

    def _vae_loss(self, x_recon, x, mu, log_var):
        """Funkcja straty VAE: błąd rekonstrukcji + regularyzacja KL."""
        recon_loss = F.mse_loss(x_recon, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + self.beta * kl_loss

    def _train_model(self, data):
        """Trenuje model VAE na danych referencyjnych."""
        if len(data) < self.batch_size:
            return

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

    def _compute_reconstruction_errors(self, data):
        """Oblicza błędy rekonstrukcji dla zbioru danych."""
        self.model.eval()
        data_tensor = torch.FloatTensor(np.array(data)).to(self.device)
        with torch.no_grad():
            recon, _, _ = self.model(data_tensor)
            errors = torch.sum((recon - data_tensor) ** 2, dim=1).cpu().numpy()
        return errors

    def _detect_drift(self):
        """Wykrywa dryf porównując rozkłady błędów rekonstrukcji."""
        if not self.is_trained or len(self.test_data) < self.test_size:
            return False

        # Oblicz błędy rekonstrukcji dla danych testowych
        test_errors = self._compute_reconstruction_errors(self.test_data)

        # Wykonaj test Kołmogorowa-Smirnowa
        ks_stat, p_value = stats.ks_2samp(self.reference_errors, test_errors)

        # Jeśli p-wartość jest mniejsza od progu, wykryto dryf
        return p_value < self.p_threshold, p_value

    def update(self, x):
        """Aktualizuje detektor dryfu z nową obserwacją.

        Parameters
        ----------
        x : array-like
            Nowa obserwacja.

        Returns
        -------
        self : VAEDriftDetector
            Ten obiekt.
        """
        self.n_samples += 1

        # Pomijamy burn-in
        if self.n_samples <= self.burn_in:
            return self

        # Konwersja x do formatu numpy
        x_array = np.asarray(x).reshape(1, -1)

        # Faza zbierania danych referencyjnych
        if len(self.reference_data) < self.reference_size:
            self.reference_data.append(x_array[0])

            # Trenuj model gdy zbierzemy wystarczająco danych
            if len(self.reference_data) == self.reference_size:
                self._train_model(self.reference_data)
                self.reference_errors = self._compute_reconstruction_errors(
                    self.reference_data
                )

            self.is_drift_detected = False
            return self

        # Faza testowania - dodaj nową obserwację do okna testowego
        self.test_data.append(x_array[0])

        # Wykryj dryf gdy mamy wystarczająco danych testowych
        if len(self.test_data) >= self.test_size:
            is_drift, p_value = self._detect_drift()
            self.is_drift_detected = is_drift

            # Jeśli wykryto dryf, ustaw test_data jako nowe reference_data i retrenuj model
            if is_drift:
                print("Retraining model...")
                self.reference_data = deque(self.test_data, maxlen=self.reference_size)
                self.test_data.clear()
                self._train_model(self.reference_data)
                self.reference_errors = self._compute_reconstruction_errors(
                    self.reference_data
                )

        return self

    def drift_detected(self):
        """Zwraca True, jeśli wykryto dryf, w przeciwnym razie False."""
        return self.is_drift_detected
