import math
import numpy as np

class LowPassFilter:
    def __init__(self, alpha: float = 0.5):
        self.__set_alpha(alpha)
        self.__y = None

    def __set_alpha(self, alpha: float):
        self.__alpha = max(0.0, min(1.0, alpha))

    def filter(self, value, alpha: float = None):
        if alpha is not None:
            self.__set_alpha(alpha)
        if self.__y is None:
            self.__y = value
        else:
            self.__y = self.__alpha * value + (1.0 - self.__alpha) * self.__y
        return self.__y

    def reset(self):
        self.__y = None


class OneEuroFilter:
    """
    Filtro 1 Euro para eliminar el ruido (jitter) en coordenadas
    sin anular la respuesta rapida en movimientos explosivos.
    """
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filt = LowPassFilter()
        self.dx_filt = LowPassFilter()
        self.last_time = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, timestamp_s: float):
        if self.last_time is None:
            self.last_time = timestamp_s
            return x

        dt = timestamp_s - self.last_time
        self.last_time = timestamp_s

        if dt <= 1e-4:
            return x

        # Estimación de la velocidad de variación
        prev_x = self.x_filt.filter(x)
        dx = (x - prev_x) / dt
        edx = self.dx_filt.filter(dx, self._alpha(self.d_cutoff, dt))

        # Frecuencia de corte adaptativa
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filt.filter(x, self._alpha(cutoff, dt))

    def reset(self):
        self.x_filt.reset()
        self.dx_filt.reset()
        self.last_time = None