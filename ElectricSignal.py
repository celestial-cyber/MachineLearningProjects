
import numpy as np
import matplotlib.pyplot as plt

# FM wave parameters
Fs = 2000        # Sampling frequency (samples/sec)
t = np.arange(0, 0.2, 1/Fs)  # Time vector for 0.2 seconds

fc = 100         # Carrier frequency (Hz)
fm = 30          # Modulating signal frequency (Hz)
b = 1            # Modulation index

# Message (modulating) signal
m_signal = np.sin(2 * np.pi * fm * t)

# Carrier signal (just for visualization)
carrier_signal = np.sin(2 * np.pi * fc * t)

# FM Signal: frequency of carrier is modulated by the amplitude of m_signal
fm_wave = np.sin(2 * np.pi * fc * t + b * m_signal)

# Plot all three on one figure
plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
plt.plot(t, m_signal)
plt.title("Message Signal")
plt.subplot(3,1,2)
plt.plot(t, carrier_signal)
plt.title("Carrier Signal")
plt.subplot(3,1,3)
plt.plot(t, fm_wave)
plt.title("FM Signal")
plt.tight_layout()
plt.show()
