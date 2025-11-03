import numpy as np
from scipy.io import wavfile
from ligotools.utils import whiten, write_wavfile, reqshift


def test_whiten_constant_psd_scales_signal():
    rng = np.random.default_rng(123)
    dt = 1.0 / 4096.0
    c = 4.0  # constant PSD
    x = rng.standard_normal(4096)

    def const_psd(freqs):
        return np.full_like(freqs, c, dtype=float)

    y = whiten(x, const_psd, dt)

    expected = x * np.sqrt(2 * dt / c)
    # fft/ifft roundoff -> small tolerance
    np.testing.assert_allclose(y, expected, rtol=1e-10, atol=1e-10)
    assert y.shape == x.shape
    assert np.isrealobj(y)


def test_reqshift_moves_tone_up_in_frequency():
    fs = 4096
    T = 1.0
    N = int(fs * T)
    t = np.arange(N) / fs

    f0 = 100.0
    fshift = 200.0

    x = np.sin(2 * np.pi * f0 * t)
    y = reqshift(x, fshift=fshift, sample_rate=fs)

    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(N, d=1 / fs)

    f_peak_x = freqs[np.argmax(np.abs(X))]
    f_peak_y = freqs[np.argmax(np.abs(Y))]
    assert abs(f_peak_x - f0) <= 1.0
    assert abs(f_peak_y - (f0 + fshift)) <= 1.0
    assert np.isclose(np.linalg.norm(y), np.linalg.norm(x), rtol=0.05)


def test_write_wavfile_roundtrip(tmp_path):
    fs = 8000
    t = np.arange(8000) / fs
    x = np.sin(2 * np.pi * 440 * t)  # in [-1, 1]
    x[0] = 1.0  # ensure exact peak present

    out = tmp_path / "test.wav"
    write_wavfile(str(out), fs, x)

    assert out.exists(), "WAV file was not created"

    sr_read, data_read = wavfile.read(out)
    assert sr_read == fs
    assert data_read.dtype == np.int16
    expected_max = int(32767 * 0.9)
    assert np.max(np.abs(data_read)) == expected_max
