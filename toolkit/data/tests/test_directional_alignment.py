import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from toolkit.data.utils.ndbc_data import (
    align_directional_coefficients,
    build_directional_spectrum_dataset,
)
from toolkit.data.utils.ndbc_plotting import plot_average_directional_spectrum


class DirectionalAlignmentTests(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2020-01-01", periods=4, freq="h")
        self.complete = [pd.DataFrame(value, index=times, columns=["0.10", "0.20"])
                         for value in [90, 120, 20, 10, 2]]
        self.misaligned = [frame.copy() for frame in self.complete]
        self.misaligned[0] = self.misaligned[0].iloc[[3, 1, 2], ::-1]
        self.misaligned[1] = self.misaligned[1].iloc[[0, 2, 1]]
        self.misaligned[2].columns = [0.1, 0.2]
        self.expected = [frame.iloc[1:3] for frame in self.complete]

    def tearDown(self):
        plt.close("all")

    def test_aligns_labels_before_reconstruction_and_plot(self):
        actual = build_directional_spectrum_dataset(*self.misaligned, 44.6, -124.5)
        expected = build_directional_spectrum_dataset(*self.expected, 44.6, -124.5)
        xr.testing.assert_identical(actual, expected)
        args = ("46050", "2020-01-01", "2020-01-02")
        figure = plot_average_directional_spectrum(*self.misaligned, *args)
        expected_figure = plot_average_directional_spectrum(*self.expected, *args)
        np.testing.assert_allclose(figure.axes[0].collections[0].get_array(),
                                   expected_figure.axes[0].collections[0].get_array())

    def test_missing_values_share_a_mask_and_common_frequencies(self):
        frames = [frame.copy() for frame in self.complete]
        frames[0] = frames[0][["0.20"]]
        frames[0].iloc[0, 0] = 999
        aligned = align_directional_coefficients(*frames)
        for frame in aligned:
            self.assertEqual(frame.shape, (3, 1))
            self.assertEqual(list(frame.columns), [0.2])
            self.assertEqual(frame.index[0], self.complete[0].index[1])

    def test_no_overlap_and_duplicate_dates_raise(self):
        frames = [frame.copy() for frame in self.complete]
        frames[0].index += pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "no shared"):
            align_directional_coefficients(*frames)
        frames[0] = pd.concat([self.complete[0], self.complete[0]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            align_directional_coefficients(*frames)


if __name__ == "__main__":
    unittest.main()
