"""read_xsens_txt — the comment-block scan, which is the whole reason it was rewritten.

Exports in hand run to 4 comment lines (IMoVE's 40 Hz sessions), 5 (Al Borno) and 12 (IMoVE's
long walks). The old code hardcoded `skiprows=5`, so it silently consumed the column header on
the first and third. The only existing test used a 5-line block -- the one case that already
worked -- so the regression the rewrite exists to prevent was untested.

The rate matters just as much: defaulting a missing `// Update Rate` to 100 Hz would mislabel
the 40 Hz sessions by a factor of 2.5, which is a wrong answer rather than a failure.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.toolchest.building.xsens import read_xsens_txt

COLUMNS = ("PacketCounter\tSampleTimeFine\tAcc_X\tAcc_Y\tAcc_Z\t"
           "Gyr_X\tGyr_Y\tGyr_Z\tMag_X\tMag_Y\tMag_Z")
ROWS = "\n".join(f"{i}\t\t9.8{i}\t0.1\t-0.2\t0.01\t0.02\t0.03\t-0.6\t-0.2\t0.5"
                 for i in range(5))

SHORT_HEADER = """// Start Time: Unknown
// Update Rate: 40.0Hz
// Filter Profile: human (46.1)
// Firmware Version: 4.6.0"""

AL_BORNO_HEADER = SHORT_HEADER + "\n// Option Flags: none"

LONG_HEADER = """// General information:
//  MT Manager version: 2022.2.0
//  XDA version: 2022.2.0 build 7381
// Device information:
//  DeviceId: 00B4D6D1
//  ProductCode: MTW2-3A7G6
//  Firmware Version: 4.6.0
//  Hardware Version: 2.0.0
// Device settings:
//  Filter Profile: human(46.1)
//  Update Rate: 100.0Hz
// Coordinate system: ENU"""


class TestCommentBlockScan(unittest.TestCase):
    def _read(self, header, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sensor.txt'
            path.write_text(f"{header}\n{COLUMNS}\n{ROWS}\n")
            return read_xsens_txt(path, **kwargs)

    def test_a_four_line_header(self):
        """IMoVE's 40 Hz sessions. skiprows=5 ate the column header here."""
        trace = self._read(SHORT_HEADER)
        self.assertEqual(len(trace), 5)
        self.assertAlmostEqual(trace.get_sample_frequency(), 40.0, places=6)

    def test_a_five_line_header(self):
        """Al Borno. The case the old hardcoded skip happened to fit."""
        self.assertEqual(len(self._read(AL_BORNO_HEADER)), 5)

    def test_a_twelve_line_header(self):
        """IMoVE's long walks, and the header where the rate sits on line 11 rather than 2,
        so finding it means scanning rather than indexing."""
        trace = self._read(LONG_HEADER)
        self.assertEqual(len(trace), 5)
        self.assertAlmostEqual(trace.get_sample_frequency(), 100.0, places=6)

    def test_the_columns_are_not_eaten_by_the_scan(self):
        """The failure mode of a wrong skip count is a lost first row, not an exception."""
        trace = self._read(SHORT_HEADER)
        np.testing.assert_allclose(trace.acc[0], [9.80, 0.1, -0.2])
        np.testing.assert_allclose(trace.acc[-1], [9.84, 0.1, -0.2])


class TestSampleRate(unittest.TestCase):
    def _read(self, header, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sensor.txt'
            path.write_text(f"{header}\n{COLUMNS}\n{ROWS}\n")
            return read_xsens_txt(path, **kwargs)

    def test_a_missing_rate_raises_rather_than_defaulting(self):
        """Defaulting to 100 Hz would mislabel the 40 Hz sessions by 2.5x, and a wrong
        timebase is far worse than a refusal -- every derivative and every sync depends on it.
        """
        headerless = "// Start Time: Unknown\n// Filter Profile: human (46.1)"
        with self.assertRaises(ValueError) as caught:
            self._read(headerless)
        self.assertIn('Update Rate', str(caught.exception))

    def test_an_explicit_rate_overrides_the_header(self):
        trace = self._read(SHORT_HEADER, sample_rate_hz=100.0)
        self.assertAlmostEqual(trace.get_sample_frequency(), 100.0, places=6)

    def test_an_explicit_rate_rescues_a_file_with_no_header_rate(self):
        """How the long-walk exports were readable before the rate was added to them."""
        headerless = "// Start Time: Unknown"
        trace = self._read(headerless, sample_rate_hz=100.0)
        self.assertAlmostEqual(trace.get_sample_frequency(), 100.0, places=6)

    def test_timestamps_start_at_zero_and_step_by_the_rate(self):
        trace = self._read(SHORT_HEADER)
        np.testing.assert_allclose(trace.timestamps, np.arange(5) / 40.0)


if __name__ == '__main__':
    unittest.main()
