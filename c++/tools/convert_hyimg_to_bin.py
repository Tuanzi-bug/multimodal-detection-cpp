"""
convert_hyimg_to_bin.py
=======================
Convert hyperspectral image .mat files to binary .bin for C++ reading.

Source data layout:
    test_data = scipy.io.loadmat(path)['data']  # shape [H, W, Bands], dtype float64

Binary format written:
    [uint32_t height][uint32_t width][uint32_t bands][uint32_t dtype_size=8]
    [float64  H×W×bands row-major data]

The C++ reader performs transpose/reshape; this script writes the raw
[H, W, Bands] order without pre-transposing.

Usage:
    python c++/tools/convert_hyimg_to_bin.py [--input-dir DIR] [--output-dir DIR] [--verify]

Defaults:
    --input-dir  hyperspectral/          (relative to cwd / project root)
    --output-dir c++/data/hyperspectral/ (relative to cwd / project root)

Run doctests:
    python -m doctest c++/tools/convert_hyimg_to_bin.py -v

Run unittests:
    python -m pytest c++/tools/convert_hyimg_to_bin.py
"""

import argparse
import os
import struct
import sys
import unittest
import tempfile


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DTYPE_SIZE   = 8           # float64 → 8 bytes
HEADER_FMT   = '<IIII'     # little-endian: uint32 H, W, bands, dtype_size
HEADER_BYTES = struct.calcsize(HEADER_FMT)   # 16 bytes


# ---------------------------------------------------------------------------
# Core I/O helpers
# ---------------------------------------------------------------------------

def encode_header(height: int, width: int, bands: int,
                  dtype_size: int = DTYPE_SIZE) -> bytes:
    """Return the 16-byte binary header.

    >>> encode_header(100, 200, 129)[:4]
    b'd\\x00\\x00\\x00'
    >>> len(encode_header(1, 1, 1))
    16
    >>> encode_header(1, 1, 1)[-4:]
    b'\\x08\\x00\\x00\\x00'
    """
    return struct.pack(HEADER_FMT, height, width, bands, dtype_size)


def decode_header(data: bytes) -> tuple:
    """Return (height, width, bands, dtype_size) decoded from a 16-byte header.

    >>> decode_header(encode_header(100, 200, 129))
    (100, 200, 129, 8)
    >>> decode_header(encode_header(3, 7, 11, 8))
    (3, 7, 11, 8)
    """
    if len(data) < HEADER_BYTES:
        raise ValueError(
            f'Header too short: expected {HEADER_BYTES} bytes, got {len(data)}'
        )
    height, width, bands, dtype_size = struct.unpack(HEADER_FMT, data[:HEADER_BYTES])
    return height, width, bands, dtype_size


def write_hybin(array, dest_path: str) -> None:
    """Write a 3-D float64 numpy array [H, W, Bands] to a .bin file.

    Format:
        [uint32 H][uint32 W][uint32 Bands][uint32 dtype_size=8]
        [float64 row-major H×W×Bands data]

    The data is stored in the original [H, W, Bands] order.
    The C++ reader is responsible for any required transpose/reshape.

    Args:
        array:     3-D numpy array with shape (H, W, Bands) and dtype float64.
        dest_path: Absolute or relative path for the output .bin file.

    Raises:
        ValueError: If array is not 3-D or not float64-convertible.
        OSError:    If the destination path cannot be written.
    """
    import numpy as np
    if array.ndim != 3:
        raise ValueError(f'Expected 3-D array [H,W,Bands], got {array.ndim}-D')
    arr64 = np.asarray(array, dtype=np.float64)
    height, width, bands = arr64.shape
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    with open(dest_path, 'wb') as fh:
        fh.write(encode_header(height, width, bands))
        fh.write(arr64.tobytes(order='C'))   # row-major: H varies slowest


def read_hybin(src_path: str) -> tuple:
    """Read a .bin file produced by write_hybin.

    Args:
        src_path: Path to the .bin file.

    Returns:
        (height, width, bands, numpy.ndarray) where array has shape
        (height, width, bands) and dtype float64.

    Raises:
        FileNotFoundError: If src_path does not exist.
        ValueError:        If the file is shorter than header + data size.
    """
    import numpy as np
    with open(src_path, 'rb') as fh:
        header = fh.read(HEADER_BYTES)
        height, width, bands, dtype_size = decode_header(header)
        expected_bytes = height * width * bands * dtype_size
        raw = fh.read(expected_bytes)
    if len(raw) < expected_bytes:
        raise ValueError(
            f'Truncated data in {src_path}: '
            f'expected {expected_bytes} bytes, got {len(raw)}'
        )
    arr = np.frombuffer(raw, dtype=np.float64).reshape(height, width, bands)
    return height, width, bands, arr


def validate_hyimg_data(data, path: str) -> None:
    """Validate shape constraints for a hyperspectral image cube.

    Args:
        data: 3-D numpy array (H, W, Bands).
        path: Source path used only for error messages.

    Raises:
        ValueError: If the array is not 3-D or any dimension is zero.
    """
    if data.ndim != 3:
        raise ValueError(
            f'{path}: expected 3-D [H,W,Bands] data, got shape {data.shape}'
        )
    h, w, b = data.shape
    if h <= 0 or w <= 0 or b <= 0:
        raise ValueError(
            f'{path}: all dimensions must be > 0, got H={h}, W={w}, Bands={b}'
        )


# ---------------------------------------------------------------------------
# Conversion logic
# ---------------------------------------------------------------------------

def load_hymat(path: str):
    """Load a hyperspectral .mat file and return the 'data' array.

    Args:
        path: Path to the .mat file.

    Returns:
        numpy.ndarray with dtype float64, shape (H, W, Bands).

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError:          If the .mat file has no 'data' key.
        ValueError:        If the shape constraints are not met.
        ImportError:       If scipy is not installed.
    """
    try:
        import scipy.io as scio
    except ImportError as exc:
        raise ImportError(
            'scipy is required to read .mat files. '
            'Install with: pip install scipy'
        ) from exc

    if not os.path.isfile(path):
        raise FileNotFoundError(f'Mat file not found: {path}')

    mat = scio.loadmat(path)
    if 'data' not in mat:
        raise KeyError(
            f"Key 'data' not found in {path}. "
            f"Available keys: {[k for k in mat if not k.startswith('_')]}"
        )
    import numpy as np
    data = np.array(mat['data'], dtype=np.float64)
    validate_hyimg_data(data, path)
    return data


def convert_hyfile(mat_path: str, bin_path: str, verify: bool = False) -> dict:
    """Convert one hyperspectral .mat file to .bin.

    Args:
        mat_path: Source .mat file path.
        bin_path: Destination .bin file path.
        verify:   If True, verify header values and a sample of data values.

    Returns:
        dict with keys: filename, shape, file_size_bytes, status, error.
    """
    import numpy as np
    filename = os.path.basename(mat_path)
    result = {
        'filename':        filename,
        'shape':           None,
        'file_size_bytes': None,
        'status':          'OK',
        'error':           None,
    }

    try:
        data = load_hymat(mat_path)
        result['shape'] = data.shape          # (H, W, Bands)
        write_hybin(data, bin_path)
        result['file_size_bytes'] = os.path.getsize(bin_path)

        if verify:
            h_r, w_r, b_r, arr_r = read_hybin(bin_path)
            h, w, b = data.shape

            # Verify header values
            if (h_r, w_r, b_r) != (h, w, b):
                raise ValueError(
                    f'Header mismatch: wrote ({h},{w},{b}), '
                    f'read ({h_r},{w_r},{b_r})'
                )

            # Sample verification: compare first and last 100 elements
            flat_orig = data.ravel()
            flat_read = arr_r.ravel()
            sample_idx = list(range(min(100, flat_orig.size)))
            if flat_orig.size > 100:
                sample_idx += list(range(flat_orig.size - 100, flat_orig.size))
            sample_diff = float(
                np.max(np.abs(flat_orig[sample_idx] - flat_read[sample_idx]))
            )
            if sample_diff >= 1e-15:
                raise ValueError(
                    f'Sample max diff {sample_diff:.2e} exceeds 1e-15'
                )
            result['status'] = f'OK (verified)'

    except Exception as exc:
        result['status'] = 'FAIL'
        result['error']  = str(exc)

    return result


def convert_hydir(
    input_dir:  str,
    output_dir: str,
    verify:     bool = False,
) -> list:
    """Convert all .mat files in the hyperspectral input directory.

    Args:
        input_dir:  Directory containing hyperspectral .mat files
                    (flat structure, no class subdirs).
        output_dir: Output directory; .bin files are written here.
        verify:     If True, run verification on each output file.

    Returns:
        List of result dicts (one per .mat file processed or attempted).
    """
    results = []

    if not os.path.isdir(input_dir):
        print(f'WARNING: input directory not found: {input_dir}', file=sys.stderr)
        return results

    mat_files = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith('.mat')
    )
    if not mat_files:
        print(f'WARNING: no .mat files found in {input_dir}', file=sys.stderr)
        return results

    os.makedirs(output_dir, exist_ok=True)

    for fname in mat_files:
        mat_path = os.path.join(input_dir, fname)
        bin_name = os.path.splitext(fname)[0] + '.bin'
        bin_path = os.path.join(output_dir, bin_name)
        result   = convert_hyfile(mat_path, bin_path, verify=verify)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------

def _format_size(bytes_: int) -> str:
    """Return a human-readable file size string."""
    if bytes_ is None:
        return '—'
    if bytes_ >= 1024 ** 3:
        return f'{bytes_ / 1024**3:.1f} GB'
    if bytes_ >= 1024 ** 2:
        return f'{bytes_ / 1024**2:.1f} MB'
    if bytes_ >= 1024:
        return f'{bytes_ / 1024:.1f} KB'
    return f'{bytes_} B'


def print_summary(results: list) -> int:
    """Print a formatted summary table.

    Args:
        results: List of result dicts from convert_hydir.

    Returns:
        Number of failed conversions.
    """
    ok_count   = sum(1 for r in results if r['status'].startswith('OK'))
    fail_count = sum(1 for r in results if r['status'] == 'FAIL')

    file_w   = max(8, max((len(r['filename'])      for r in results), default=0))
    shape_w  = 18
    size_w   = 10
    status_w = max(6, max((len(r['status'])        for r in results), default=0))

    header = (
        f"{'Filename':<{file_w}}  {'Shape (H×W×Bands)':<{shape_w}}  "
        f"{'File size':<{size_w}}  {'Status':<{status_w}}"
    )
    print()
    print(header)
    print('-' * len(header))

    for r in results:
        shape_str = (
            f"{r['shape'][0]}×{r['shape'][1]}×{r['shape'][2]}"
            if r['shape'] else '—'
        )
        size_str  = _format_size(r['file_size_bytes'])
        print(
            f"{r['filename']:<{file_w}}  {shape_str:<{shape_w}}  "
            f"{size_str:<{size_w}}  {r['status']}"
        )
        if r['error']:
            print(f"  ERROR: {r['error']}")

    print()
    print(f'Summary: {ok_count} OK, {fail_count} FAIL')
    return fail_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            'Convert hyperspectral image .mat files to binary .bin '
            'for C++ reading.'
        )
    )
    parser.add_argument(
        '--input-dir',
        default='hyperspectral',
        help='Directory containing hyperspectral .mat files (default: hyperspectral/)',
    )
    parser.add_argument(
        '--output-dir',
        default=os.path.join('c++', 'data', 'hyperspectral'),
        help='Output directory (default: c++/data/hyperspectral/)',
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify header and sample data values after writing',
    )
    return parser


def main(argv=None) -> int:
    """Entry point for CLI use.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 if all conversions succeeded, 1 otherwise.
    """
    parser = build_parser()
    args   = parser.parse_args(argv)

    print(f'Input  dir : {args.input_dir}')
    print(f'Output dir : {args.output_dir}')
    print(f'Verify     : {args.verify}')

    results    = convert_hydir(args.input_dir, args.output_dir, args.verify)
    fail_count = print_summary(results)

    return 1 if fail_count > 0 else 0


# ---------------------------------------------------------------------------
# Tests (unittest)
# ---------------------------------------------------------------------------

class TestEncodeDecodeHeader(unittest.TestCase):
    """Unit tests for the 4-field binary header."""

    def test_roundtrip(self):
        """encode then decode returns original values."""
        header = encode_header(100, 200, 129)
        h, w, b, ds = decode_header(header)
        self.assertEqual((h, w, b, ds), (100, 200, 129, 8))

    def test_header_length(self):
        """Header is exactly 16 bytes."""
        self.assertEqual(len(encode_header(1, 1, 1)), 16)

    def test_dtype_size_field(self):
        """dtype_size field is always 8."""
        _, _, _, ds = decode_header(encode_header(5, 10, 20))
        self.assertEqual(ds, 8)

    def test_little_endian(self):
        """First 4 bytes encode height in little-endian."""
        header = encode_header(1, 2, 3)
        self.assertEqual(header[:4], b'\x01\x00\x00\x00')

    def test_decode_too_short(self):
        """decode_header raises ValueError for short input."""
        with self.assertRaises(ValueError):
            decode_header(b'\x00' * 10)


class TestWriteReadHybin(unittest.TestCase):
    """Integration tests for write_hybin / read_hybin round-trip."""

    def _make_cube(self, h=20, w=30, b=10):
        import numpy as np
        rng = np.random.default_rng(seed=7)
        return np.asarray(rng.random((h, w, b)), dtype=np.float64)

    def test_roundtrip_shape(self):
        """Read-back shape matches written shape."""
        cube = self._make_cube(20, 30, 129)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'cube.bin')
            write_hybin(cube, path)
            h, w, b, arr = read_hybin(path)
        self.assertEqual((h, w, b), (20, 30, 129))
        self.assertEqual(arr.shape, (20, 30, 129))

    def test_roundtrip_values(self):
        """Values are bit-identical after round-trip."""
        import numpy as np
        cube = self._make_cube(10, 12, 5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'cube.bin')
            write_hybin(cube, path)
            _, _, _, arr = read_hybin(path)
        self.assertTrue(np.array_equal(cube, arr))

    def test_file_size(self):
        """Output file size matches header + data."""
        cube = self._make_cube(5, 6, 7)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'cube.bin')
            write_hybin(cube, path)
            expected = HEADER_BYTES + 5 * 6 * 7 * DTYPE_SIZE
            self.assertEqual(os.path.getsize(path), expected)

    def test_creates_parent_dirs(self):
        """write_hybin creates intermediate directories."""
        import numpy as np
        cube = self._make_cube(3, 3, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'a', 'b', 'cube.bin')
            write_hybin(cube, path)
            self.assertTrue(os.path.isfile(path))

    def test_rejects_non_3d(self):
        """write_hybin raises ValueError for non-3D array."""
        import numpy as np
        arr2d = np.zeros((5, 5), dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                write_hybin(arr2d, os.path.join(tmpdir, 'bad.bin'))

    def test_row_major_order(self):
        """Data is stored in C (row-major) order: last axis varies fastest."""
        import numpy as np
        # Build a known array and check the raw bytes match C order
        cube = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'order_test.bin')
            write_hybin(cube, path)
            with open(path, 'rb') as fh:
                fh.read(HEADER_BYTES)
                raw = np.frombuffer(fh.read(), dtype=np.float64)
        self.assertTrue(np.array_equal(raw, cube.ravel(order='C')))


class TestValidateHyimgData(unittest.TestCase):
    """Tests for shape validation of hyperspectral cubes."""

    def _make(self, h, w, b):
        import numpy as np
        return np.zeros((h, w, b), dtype=np.float64)

    def test_valid_shape(self):
        """Valid shape does not raise."""
        validate_hyimg_data(self._make(100, 200, 129), 'dummy.mat')

    def test_non_3d_raises(self):
        """2-D array raises ValueError."""
        import numpy as np
        with self.assertRaises(ValueError):
            validate_hyimg_data(np.zeros((10, 10)), 'dummy.mat')

    def test_zero_height_raises(self):
        """Zero height raises ValueError."""
        with self.assertRaises(ValueError):
            validate_hyimg_data(self._make(0, 10, 5), 'dummy.mat')


class TestConvertHydir(unittest.TestCase):
    """Integration tests using synthetic .mat files."""

    def _write_fake_hymat(self, path, h=50, w=60, b=10):
        """Write a minimal fake hyperspectral .mat file."""
        try:
            import scipy.io as scio
            import numpy as np
            os.makedirs(os.path.dirname(path), exist_ok=True)
            rng  = np.random.default_rng(seed=1)
            data = rng.random((h, w, b))
            scio.savemat(path, {'data': data})
            return data
        except ImportError:
            self.skipTest('scipy not available')

    def test_empty_directory(self):
        """Missing input directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = convert_hydir(
                os.path.join(tmpdir, 'nonexistent'),
                os.path.join(tmpdir, 'out'),
            )
        self.assertEqual(results, [])

    def test_valid_mat_converted(self):
        """A valid hyperspectral .mat is converted and verified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir  = os.path.join(tmpdir, 'hyperspectral')
            out_dir = os.path.join(tmpdir, 'out')
            self._write_fake_hymat(os.path.join(in_dir, 'img1.mat'))

            results = convert_hydir(in_dir, out_dir, verify=True)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]['status'].startswith('OK'))

    def test_invalid_mat_missing_data_key(self):
        """A .mat file without 'data' key is reported as FAIL."""
        try:
            import scipy.io as scio
            import numpy as np
        except ImportError:
            self.skipTest('scipy not available')

        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir  = os.path.join(tmpdir, 'hyperspectral')
            out_dir = os.path.join(tmpdir, 'out')
            os.makedirs(in_dir)
            bad_mat = os.path.join(in_dir, 'bad.mat')
            scio.savemat(bad_mat, {'wrong_key': np.zeros((10, 10, 5))})

            results = convert_hydir(in_dir, out_dir)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['status'], 'FAIL')

    def test_multiple_files(self):
        """Multiple .mat files are all converted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir  = os.path.join(tmpdir, 'hyperspectral')
            out_dir = os.path.join(tmpdir, 'out')
            for i in range(3):
                self._write_fake_hymat(os.path.join(in_dir, f'img{i}.mat'))

            results = convert_hydir(in_dir, out_dir)

        self.assertEqual(len(results), 3)
        self.assertTrue(all(r['status'].startswith('OK') for r in results))

    def test_output_bin_shape_matches_input(self):
        """Shape stored in .bin header matches original array shape."""
        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir  = os.path.join(tmpdir, 'hyperspectral')
            out_dir = os.path.join(tmpdir, 'out')
            self._write_fake_hymat(os.path.join(in_dir, 'test.mat'), h=15, w=20, b=8)

            convert_hydir(in_dir, out_dir)

            bin_path = os.path.join(out_dir, 'test.bin')
            h, w, b, _ = read_hybin(bin_path)

        self.assertEqual((h, w, b), (15, 20, 8))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if '--test' in sys.argv:
        sys.argv.remove('--test')
        unittest.main(verbosity=2)
    else:
        sys.exit(main())
