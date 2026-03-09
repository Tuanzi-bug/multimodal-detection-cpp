"""
convert_mat_to_bin.py
=====================
Convert spectral library .mat files to binary .bin for C++ reading.

Source data layout (each .mat file):
    data = scipy.io.loadmat(path)['data']   # shape (131, 129), dtype float64
    row 0      : mu            (1×129 background mean vector)
    rows 1~129 : sqrtDxinvU   (129×129 whitening matrix)
    row 130    : signature     (1×129 target spectral signature)

Binary format written:
    [uint32_t rows][uint32_t cols][uint32_t dtype_size=8][float64 row-major data]

Usage:
    python c++/tools/convert_mat_to_bin.py [--input-dir DIR] [--output-dir DIR] [--verify]

Defaults:
    --input-dir  spectral_lib/          (relative to cwd / project root)
    --output-dir c++/data/spectral_lib/ (relative to cwd / project root)

The script processes ALL .mat files found in the five class subdirectories:
    aircraft/, car/, oiltank/, roof/, ship/

With --verify, each .bin is round-trip verified: write → read back → compare
(max element-wise difference must be < 1e-15).

Run doctests:
    python -m doctest c++/tools/convert_mat_to_bin.py -v

Run unittests:
    python -m pytest c++/tools/convert_mat_to_bin.py

"""

import argparse
import os
import struct
import sys
import io
import unittest
import tempfile


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSES = ('aircraft', 'car', 'oiltank', 'roof', 'ship')
DTYPE_SIZE = 8          # float64 → 8 bytes
HEADER_FMT = '<III'     # little-endian: uint32 rows, uint32 cols, uint32 dtype_size
HEADER_BYTES = struct.calcsize(HEADER_FMT)   # 12 bytes
MAX_VERIFY_DIFF = 1e-15


# ---------------------------------------------------------------------------
# Core I/O helpers
# ---------------------------------------------------------------------------

def encode_header(rows: int, cols: int, dtype_size: int = DTYPE_SIZE) -> bytes:
    """Return the 12-byte binary header.

    >>> encode_header(131, 129)
    b'\\x83\\x00\\x00\\x00\\x81\\x00\\x00\\x00\\x08\\x00\\x00\\x00'
    >>> len(encode_header(1, 1))
    12
    """
    return struct.pack(HEADER_FMT, rows, cols, dtype_size)


def decode_header(data: bytes) -> tuple:
    """Return (rows, cols, dtype_size) decoded from a 12-byte header.

    >>> decode_header(encode_header(131, 129))
    (131, 129, 8)
    >>> decode_header(encode_header(3, 7, 8))
    (3, 7, 8)
    """
    if len(data) < HEADER_BYTES:
        raise ValueError(
            f'Header too short: expected {HEADER_BYTES} bytes, got {len(data)}'
        )
    rows, cols, dtype_size = struct.unpack(HEADER_FMT, data[:HEADER_BYTES])
    return rows, cols, dtype_size


def write_bin(array, dest_path: str) -> None:
    """Write a 2-D float64 numpy array to a .bin file.

    Format: [uint32 rows][uint32 cols][uint32 dtype_size=8][float64 row-major]

    Args:
        array:     2-D numpy array with dtype float64.
        dest_path: Absolute or relative path for the output .bin file.

    Raises:
        ValueError: If array is not 2-D or not float64.
        OSError:    If the destination path cannot be written.
    """
    import numpy as np
    if array.ndim != 2:
        raise ValueError(f'Expected 2-D array, got {array.ndim}-D')
    arr64 = np.asarray(array, dtype=np.float64)
    rows, cols = arr64.shape
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    with open(dest_path, 'wb') as fh:
        fh.write(encode_header(rows, cols))
        fh.write(arr64.tobytes(order='C'))


def read_bin(src_path: str):
    """Read a .bin file produced by write_bin and return a (rows, cols, array) tuple.

    Args:
        src_path: Path to the .bin file.

    Returns:
        (rows, cols, numpy.ndarray) where array has shape (rows, cols) and
        dtype float64.

    Raises:
        FileNotFoundError: If src_path does not exist.
        ValueError:        If the file is shorter than the header + data size.
    """
    import numpy as np
    with open(src_path, 'rb') as fh:
        header = fh.read(HEADER_BYTES)
        rows, cols, dtype_size = decode_header(header)
        expected_bytes = rows * cols * dtype_size
        raw = fh.read(expected_bytes)
    if len(raw) < expected_bytes:
        raise ValueError(
            f'Truncated data in {src_path}: '
            f'expected {expected_bytes} bytes, got {len(raw)}'
        )
    arr = np.frombuffer(raw, dtype=np.float64).reshape(rows, cols)
    return rows, cols, arr


def validate_mat_data(data, path: str) -> None:
    """Validate shape constraints for a spectral library matrix.

    Args:
        data: 2-D numpy array (rows × cols).
        path: Source path used only for error messages.

    Raises:
        ValueError: If shape constraints are violated.
    """
    if data.ndim != 2:
        raise ValueError(f'{path}: expected 2-D data, got shape {data.shape}')
    rows, cols = data.shape
    if rows < 3:
        raise ValueError(
            f'{path}: rows must be >= 3 (got {rows}); '
            'need at least mu, one matrix row, and signature'
        )
    if cols <= 0:
        raise ValueError(f'{path}: cols must be > 0 (got {cols})')


# ---------------------------------------------------------------------------
# Conversion logic
# ---------------------------------------------------------------------------

def load_mat(path: str):
    """Load a spectral library .mat file and return the 'data' array.

    Args:
        path: Path to the .mat file.

    Returns:
        numpy.ndarray with dtype float64, shape (rows, cols).

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError:          If the .mat file has no 'data' key.
        ValueError:        If the shape constraints are not met.
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
    data = mat['data']
    import numpy as np
    data = np.array(data, dtype=np.float64)
    validate_mat_data(data, path)
    return data


def convert_file(mat_path: str, bin_path: str, verify: bool = False) -> dict:
    """Convert one .mat file to .bin.

    Args:
        mat_path: Source .mat file path.
        bin_path: Destination .bin file path.
        verify:   If True, round-trip verify after writing.

    Returns:
        dict with keys: class_name, filename, shape, status, error.
    """
    import numpy as np
    filename = os.path.basename(mat_path)
    class_name = os.path.basename(os.path.dirname(mat_path))
    result = {
        'class_name': class_name,
        'filename': filename,
        'shape': None,
        'status': 'OK',
        'error': None,
    }

    try:
        data = load_mat(mat_path)
        result['shape'] = data.shape
        write_bin(data, bin_path)

        if verify:
            rows_r, cols_r, arr_r = read_bin(bin_path)
            max_diff = float(np.max(np.abs(data - arr_r)))
            if max_diff >= MAX_VERIFY_DIFF:
                raise ValueError(
                    f'Round-trip max diff {max_diff:.2e} exceeds '
                    f'threshold {MAX_VERIFY_DIFF:.2e}'
                )
            result['status'] = f'OK (verified, max_diff={max_diff:.2e})'

    except Exception as exc:
        result['status'] = 'FAIL'
        result['error'] = str(exc)

    return result


def convert_directory(
    input_dir: str,
    output_dir: str,
    verify: bool = False,
) -> list:
    """Convert all .mat files in spectral_lib subdirectories.

    Args:
        input_dir:  Root spectral library directory containing class subdirs.
        output_dir: Root output directory; class subdirs are created automatically.
        verify:     If True, run round-trip verification on each output file.

    Returns:
        List of result dicts (one per .mat file processed or attempted).
    """
    results = []

    if not os.path.isdir(input_dir):
        print(f'WARNING: input directory not found: {input_dir}', file=sys.stderr)
        return results

    for class_name in CLASSES:
        class_in  = os.path.join(input_dir,  class_name)
        class_out = os.path.join(output_dir, class_name)

        if not os.path.isdir(class_in):
            results.append({
                'class_name': class_name,
                'filename': '—',
                'shape': None,
                'status': 'SKIP',
                'error': f'Directory not found: {class_in}',
            })
            continue

        mat_files = sorted(
            f for f in os.listdir(class_in) if f.lower().endswith('.mat')
        )
        if not mat_files:
            results.append({
                'class_name': class_name,
                'filename': '—',
                'shape': None,
                'status': 'SKIP',
                'error': f'No .mat files in {class_in}',
            })
            continue

        os.makedirs(class_out, exist_ok=True)

        for fname in mat_files:
            mat_path = os.path.join(class_in, fname)
            bin_name = os.path.splitext(fname)[0] + '.bin'
            bin_path = os.path.join(class_out, bin_name)
            result = convert_file(mat_path, bin_path, verify=verify)
            results.append(result)

    return results


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------

def print_summary(results: list) -> None:
    """Print a formatted summary table and return the number of failures."""
    ok_count   = sum(1 for r in results if r['status'].startswith('OK'))
    fail_count = sum(1 for r in results if r['status'] == 'FAIL')
    skip_count = sum(1 for r in results if r['status'] == 'SKIP')

    # Column widths
    cls_w  = max(9, max((len(r['class_name']) for r in results), default=0))
    file_w = max(8, max((len(r['filename'])   for r in results), default=0))
    shp_w  = 12
    sts_w  = max(6, max((len(r['status'])     for r in results), default=0))

    header = (
        f"{'Class':<{cls_w}}  {'Filename':<{file_w}}  "
        f"{'Shape':<{shp_w}}  {'Status':<{sts_w}}"
    )
    print()
    print(header)
    print('-' * len(header))

    for r in results:
        shape_str = (
            f"{r['shape'][0]}x{r['shape'][1]}" if r['shape'] else '—'
        )
        print(
            f"{r['class_name']:<{cls_w}}  {r['filename']:<{file_w}}  "
            f"{shape_str:<{shp_w}}  {r['status']:<{sts_w}}"
        )
        if r['error']:
            print(f"  ERROR: {r['error']}")

    print()
    print(f'Summary: {ok_count} OK, {fail_count} FAIL, {skip_count} SKIP')
    return fail_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            'Convert spectral library .mat files to binary .bin '
            'for C++ reading.'
        )
    )
    parser.add_argument(
        '--input-dir',
        default='spectral_lib',
        help='Root spectral library directory (default: spectral_lib/)',
    )
    parser.add_argument(
        '--output-dir',
        default=os.path.join('c++', 'data', 'spectral_lib'),
        help='Root output directory (default: c++/data/spectral_lib/)',
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Round-trip verify each .bin file after writing',
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
    args = parser.parse_args(argv)

    print(f'Input  dir : {args.input_dir}')
    print(f'Output dir : {args.output_dir}')
    print(f'Verify     : {args.verify}')

    results   = convert_directory(args.input_dir, args.output_dir, args.verify)
    fail_count = print_summary(results)

    return 1 if fail_count > 0 else 0


# ---------------------------------------------------------------------------
# Tests (unittest)
# ---------------------------------------------------------------------------

class TestEncodeDecodeHeader(unittest.TestCase):
    """Unit tests for binary header encoding and decoding."""

    def test_roundtrip_standard(self):
        """encode then decode returns the original values."""
        header = encode_header(131, 129)
        rows, cols, dtype_size = decode_header(header)
        self.assertEqual(rows, 131)
        self.assertEqual(cols, 129)
        self.assertEqual(dtype_size, 8)

    def test_header_length(self):
        """Header is exactly 12 bytes."""
        self.assertEqual(len(encode_header(1, 1)), 12)

    def test_decode_too_short(self):
        """decode_header raises ValueError for short input."""
        with self.assertRaises(ValueError):
            decode_header(b'\x00' * 8)

    def test_little_endian(self):
        """Header is stored in little-endian byte order."""
        header = encode_header(1, 1)
        # uint32 value 1 in little-endian is b'\x01\x00\x00\x00'
        self.assertEqual(header[:4], b'\x01\x00\x00\x00')


class TestWriteReadBin(unittest.TestCase):
    """Integration tests for write_bin / read_bin round-trip."""

    def _make_array(self, rows=5, cols=3):
        import numpy as np
        rng = np.random.default_rng(seed=42)
        return np.asarray(rng.random((rows, cols)), dtype=np.float64)

    def test_roundtrip_exact(self):
        """Write then read produces bit-identical array."""
        import numpy as np
        arr = self._make_array(131, 129)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.bin')
            write_bin(arr, path)
            rows_r, cols_r, arr_r = read_bin(path)
        self.assertEqual(rows_r, 131)
        self.assertEqual(cols_r, 129)
        self.assertTrue(np.array_equal(arr, arr_r))

    def test_roundtrip_max_diff(self):
        """Max element-wise difference after round-trip is below 1e-15."""
        import numpy as np
        arr = self._make_array(10, 7)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.bin')
            write_bin(arr, path)
            _, _, arr_r = read_bin(path)
        max_diff = float(np.max(np.abs(arr - arr_r)))
        self.assertLess(max_diff, 1e-15)

    def test_write_creates_parent_dirs(self):
        """write_bin creates parent directories automatically."""
        import numpy as np
        arr = self._make_array(3, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'a', 'b', 'c', 'test.bin')
            write_bin(arr, path)
            self.assertTrue(os.path.isfile(path))

    def test_write_rejects_non_2d(self):
        """write_bin raises ValueError for non-2D array."""
        import numpy as np
        arr3d = np.zeros((2, 3, 4), dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                write_bin(arr3d, os.path.join(tmpdir, 'bad.bin'))

    def test_file_size(self):
        """Output .bin file size matches header + data exactly."""
        import numpy as np
        rows, cols = 131, 129
        arr = self._make_array(rows, cols)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.bin')
            write_bin(arr, path)
            expected = HEADER_BYTES + rows * cols * DTYPE_SIZE
            actual   = os.path.getsize(path)
        self.assertEqual(actual, expected)


class TestValidateMatData(unittest.TestCase):
    """Tests for shape validation of spectral library matrices."""

    def _make(self, rows, cols):
        import numpy as np
        return np.zeros((rows, cols), dtype=np.float64)

    def test_valid_shape(self):
        """Valid shape (131, 129) does not raise."""
        validate_mat_data(self._make(131, 129), 'dummy.mat')

    def test_minimum_valid_rows(self):
        """Exactly 3 rows is valid."""
        validate_mat_data(self._make(3, 1), 'dummy.mat')

    def test_too_few_rows(self):
        """Fewer than 3 rows raises ValueError."""
        with self.assertRaises(ValueError):
            validate_mat_data(self._make(2, 5), 'dummy.mat')

    def test_zero_rows(self):
        """Zero rows raises ValueError."""
        with self.assertRaises(ValueError):
            validate_mat_data(self._make(0, 5), 'dummy.mat')


class TestConvertDirectory(unittest.TestCase):
    """Integration tests using synthetic .mat files."""

    def _write_fake_mat(self, path, rows=131, cols=129):
        """Write a minimal fake .mat file using scipy if available."""
        try:
            import scipy.io as scio
            import numpy as np
            os.makedirs(os.path.dirname(path), exist_ok=True)
            rng = np.random.default_rng(seed=0)
            data = rng.random((rows, cols))
            scio.savemat(path, {'data': data})
            return data
        except ImportError:
            self.skipTest('scipy not available')

    def test_empty_directory(self):
        """Empty input directory returns empty results list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = convert_directory(
                os.path.join(tmpdir, 'nonexistent_input'),
                os.path.join(tmpdir, 'output'),
            )
        self.assertEqual(results, [])

    def test_missing_class_subdir(self):
        """Missing class subdir is reported as SKIP."""
        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = os.path.join(tmpdir, 'spectral_lib')
            os.makedirs(in_dir)
            # Create only 'aircraft' subdir, leave others missing
            os.makedirs(os.path.join(in_dir, 'aircraft'))
            results = convert_directory(in_dir, os.path.join(tmpdir, 'out'))
        skip_classes = [r['class_name'] for r in results if r['status'] == 'SKIP']
        self.assertIn('car', skip_classes)
        self.assertIn('ship', skip_classes)

    def test_invalid_mat_missing_data_key(self):
        """A .mat file without 'data' key is reported as FAIL."""
        try:
            import scipy.io as scio
            import numpy as np
        except ImportError:
            self.skipTest('scipy not available')

        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir  = os.path.join(tmpdir, 'spectral_lib', 'aircraft')
            out_dir = os.path.join(tmpdir, 'out')
            os.makedirs(in_dir)
            bad_mat = os.path.join(in_dir, 'bad.mat')
            scio.savemat(bad_mat, {'wrong_key': np.zeros((131, 129))})

            results = convert_directory(
                os.path.join(tmpdir, 'spectral_lib'), out_dir
            )

        aircraft_results = [r for r in results if r['class_name'] == 'aircraft']
        self.assertEqual(len(aircraft_results), 1)
        self.assertEqual(aircraft_results[0]['status'], 'FAIL')

    def test_valid_mat_converted(self):
        """A valid .mat file is converted to .bin and round-trip verified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir  = os.path.join(tmpdir, 'spectral_lib', 'aircraft')
            out_dir = os.path.join(tmpdir, 'out')
            mat_path = os.path.join(in_dir, 'aircraft1.mat')
            data = self._write_fake_mat(mat_path)

            results = convert_directory(
                os.path.join(tmpdir, 'spectral_lib'), out_dir, verify=True
            )

        aircraft_results = [r for r in results if r['class_name'] == 'aircraft']
        self.assertEqual(len(aircraft_results), 1)
        self.assertTrue(aircraft_results[0]['status'].startswith('OK'))

    def test_output_structure_matches_input(self):
        """Output directory mirrors the input class structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for cls in ('aircraft', 'car'):
                in_dir = os.path.join(tmpdir, 'spectral_lib', cls)
                self._write_fake_mat(os.path.join(in_dir, f'{cls}1.mat'))

            out_root = os.path.join(tmpdir, 'out')
            convert_directory(
                os.path.join(tmpdir, 'spectral_lib'), out_root
            )

            self.assertTrue(os.path.isdir(os.path.join(out_root, 'aircraft')))
            self.assertTrue(os.path.isdir(os.path.join(out_root, 'car')))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # If --test is passed, run unittests; otherwise run the CLI.
    if '--test' in sys.argv:
        sys.argv.remove('--test')
        unittest.main(verbosity=2)
    else:
        sys.exit(main())
