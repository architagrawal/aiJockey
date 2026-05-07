"""Verify all libs install on laptop. No GPU needed.

Usage: python scripts/00_imports_only.py
"""
import sys


def main() -> int:
    failures = []
    libs = [
        ('torch', None),
        ('torchaudio', None),
        ('demucs', None),
        ('madmom', None),
        ('librosa', None),
        ('soundfile', None),
        ('pyrubberband', None),
        ('pyloudnorm', None),
        ('scipy', None),
        ('numpy', None),
    ]
    for name, _ in libs:
        try:
            mod = __import__(name)
            ver = getattr(mod, '__version__', '?')
            print(f"  OK  {name:14s} {ver}")
        except Exception as e:
            failures.append((name, str(e)))
            print(f"FAIL  {name:14s} {e}")
    if failures:
        print(f"\n{len(failures)} import(s) failed. Fix env before proceeding.")
        return 1
    print("\nAll imports OK.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
