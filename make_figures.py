"""Compatibility entry point for the result-only revised figure pipeline.

The former renderer used historical results/tables. Its source remains in Git
history; both current entry points now require an explicit result input.
"""
from run_figures import main

if __name__ == '__main__':
    raise SystemExit(main())
