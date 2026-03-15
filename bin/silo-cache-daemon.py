#!/usr/bin/env python3
"""Lease-aware daemon for the local hashall qB shared cache."""

from hashall.qb_cache import daemon_main


if __name__ == "__main__":
    raise SystemExit(daemon_main())
