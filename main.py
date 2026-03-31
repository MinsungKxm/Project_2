"""CSC111 Winter 2026 Project: Constellation Explorer

main.py - Program entry point

This file exists to run the entire project with:
    python main.py

It starts the interactive pygame application defined in game.py (load CSV
datasets, display the sky map, score the user's constellation drawing).
"""

from __future__ import annotations

import sys
from game import main as run_game


if __name__ == "__main__":
    start_mode = "draw"
    if "--hop" in sys.argv or "--star-hop" in sys.argv:
        start_mode = "hop"
    run_game(start_mode)