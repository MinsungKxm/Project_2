"""
process_stars.py

Generates data/stars.csv from data/asterisms.csv — no external dataset needed.

Each star in asterisms.csv already has a HIP id, RA, and Dec embedded in the
constellation line data. This script extracts every unique star from that file
and writes them to stars.csv in the format expected by constellations.py.

Run this once before running game.py or visualization.py:
    python process_stars.py
"""

import csv
import ast

INPUT_FILE  = './data/asterisms.csv'
OUTPUT_FILE = './data/stars.csv'


def generate_stars_csv() -> None:
    """Read asterisms.csv and write every unique star to stars.csv."""

    # Maps hip_id -> (ra, dec)
    stars: dict[str, tuple[float, float]] = {}

    with open(INPUT_FILE, encoding='latin-1') as f:
        reader = csv.DictReader(f)
        for row in reader:
            hip_list = ast.literal_eval(row['stars'])
            ra_list  = ast.literal_eval(row['ra'])
            dec_list = ast.literal_eval(row['dec'])

            for i, hip in enumerate(hip_list):
                hip = str(hip)
                if hip not in stars:
                    stars[hip] = (ra_list[i], dec_list[i])

    # Write stars.csv in the format constellations.py expects:
    # hip, ra (hours), dec, mag, proper
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['hip', 'ra', 'dec', 'mag', 'proper'])
        for hip, (ra, dec) in sorted(stars.items(), key=lambda x: int(x[0])):
            # ra in asterisms.csv is already in hours (0-24), not degrees
            writer.writerow([hip, ra, dec, 3.0, ''])

    print(f"Done! {len(stars)} stars written to {OUTPUT_FILE}")


if __name__ == '__main__':
    generate_stars_csv()

