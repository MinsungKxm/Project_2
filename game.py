"""CSC111 Winter 2026 Project: Constellation Explorer

game.py - Main Game Entry Point

This is the file you run to start the game. It:
    1. Loads all star and constellation data
    2. Picks a random constellation for the player to draw
    3. Launches the pygame visualization window
    4. Shows the player's score when they press ENTER

How to run:
    python game.py

Controls:
    - Click stars       : draw lines between them
    - Click last star   : undo last point
    - ENTER             : score your drawing
    - R                 : reset drawing
    - N                 : next constellation (random)
    - Q / close window  : quit
"""

from __future__ import annotations
import random
import pygame

from constellations import load_constellation_graph, load_constellation_info
from similarity import ConstellationGraph, compute_similarity_score, score_to_percentage
from visualization import (
    build_star_list, build_real_graph,
    draw_stars, draw_real_lines, draw_user_lines, draw_hud,
    nearest_star, score_drawing,
    SCREEN_W, SCREEN_H, BLACK,
    RA_CENTER, DEC_CENTER
)


def pick_constellation(constellation_map: dict, info: dict) -> str:
    """Return a random constellation code that exists in both constellation_map and info."""
    valid = [code for code in constellation_map if code in info]
    return random.choice(valid) if valid else next(iter(constellation_map))


def draw_game_hud(screen: pygame.Surface, score_text: str, code: str,
                  info: dict, font_sm: pygame.font.Font, font_md: pygame.font.Font) -> None:
    """Draw HUD with extra N key hint for next constellation."""
    from visualization import GREY, WHITE, GREEN
    hint = "Click stars to draw  |  ENTER: Score  |  R: Reset  |  N: Next  |  Q: Quit"
    screen.blit(font_sm.render(hint, True, GREY), (10, SCREEN_H - 22))
    if code in info:
        title = f"Draw this constellation: {info[code]['name']}  ({code})"
        screen.blit(font_md.render(title, True, WHITE), (10, 10))
        meaning = info[code].get('meaning', '')
        if meaning:
            screen.blit(font_sm.render(f"Meaning: {meaning}", True, GREY), (10, 34))
    if score_text:
        from visualization import GREEN
        screen.blit(font_md.render(score_text, True, GREEN), (10, 56))


def constellation_center(code: str, constellation_map: dict, g) -> tuple[float, float]:
    """Return the mean RA and Dec of a constellation's stars."""
    hips = constellation_map.get(code, set())
    ra_vals = [g.get_vertex_data(h)['ra'] for h in hips if g.has_vertex(h)]
    dec_vals = [g.get_vertex_data(h)['dec'] for h in hips if g.has_vertex(h)]
    ra = sum(ra_vals) / len(ra_vals) if ra_vals else RA_CENTER
    dec = sum(dec_vals) / len(dec_vals) if dec_vals else DEC_CENTER
    return ra, dec


def main() -> None:
    """Run the Constellation Explorer game."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Constellation Explorer")
    clock = pygame.time.Clock()

    font_sm = pygame.font.SysFont('Arial', 13)
    font_md = pygame.font.SysFont('Arial', 18, bold=True)

    # Load all data
    g, constellation_map = load_constellation_graph(
        "data/stars.csv", "data/asterisms.csv"
    )
    info = load_constellation_info("data/centered_constellations.csv")

    # Build star list and lookup once
    all_stars = build_star_list(g, constellation_map)
    lookup = {s['hip']: s for s in all_stars}

    # Pick starting constellation
    active = pick_constellation(constellation_map, info)
    real_graph = build_real_graph(active, constellation_map, g)

    # Centre view on starting constellation
    view_ra, view_dec = constellation_center(active, constellation_map, g)
    all_stars = build_star_list(g, constellation_map, view_ra, view_dec)
    lookup = {s['hip']: s for s in all_stars}

    selected: list[int] = []
    hovered = None
    score_text = ""

    running = True
    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

                elif event.key == pygame.K_r:
                    selected, score_text = [], ""

                elif event.key == pygame.K_RETURN:
                    score_text = score_drawing(selected, lookup, real_graph)

                elif event.key == pygame.K_n:
                    # Pick a new random constellation and re-centre the view
                    active = pick_constellation(constellation_map, info)
                    real_graph = build_real_graph(active, constellation_map, g)
                    view_ra, view_dec = constellation_center(active, constellation_map, g)
                    all_stars = build_star_list(g, constellation_map, view_ra, view_dec)
                    lookup = {s['hip']: s for s in all_stars}
                    selected, score_text = [], ""

            elif event.type == pygame.MOUSEMOTION:
                hovered = nearest_star(event.pos, all_stars)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked = nearest_star(event.pos, all_stars)
                if clicked is not None:
                    if selected and selected[-1] == clicked:
                        selected.pop()
                    else:
                        selected.append(clicked)
                    score_text = ""

        screen.fill(BLACK)
        draw_real_lines(screen, active, constellation_map, lookup, g)
        draw_user_lines(screen, selected, lookup)
        draw_stars(screen, all_stars, selected, hovered, font_sm)
        draw_game_hud(screen, score_text, active, info, font_sm, font_md)
        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
