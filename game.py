"""CSC111 Winter 2026 Project: Constellation Explorer

game.py  - Main Game Entry Point

Controls:
    - Click stars           : draw lines between them
    - Click last star or Z  : undo last point
    - ENTER                 : score your drawing
    - R                     : reset drawing
    - G                     : toggle reference constellation graph
    - N                     : next constellation (random)
    - Q / close window      : quit

Copyright (c) 2026 Jenny Lin and Project Group
"""
from __future__ import annotations
import doctest
import python_ta
import random
import pygame

from constellations import load_constellation_graph, load_constellation_info
from similarity import ConstellationGraph, compute_similarity_score, score_to_percentage
from star_hopping import Star, build_hoppable_graph, find_easiest_path
from visualization import (
    build_star_list, build_real_graph,
    draw_stars, draw_real_lines, draw_user_lines, draw_hud,
    nearest_star, score_drawing,
    SCREEN_W, SCREEN_H, BLACK, GREY, WHITE, GREEN,
    RA_CENTER, DEC_CENTER, RA_SPAN, DEC_SPAN
)


def pick_constellation(constellation_map: dict, info: dict) -> str:
    """Return a random constellation code that exists in both constellation_map and info."""
    valid = [code for code in constellation_map if code in info]
    return random.choice(valid) if valid else next(iter(constellation_map))


def draw_game_hud(screen: pygame.Surface, score_text: str, code: str,
                  info: dict, font_sm: pygame.font.Font, font_md: pygame.font.Font,
                  mode: str, show_graph: bool) -> None:
    """Draw the HUD with key hints, constellation name, meaning, and score.

    Shows different key hints depending on whether the player is in draw mode
    or star-hopping mode, and whether the reference graph is currently visible.

    Parameters:
        - screen: the pygame surface to draw onto
        - score_text: the most recent score message, or '' if none yet
        - code: the active constellation code (e.g. 'Ori')
        - info: mapping from constellation code to metadata dict
        - font_sm: small font used for hints and meaning text
        - font_md: medium bold font used for title and score
        - mode: either 'draw' or 'hop'
        - show_graph: whether the reference constellation lines are visible
    """
    graph_hint = "G: Hide graph" if show_graph else "G: Show graph"
    if mode == "hop":
        hint = f"Hop mode: click START then TARGET | ENTER: recompute | R: Reset | H: Toggle | N: Next | {graph_hint} | Q: Quit"
    else:
        hint = f"Left-click: draw  |  Right-click: lift pen  |  Z: undo  |  ENTER: Score  |  R: Reset  |  N: Next  |  {graph_hint}  |  Q: Quit"

    screen.blit(font_sm.render(hint, True, GREY), (10, SCREEN_H - 22))
    if code in info:
        title = f"Draw this constellation: {info[code]['name']}  ({code})"
        screen.blit(font_md.render(title, True, WHITE), (10, 10))
        meaning = info[code].get('meaning', '')
        if meaning:
            screen.blit(font_sm.render(f"Meaning: {meaning}", True, GREY), (10, 34))
    if score_text:
        screen.blit(font_md.render(score_text, True, GREEN), (10, 56))


def constellation_center(code: str, constellation_map: dict, g) -> tuple[float, float, float, float]:
    """Return view centre (RA, Dec) and span (ra_span, dec_span) for a constellation.

    Handles RA wraparound (stars near 0°/360°) and auto-zooms so every
    constellation fits on screen with padding.
    """
    hips = constellation_map.get(code, set())
    ra_vals = [g.get_vertex_data(h)['ra'] for h in hips if g.has_vertex(h)]
    dec_vals = [g.get_vertex_data(h)['dec'] for h in hips if g.has_vertex(h)]

    if not ra_vals:
        return RA_CENTER, DEC_CENTER, RA_SPAN, DEC_SPAN

    # Detect wraparound: if naive span > 180°, shift stars > 180° down by 360°
    naive_span = max(ra_vals) - min(ra_vals)
    if naive_span > 180.0:
        ra_vals = [r - 360.0 if r > 180.0 else r for r in ra_vals]

    ra_center = (max(ra_vals) + min(ra_vals)) / 2
    dec_center = (max(dec_vals) + min(dec_vals)) / 2

    # Add 50% padding so stars don't sit at the very edge
    ra_needed = (max(ra_vals) - min(ra_vals)) * 1.5 + 20
    dec_needed = (max(dec_vals) - min(dec_vals)) * 1.5 + 20

    # Minimum zoom so small constellations aren't too zoomed in
    ra_span = max(ra_needed, 40.0)
    dec_span = max(dec_needed, 30.0)

    # Keep the same aspect ratio as the screen to avoid distortion
    if ra_span / dec_span < 1.5:
        ra_span = dec_span * 1.5
    else:
        dec_span = ra_span / 1.5

    return ra_center, dec_center, ra_span, dec_span


def main(start_mode: str = "draw") -> None:
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

    # Build the star-hopping graph once (used in hop mode).
    hop_graph = build_hoppable_graph([
        Star(
            star_id=str(hip),
            ra=g.get_vertex_data(hip)['ra'],
            dec=g.get_vertex_data(hip)['dec'],
            magnitude=g.get_vertex_data(hip)['magnitude'],
            name=g.get_vertex_data(hip)['name'],
        )
        for hip in g.all_vertices()
    ])

    # Pick starting constellation
    active = pick_constellation(constellation_map, info)
    real_graph = build_real_graph(active, constellation_map, g)

    # Centre view on starting constellation
    view_ra, view_dec, view_ra_span, view_dec_span = constellation_center(active, constellation_map, g)
    all_stars = build_star_list(g, constellation_map, view_ra, view_dec, view_ra_span, view_dec_span)
    lookup = {s['hip']: s for s in all_stars}

    mode = start_mode if start_mode in ("draw", "hop") else "draw"

    # strokes: list of strokes; each stroke is a list of HIP ids.
    # Right-click lifts the pen and starts a new stroke.
    strokes: list[list[int]] = [[]]
    hop_selection: list[int] = []  # [start] or [start, target]
    hop_route: list[int] = []       # full path returned by Dijkstra
    hovered = None
    score_text = ""
    show_graph = False  # G key toggles the reference constellation lines

    running = True
    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

                elif event.key == pygame.K_h:
                    # Toggle between constellation-building and star-hopping.
                    if mode == "draw":
                        mode = "hop"
                        strokes, score_text = [[]], ""
                        hop_selection, hop_route = [], []
                    else:
                        mode = "draw"
                        strokes, score_text = [[]], ""
                        hop_selection, hop_route = [], []

                elif event.key == pygame.K_r:
                    if mode == "draw":
                        strokes, score_text = [[]], ""
                    else:
                        hop_selection, hop_route, score_text = [], [], ""

                elif event.key == pygame.K_z:
                    if mode == "draw":
                        # Undo: remove last star; if current stroke is empty, remove it too
                        while strokes and not strokes[-1]:
                            if len(strokes) > 1:
                                strokes.pop()
                            else:
                                break
                        if strokes and strokes[-1]:
                            strokes[-1].pop()
                        score_text = ""

                elif event.key == pygame.K_RETURN:
                    if mode == "draw":
                        score_text = score_drawing(strokes, lookup, real_graph)
                    elif len(hop_selection) == 2 and hop_selection[0] != hop_selection[1]:
                        start, target = hop_selection[0], hop_selection[1]
                        path, cost = find_easiest_path(hop_graph, str(start), str(target))
                        if path is None:
                            hop_route = []
                            score_text = "No reachable hop path found."
                        else:
                            hop_route = [int(pid) for pid in path]
                            score_text = f"Easiest hop difficulty: {cost:.2f}"

                elif event.key == pygame.K_n:
                    # Pick a new random constellation and re-centre the view
                    active = pick_constellation(constellation_map, info)
                    real_graph = build_real_graph(active, constellation_map, g)
                    view_ra, view_dec, view_ra_span, view_dec_span = constellation_center(active, constellation_map, g)
                    all_stars = build_star_list(g, constellation_map, view_ra, view_dec, view_ra_span, view_dec_span)
                    lookup = {s['hip']: s for s in all_stars}
                    strokes, score_text = [[]], ""
                    hop_selection, hop_route = [], []

                elif event.key == pygame.K_g:
                    # Toggle the reference constellation graph on/off
                    show_graph = not show_graph

            elif event.type == pygame.MOUSEMOTION:
                hovered = nearest_star(event.pos, all_stars)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked = nearest_star(event.pos, all_stars)
                if clicked is not None:
                    if mode == "draw":
                        cur = strokes[-1]
                        if cur and cur[-1] == clicked:
                            cur.pop()  # undo last point in current stroke
                        else:
                            cur.append(clicked)
                        score_text = ""
                    else:
                        # Hop mode: first click = START, second click = TARGET
                        if len(hop_selection) == 0:
                            hop_selection = [clicked]
                            hop_route = []
                            score_text = ""
                        elif len(hop_selection) == 1:
                            start = hop_selection[0]
                            if clicked == start:
                                hop_selection = []
                                hop_route = []
                                score_text = ""
                            else:
                                hop_selection = [start, clicked]
                                hop_route = []
                                path, cost = find_easiest_path(hop_graph, str(start), str(clicked))
                                if path is None:
                                    hop_route = []
                                    score_text = "No reachable hop path found."
                                else:
                                    hop_route = [int(pid) for pid in path]
                                    score_text = f"Easiest hop difficulty: {cost:.2f}"
                        else:
                            hop_selection = [clicked]
                            hop_route = []
                            score_text = ""

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                if mode == "draw":
                    # Right-click: lift pen — start a new stroke
                    strokes.append([])
                    score_text = ""

        all_selected = [hip for stroke in strokes for hip in stroke]
        screen.fill(BLACK)
        if show_graph:
            draw_real_lines(screen, active, constellation_map, lookup, g)

        if mode == "hop":
            draw_user_lines(screen, [hop_route] if hop_route else [hop_selection], lookup)
            draw_stars(screen, all_stars, hop_route if hop_route else hop_selection, hovered, font_sm)
        else:
            draw_user_lines(screen, strokes, lookup)
            draw_stars(screen, all_stars, all_selected, hovered, font_sm)

        draw_game_hud(screen, score_text, active, info, font_sm, font_md, mode, show_graph)
        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    doctest.testmod()
    python_ta.check_all(config={
        'extra-imports': ['random', 'pygame', 'constellations', 'similarity',
                          'star_hopping', 'visualization'],
        'allowed-io': [],
        'max-line-length': 120
    })
