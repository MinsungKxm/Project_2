"""CSC111 Winter 2026 Project: Constellation Explorer

visualization.py - Pygame Interactive Star Map

How to run:
    python visualization.py

Controls:
    - Click a star      : add it to your drawing (lines connect in order)
    - Click last star   : undo the last point
    - ENTER             : score your drawing against the real constellation
    - R                 : reset your drawing
    - Q / close window  : quit

Imports from other project files:
    constellations.py  -> load_constellation_graph, load_constellation_info
    similarity.py      -> ConstellationGraph, compute_similarity_score, score_to_percentage
"""

from __future__ import annotations
import math
import pygame

from constellations import load_constellation_graph, load_constellation_info
from similarity import ConstellationGraph, compute_similarity_score, score_to_percentage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_W = 1200
SCREEN_H = 800

BLACK     = (0,   0,   0)
WHITE     = (255, 255, 255)
DIM_WHITE = (180, 180, 180)
GREY      = (100, 100, 100)
YELLOW    = (255, 220,  40)
CYAN      = (0,   210, 255)
GREEN     = (50,  220, 100)
DIM_GREY  = (60,  60,   60)

MAX_R      = 5
MIN_R      = 1
MAG_BRIGHT = 0.0
MAG_FAINT  = 6.5

RA_CENTER  = 80.0
DEC_CENTER = 0.0
RA_SPAN    = 120.0
DEC_SPAN   = 80.0

SNAP_PX = 20


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def ra_dec_to_screen(ra: float, dec: float,
                     ra_center: float = RA_CENTER,
                     dec_center: float = DEC_CENTER) -> tuple[int, int]:
    """Map (RA, Dec) in degrees to a pixel (x, y) on the screen."""
    x = int((ra - (ra_center - RA_SPAN / 2)) / RA_SPAN * SCREEN_W)
    y = int((1.0 - (dec - (dec_center - DEC_SPAN / 2)) / DEC_SPAN) * SCREEN_H)
    return x, y


def magnitude_to_radius(mag: float) -> int:
    """Return a pixel radius proportional to brightness (lower mag = bigger dot)."""
    mag = max(MAG_BRIGHT, min(MAG_FAINT, mag))
    t = (mag - MAG_BRIGHT) / (MAG_FAINT - MAG_BRIGHT)
    return max(MIN_R, int(MAX_R - t * (MAX_R - MIN_R)))


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_star_list(g, constellation_map: dict,
                    ra_center: float = RA_CENTER,
                    dec_center: float = DEC_CENTER) -> list[dict]:
    """Return one dict per star in g, including its screen (x, y) position."""
    stars = []
    for hip in g.all_vertices():
        data = g.get_vertex_data(hip)
        sx, sy = ra_dec_to_screen(data['ra'], data['dec'], ra_center, dec_center)
        codes = [c for c, members in constellation_map.items() if hip in members]
        stars.append({
            'hip':   hip,
            'ra':    data['ra'],
            'dec':   data['dec'],
            'mag':   data['magnitude'],
            'name':  data['name'],
            'sx':    sx,
            'sy':    sy,
            'codes': codes,
        })
    return stars


def build_real_graph(code: str, constellation_map: dict, g) -> ConstellationGraph:
    """Build a similarity.ConstellationGraph for the given constellation code."""
    real = ConstellationGraph(code)
    hips = constellation_map.get(code, set())

    for hip in hips:
        if g.has_vertex(hip):
            d = g.get_vertex_data(hip)
            real.add_star(str(hip), d['ra'], d['dec'])

    for hip in hips:
        if not g.has_vertex(hip):
            continue
        for nb in g.get_neighbours(hip):
            if nb not in hips:
                continue
            id1, id2 = str(hip), str(nb)
            nodes = real.get_nodes()
            if id1 in nodes and id2 in nodes and id2 not in nodes[id1].neighbours:
                real.add_edge(id1, id2)

    return real


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_stars(screen: pygame.Surface, stars: list[dict],
               selected: list[int], hovered, font: pygame.font.Font) -> None:
    """Draw every star; highlight selected (yellow) and hovered (cyan)."""
    sel_set = set(selected)
    for s in stars:
        r = magnitude_to_radius(s['mag'])
        hip = s['hip']
        if hip == hovered:
            colour, r = CYAN, max(r, 4)
        elif hip in sel_set:
            colour, r = YELLOW, max(r, 3)
        else:
            colour = DIM_WHITE
        pygame.draw.circle(screen, colour, (s['sx'], s['sy']), r)
        if s['name'] and (hip in sel_set or hip == hovered):
            lbl = font.render(s['name'], True, CYAN)
            screen.blit(lbl, (s['sx'] + r + 3, s['sy'] - 6))


def draw_real_lines(screen: pygame.Surface, code: str,
                    constellation_map: dict, lookup: dict, g) -> None:
    """Draw the official constellation lines in dim grey for reference."""
    hips = constellation_map.get(code, set())
    drawn: set = set()
    for hip in hips:
        if not g.has_vertex(hip):
            continue
        for nb in g.get_neighbours(hip):
            if nb not in hips:
                continue
            edge = (min(hip, nb), max(hip, nb))
            if edge in drawn:
                continue
            drawn.add(edge)
            if hip in lookup and nb in lookup:
                a, b = lookup[hip], lookup[nb]
                pygame.draw.line(screen, DIM_GREY, (a['sx'], a['sy']), (b['sx'], b['sy']), 1)


def draw_user_lines(screen: pygame.Surface, selected: list[int], lookup: dict) -> None:
    """Draw yellow lines between the user clicked stars in order."""
    for i in range(len(selected) - 1):
        a, b = lookup.get(selected[i]), lookup.get(selected[i + 1])
        if a and b:
            pygame.draw.line(screen, YELLOW, (a['sx'], a['sy']), (b['sx'], b['sy']), 2)


def draw_hud(screen: pygame.Surface, score_text: str, code: str,
             info: dict, font_sm: pygame.font.Font, font_md: pygame.font.Font) -> None:
    """Draw the title, score, and control instructions."""
    hint = "Click stars to draw  |  ENTER: Score  |  R: Reset  |  Q: Quit"
    screen.blit(font_sm.render(hint, True, GREY), (10, SCREEN_H - 22))
    if code in info:
        title = f"Constellation: {info[code]['name']}  ({code})"
        screen.blit(font_md.render(title, True, WHITE), (10, 10))
    if score_text:
        screen.blit(font_md.render(score_text, True, GREEN), (10, 36))


def nearest_star(mouse: tuple, stars: list[dict]):
    """Return the HIP id of the nearest star within SNAP_PX pixels, or None."""
    mx, my = mouse
    best_hip, best_d = None, SNAP_PX
    for s in stars:
        d = math.hypot(s['sx'] - mx, s['sy'] - my)
        if d < best_d:
            best_d, best_hip = d, s['hip']
    return best_hip


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_drawing(selected: list[int], lookup: dict, real: ConstellationGraph) -> str:
    """Build a user ConstellationGraph from selected stars and return a score string."""
    if len(selected) < 2:
        return "Select at least 2 stars, then press ENTER to score!"

    user = ConstellationGraph('user')
    for hip in selected:
        if hip in lookup:
            s = lookup[hip]
            user.add_star(str(hip), s['ra'], s['dec'])

    nodes = user.get_nodes()
    for i in range(len(selected) - 1):
        id1, id2 = str(selected[i]), str(selected[i + 1])
        if id1 in nodes and id2 in nodes and id2 not in nodes[id1].neighbours:
            user.add_edge(id1, id2)

    pct = score_to_percentage(compute_similarity_score(user, real))
    return f"Your similarity score: {pct}%"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the Constellation Explorer window."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Constellation Explorer")
    clock = pygame.time.Clock()

    font_sm = pygame.font.SysFont('Arial', 13)
    font_md = pygame.font.SysFont('Arial', 18, bold=True)

    g, constellation_map = load_constellation_graph(
        "data/stars.csv", "data/asterisms.csv"
    )
    info = load_constellation_info("data/centered_constellations.csv")

    active = "Ori" if "Ori" in constellation_map else next(iter(constellation_map))

    # Auto-centre the view on the active constellation
    hips = constellation_map.get(active, set())
    ra_vals = [g.get_vertex_data(h)['ra'] for h in hips if g.has_vertex(h)]
    dec_vals = [g.get_vertex_data(h)['dec'] for h in hips if g.has_vertex(h)]
    view_ra = sum(ra_vals) / len(ra_vals) if ra_vals else RA_CENTER
    view_dec = sum(dec_vals) / len(dec_vals) if dec_vals else DEC_CENTER

    all_stars = build_star_list(g, constellation_map, view_ra, view_dec)
    lookup = {s['hip']: s for s in all_stars}
    real_graph = build_real_graph(active, constellation_map, g)

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
        draw_hud(screen, score_text, active, info, font_sm, font_md)
        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
