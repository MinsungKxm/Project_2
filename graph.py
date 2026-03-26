"""
constellations.py is the file responsible for using the functions in graph to create a WeightedGraph object where
each vertex represents a star, connected by edges which represent constellation lines.

It also creates two dictionaries, one that maps a constellation name to the stars inside the constellation, and another
which maps the constellation's official name to information such as the meaning of the constellation, etc.

Call these lines:

g, constellations = load_constellation_graph("data/stars.csv", "data/asterisms.csv")
info = load_constellation_info("data/centered_constellations.csv")

g is the global graph that has ALL the stars & edges

constellations = {"Aql": {98036, ...}, "Ori": {...}}

info = {"Aql": {name: "Aquila", meaning: "Eagle Eye", ...}}
"""
import ast  # to convert a list represented as a string into an actual list
import csv
from graph import WeightedGraph


def load_stars(filename: str) -> WeightedGraph:
    """
    Loads the given csv file into a WeightedGraph object.
    Creates the file stars.csv by inputting hygdata_v37.csv (the raw data) by running the program.
    """
    g = WeightedGraph()

    with open(filename) as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Skip invalid rows
            if row["hip"] == "" or row["hip"] == "nan":
                continue
            hip = int(float(row["hip"]))  # convert the float values for hip ID of stars into integers
            g.add_vertex(
                item=hip,
                ra=float(row["ra"]) * 15,  # 1 hour = 15 degrees, in hygdata_v37.csv, RA is represented as hours
                dec=float(row["dec"]),
                magnitude=float(row["mag"]) if row["mag"] != "" else 0.0,  # if the magnitude is not given, assume 0
                name=row.get("proper", "")
            )

    return g


def load_constellation_graph(star_file: str, asterism_file: str) -> tuple[WeightedGraph, dict[str, set[int]]]:
    """
    Loads the given csv file into a WeightedGraph object.
    The line g = load_constellation_graph("data/stars.csv", "data/asterisms.csv") will make g represent the graph that
    connects all constellations and stars.

    The vertices are the stars from star.csv, and the edges represent the constellation lines that connects stars,
    based on asterisms.csv
    """
    # load all stars
    g = load_stars(star_file)

    constellation_map = {}  # represents a dictionary that maps a str of the constellation's name to its

    # add edges from asterisms
    with open(asterism_file) as f:
        reader = csv.DictReader(f)

        for row in reader:
            code = row["constellation"]  # ex: "Aql"
            # Convert string list → actual list
            star_list = ast.literal_eval(row["stars"])

            if code not in constellation_map:
                constellation_map[code] = set()

            # Process pairs
            for i in range(0, len(star_list)-1, 2):
                hip1 = int(star_list[i])
                hip2 = int(star_list[i + 1])

                constellation_map[code].add(hip1)
                constellation_map[code].add(hip2)
                # Only connect if both stars exist
                if g.has_vertex(hip1) and g.has_vertex(hip2):
                    g.add_edge(hip1, hip2)

    return g, constellation_map


def load_constellation_info(filename: str) -> dict[str, dict]:
    """
    Returns a dictionary that maps the star's name (ex: "Apr") to its commonly known name (ex: "Big Dipper")

    Call info = load_constellation_info("data/centered_constellations.csv")
    """
    info = {}

    with open(filename) as f:
        reader = csv.DictReader(f)

        for row in reader:
            code = row["constellation"]  # e.g. "Aql"

            info[code] = {
                "name": row["name"],
                "meaning": row["name_meaning"],
                "type": row["class"],
                "ra": float(row["ra"]),
                "dec": float(row["dec"]),
            }

    return info


def find_star_constellations(star_id: int, constellations: dict[str, set[int]]) -> list[str]:
    """
    Return a list of constellation codes that contain the given star.

    How to use it:

        star_id = 97649

        codes = find_star_constellations(star_id, constellations)

        for code in codes:
            print(info[code]["name"])

    Output ex: "Aquila"
    """
    result = []

    for code, stars in constellations.items():
        if star_id in stars:
            result.append(code)

    return result


def describe_star(star_id: int, g, constellations, info):
    """
    Returns the star's name, HIP ID, its brightness magnitude, and the constellations it is part of.

    How to use it:
        describe_star(97649, g, constellations, info)

    Output ex:
        Star: Altair (HIP 97649)
        Magnitude: 0.77
        Constellations:
         - Aquila
    """
    if not g.has_vertex(star_id):
        print("Star not found.")
        return

    data = g.get_vertex_data(star_id)
    codes = find_star_constellations(star_id, constellations)

    print(f"Star: {data['name']} (HIP {star_id})")
    print(f"Magnitude: {data['magnitude']}")

    print("Constellations:")
    for code in codes:
        print(f" - {info[code]['name']}")
