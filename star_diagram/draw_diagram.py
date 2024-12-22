import numpy as np
from PIL import Image, ImageDraw
from .trig_math import *


def blend_color(color_a, color_b, factor=0.5):
    diff = (b - a for a, b in zip(color_a, color_b))
    return tuple(round(a + d * factor) for a, d in zip(color_a, diff))


def draw_circle(draw, center, radius=20, fill=(0, 200, 100)):
    left_up = (center[0] - radius, center[1] - radius)
    right_down = (center[0] + radius, center[1] + radius)
    draw.ellipse([left_up, right_down], fill=fill)


def draw_line(draw, start: tuple, end: tuple, pattern: str, fill: tuple, width):
    # rise = end[1] - start[1]
    # run = end[0] - start[0]

    # slope == rise / run if run != 0 else 1000000
    # TODO: implement patterns. "solid", "dashed", "dotted", etc.
    draw.line((start[0], start[1], end[0], end[1]), fill=fill, width=width)


MAIN_LINE_COLOR = (0, 0, 0)
ORTHO_LINE_COLOR = (128, 128, 128)
BEAM_LINE_COLOR = (233, 233, 233)

TR_COLOR = (100, 60, 200)
TL_COLOR = (200, 60, 100)
BL_COLOR = (100, 200, 60)
BR_COLOR = (200, 170, 30)

CORNER_DOT_RADIUS_IN = 1 / 2
FOLD_DOT_RADIUS_IN = 1 / 12
FOLD_DOT_OUTLINE_IN = 1 / 32

MAIN_LINE_WIDTH_IN = 1 / 24
HINT_LINE_WIDTH_IN = 1 / 64
BEAM_LINE_WIDTH_IN = 1 / 24

RGB_FADE = 0.7  # 0 for full color on hint lines, 1 for all white.
ORTHO_RGB_FADE = 0.2

DPI = 300


def create_star_diagram(
    num_sides: int = 5,
    print_width=8.5,
    print_height=11,
    print_margin_left=0.125,
    print_margin_right=0.125,
    print_margin_top=0.125,
    print_margin_bottom=0.125,
    poly_height=4.0625,
    is_metric: bool = False,
):
    """
    Writes an origami diagram to file that can be used to quickly
    turn the piece of paper into a 3D star.

    Parameters:
        num_sides (int): the number of sides the star will have.
        print_width (num): the width of the paper (in inches/cm).
        print_height (num): the height of the paper (in inches/cm).
        print_margin_left (num): the length cut off the left side of the output.
        print_margin_right (num): the length cut off the right side of the output.
        print_margin_top (num): the length cut off the top side of the output.
        print_margin_bottom (num): the length cut off the bottom side of the output.
        poly_height (num): the distance from the top to bottom of the polygon.
        is_metric (bool): if True, all given measurements are treated as cm.
    """

    """
    Step 1) Initializes print and diagram settings,
            then creates Pillow images to draw the diagram.
    """
    # gets print settings in terms of inches.
    print_width_in = print_width / 2.54 if is_metric else print_width
    print_height_in = print_height / 2.54 if is_metric else print_height
    print_margin_left_in = print_margin_left / 2.54 if is_metric else print_margin_left
    print_margin_right_in = (
        print_margin_right / 2.54 if is_metric else print_margin_right
    )
    print_margin_top_in = print_margin_top / 2.54 if is_metric else print_margin_top
    print_margin_bottom_in = (
        print_margin_bottom / 2.54 if is_metric else print_margin_bottom
    )

    # diagram settings.
    poly_height_in = poly_height / 2.54 if is_metric else poly_height
    help_length_in = min(print_width_in, print_height_in) / 5

    # converts the measurements into lengths in the unit of pixels.
    w, h = round(print_width_in * DPI), round(print_height_in * DPI)
    print_margin_left = print_margin_left_in * DPI
    print_margin_right = print_margin_right_in * DPI
    print_margin_top = print_margin_top_in * DPI
    print_margin_bottom = print_margin_bottom_in * DPI
    poly_height = poly_height * DPI
    help_length = help_length_in * DPI
    line_width = max(1, round(MAIN_LINE_WIDTH_IN * DPI))
    help_line_width = max(1, round(HINT_LINE_WIDTH_IN * DPI))
    beam_line_width = max(1, round(BEAM_LINE_WIDTH_IN * DPI))
    corner_dot_radius = max(1, round(CORNER_DOT_RADIUS_IN * DPI))
    fold_dot_radius = max(1, round(FOLD_DOT_RADIUS_IN * DPI))
    fold_dot_outline = max(1, round((FOLD_DOT_OUTLINE_IN * DPI) + fold_dot_radius))

    # creates image layers and drawing objects for each.
    main_lines = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    second_lines = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    hints = Image.new("RGBA", (w, h), (255, 255, 255, 0))

    main_draw = ImageDraw.Draw(main_lines)
    second_draw = ImageDraw.Draw(second_lines)
    hints_draw = ImageDraw.Draw(hints)

    """
    Step 2) Calculates the polygon points.
    """
    top_points = gen_poly_points(
        center_x=w / 2,
        top=h / 2 - poly_height,
        bottom=h / 2,
        num_sides=num_sides,
    )
    bottom_points = [(p[0], h - p[1]) for p in top_points]

    poly_measure = (num_sides - 2) * np.pi / num_sides
    side_measure = (np.pi - poly_measure) / 2

    t, b = top_points, bottom_points  # primary points.
    side_length = point_dist(t[0], t[1])
    inner_length = side_length * (np.sin(side_measure) / np.sin(poly_measure))
    min_y = min([p[1] for p in t])
    inner_top_points = gen_poly_points(
        center_x=w / 2,
        top=(
            t[1][1]
            if num_sides % 2 == 1
            else (min_y + inner_length * np.sin(side_measure))
        ),  #
        bottom=t[num_sides // 2][1] - inner_length * np.sin(side_measure),
        upward=False,
        num_sides=num_sides,
    )
    inner_bottom_points = [(p[0], h - p[1]) for p in inner_top_points]

    # draws the main horizontal line.
    main_draw.line(
        (0, h // 2, w, h // 2),
        fill=MAIN_LINE_COLOR,
        width=line_width,
    )

    it, ib = inner_top_points, inner_bottom_points  # secondary points.

    class Fold:
        PAUSE = 0
        CONTINUE = 1
        LIMIT = 2

        def __init__(self, series):
            points = [
                s for s in series if s not in [Fold.PAUSE, Fold.CONTINUE, Fold.LIMIT]
            ]
            x1, y1 = points[0]
            x2, y2 = points[1]

            self.radians = regularize_radians(np.arctan2(y2 - y1, x2 - x1))

            self.edges = find_intersections(
                points[0], self.radians, right=w, bottom=h, both_ways=True
            )

            # ensures first edge is on y-axis, second edge is on x-axis.
            if self.edges[0][0] not in [0, w] or self.edges[0][1] in [0, h]:
                swapout = self.edges[0][:]
                self.edges[0] = self.edges[1][:]
                self.edges[1] = swapout[:]

            self.series = []
            self.series.append(
                find_intersections(
                    points[0],
                    regularize_radians(self.radians + np.pi),
                    right=w,
                    bottom=h,
                    both_ways=False,
                )[0]
            )
            self.series.append(Fold.CONTINUE)
            self.series.extend(series)

            if self.series[-1] == Fold.CONTINUE:
                self.series[-1] = find_intersections(
                    points[0], self.radians, right=w, bottom=h, both_ways=False
                )[0]

            is_top = points[0][1] < h // 2

            for i in range(len(self.series)):
                if self.series[i] == Fold.LIMIT:
                    self.series[i] = find_intersections(
                        points[0],
                        self.radians,
                        top=0 if is_top else h // 2,
                        right=w,
                        bottom=h // 2 if is_top else h,
                        both_ways=False,
                    )[0]

            self.series = [s for s in self.series if s != Fold.CONTINUE]

            self.lines = []
            line = []
            num_pauses = 0
            for s in self.series:
                if s == Fold.PAUSE:
                    num_pauses += 1
                    pass
                else:
                    line.append(s)

                if len(line) == 2:
                    self.lines.append(tuple(line))
                    line = []

            if num_pauses == 0 and len(self.lines) > 1:
                self.lines = [(self.lines[0][0], self.lines[-1][-1])]

            CORNERS = {
                "tl": (0, 0),
                "tr": (w, 0),
                "br": (w, h),
                "bl": (0, h),
            }
            reflects = {}
            reflect_dist = {}  # distances from center.
            center = (w / 2, h / 2)
            best_corner = None
            min_dist = w + h
            for name, corner in CORNERS.items():
                reflects[name] = reflect_across_line(
                    corner, self.edges[0], self.edges[1], w, h
                )
                reflect_dist[name] = point_dist(reflects[name], center)
                if reflect_dist[name] < min_dist:
                    min_dist = reflect_dist[name]
                    best_corner = name

            def dists_are_similar(c1: str, c2: str):
                return abs(reflect_dist[c2] - reflect_dist[c1]) < 1

            if best_corner in ["tl", "tr"] and dists_are_similar("tl", "tr"):
                self.corner = "t"
                self.marks = [reflects["tl"], reflects["tr"]]
            elif best_corner in ["tr", "br"] and dists_are_similar("tr", "br"):
                self.corner = "r"
                self.marks = [reflects["tr"], reflects["br"]]
            elif best_corner in ["bl", "br"] and dists_are_similar("bl", "br"):
                self.corner = "b"
                self.marks = [reflects["bl"], reflects["br"]]
            elif best_corner in ["tl", "bl"] and dists_are_similar("tl", "bl"):
                self.corner = "l"
                self.marks = [reflects["tl"], reflects["bl"]]
            elif best_corner is not None:
                self.corner = best_corner
                self.marks = [reflects[best_corner]]

            next_p = (
                self.series[0][0] + np.cos(self.radians) * 100,
                self.series[0][1] + np.sin(self.radians) * 100,
            )

            self.hint_points = tuple([e for e in self.edges])

            def select_edge(find_min_x=None, find_max_x=None):
                a, b = self.edges[0], self.edges[1]

                if find_min_x == True or find_max_x != True:
                    # returns edge with minimum X.
                    return a if a[0] < b[0] else b

                # returns edge with maximum X.
                return a if a[0] > b[0] else b

            if self.corner == "tl":
                if self.point_y_is_under(CORNERS["tr"]):
                    mirror = reflect_across_line(
                        CORNERS["tr"], self.series[0], next_p, w, h
                    )
                    hinge = select_edge(find_min_x=True)
                    self.hint_points = (hinge, mirror)
                elif self.point_y_is_under(CORNERS["bl"]):
                    mirror = reflect_across_line(
                        CORNERS["bl"], self.series[0], next_p, w, h
                    )
                    hinge = select_edge(find_max_x=True)
                    self.hint_points = (hinge, mirror)

            elif self.corner == "tr":
                if self.point_y_is_under(CORNERS["tl"]):
                    mirror = reflect_across_line(
                        CORNERS["tl"], self.series[0], next_p, w, h
                    )
                    hinge = select_edge(find_max_x=True)
                    self.hint_points = (hinge, mirror)
                elif self.point_y_is_under(CORNERS["br"]):
                    mirror = reflect_across_line(
                        CORNERS["br"], self.series[0], next_p, w, h
                    )
                    hinge = select_edge(find_min_x=True)
                    self.hint_points = (hinge, mirror)

            elif self.corner == "br":
                if not self.point_y_is_under(CORNERS["bl"]):
                    mirror = reflect_across_line(
                        CORNERS["bl"], self.series[0], next_p, w, h
                    )
                    hinge = select_edge(find_max_x=True)
                    self.hint_points = (hinge, mirror)
                elif not self.point_y_is_under(CORNERS["tr"]):
                    mirror = reflect_across_line(
                        CORNERS["tr"], self.series[0], next_p, w, h
                    )
                    hinge = select_edge(find_min_x=True)
                    self.hint_points = (hinge, mirror)

            else:
                if not self.point_y_is_under(CORNERS["br"]):
                    mirror = reflect_across_line(
                        CORNERS["br"], self.series[0], next_p, w, h
                    )
                    hinge = select_edge(find_min_x=True)
                    self.hint_points = (hinge, mirror)
                elif not self.point_y_is_under(CORNERS["tl"]):
                    mirror = reflect_across_line(
                        CORNERS["tl"], self.series[0], next_p, w, h
                    )
                    hinge = select_edge(find_max_x=True)
                    self.hint_points = (hinge, mirror)

        def point_y_is_under(self, point):
            if angle_is_vertical(self.radians):
                return False

            x0, y0 = self.series[0]
            slope = np.tan(self.radians)

            line_y = slope * (point[0] - x0) + y0
            return point[1] < line_y

    folds = []
    for i in range(num_sides):
        next_i = (i + 1) % num_sides
        second_next_i = (i + 2) % num_sides

        """
        Step 4) Looks for lines formed between primary points.
        """
        x1, y1 = t[i]
        x2, y2 = t[next_i]
        last_radians = regularize_radians(np.arctan2(y2 - y1, x2 - x1))

        b_on_line = select_points_on_line(
            points_list=b, start_point=t[next_i], radians=last_radians
        )
        ib_on_line = select_points_on_line(
            points_list=ib, start_point=t[next_i], radians=last_radians
        )

        if len(b_on_line) > 1 and len(ib_on_line) > 1:
            series = [
                ib[ib_on_line[0][0]],
                Fold.PAUSE,
                ib[ib_on_line[-1][0]],
                Fold.CONTINUE,
            ]
        elif (
            len(b_on_line) > 0
            and len(ib_on_line) == 0
            and t[next_i] != b[b_on_line[-1][0]]
        ):
            closest_i = i if t[i][1] > t[next_i][1] else next_i
            series = [t[closest_i], Fold.PAUSE, b[b_on_line[-1][0]], Fold.CONTINUE]
        elif len(b_on_line) == 0 and len(ib_on_line) > 0:
            series = [
                ib[ib_on_line[0][0]],
                Fold.PAUSE,
                ib[ib_on_line[-1][0]],
                Fold.CONTINUE,
            ]
        else:
            if last_radians < np.pi:
                series = [t[i], Fold.CONTINUE, t[next_i], Fold.CONTINUE, Fold.LIMIT]
            else:
                series = [t[next_i], Fold.CONTINUE, t[i], Fold.CONTINUE, Fold.LIMIT]

        folds.append(Fold(series))

        x1, y1 = b[i]
        x2, y2 = b[next_i]
        last_radians = regularize_radians(np.arctan2(y2 - y1, x2 - x1))

        t_on_line = select_points_on_line(
            points_list=t, start_point=b[next_i], radians=last_radians
        )
        it_on_line = select_points_on_line(
            points_list=it, start_point=b[next_i], radians=last_radians
        )

        series = None
        if len(t_on_line) > 1 and len(it_on_line) > 1:
            series = [
                it[it_on_line[0][0]],
                Fold.PAUSE,
                it[it_on_line[-1][0]],
                Fold.CONTINUE,
            ]
        elif (
            len(t_on_line) > 0
            and len(it_on_line) == 0
            and b[next_i] != t[t_on_line[-1][0]]
        ):
            closest_i = i if b[i][1] < b[next_i][1] else next_i
            series = [b[closest_i], Fold.PAUSE, t[t_on_line[-1][0]], Fold.CONTINUE]
        elif len(t_on_line) == 0 and len(it_on_line) > 0:
            series = [
                it[it_on_line[0][0]],
                Fold.PAUSE,
                it[it_on_line[-1][0]],
                Fold.CONTINUE,
            ]
        else:
            if last_radians >= np.pi:
                series = [b[i], Fold.CONTINUE, b[next_i], Fold.CONTINUE, Fold.LIMIT]
            else:
                series = [b[next_i], Fold.CONTINUE, b[i], Fold.CONTINUE, Fold.LIMIT]

        if series is not None:
            folds.append(Fold(series))

        """
        Step 5) Looks for lines formed between a primary and secondary point.
        """
        x1, y1 = it[i]
        x2, y2 = it[next_i]
        last_radians = regularize_radians(np.arctan2(y2 - y1, x2 - x1))

        b_on_line = select_points_on_line(
            points_list=b,
            start_point=it[next_i],
            radians=last_radians,
        )
        ib_on_line = select_points_on_line(
            points_list=ib, start_point=it[next_i], radians=last_radians
        )

        if len(b_on_line) > 0 and len(ib_on_line) > 0:
            if last_radians < np.pi:
                series = [
                    it[i],
                    Fold.PAUSE,
                    it[next_i],
                    Fold.CONTINUE,
                    ib[ib_on_line[0][0]],
                    Fold.PAUSE,
                    ib[ib_on_line[-1][0]],
                    Fold.CONTINUE,
                ]
            else:
                series = [
                    it[next_i],
                    Fold.PAUSE,
                    it[i],
                    Fold.CONTINUE,
                    ib[ib_on_line[0][0]],
                    Fold.PAUSE,
                    ib[ib_on_line[-1][0]],
                    Fold.CONTINUE,
                ]
        elif len(b_on_line) > 0 and len(ib_on_line) == 0:
            closest_i = closest_point_index(
                [it[i], it[next_i]], b[b_on_line[0][0]], indices=[i, next_i]
            )
            far_i = i if closest_i == next_i else next_i

            series = [
                it[far_i],
                Fold.PAUSE,
                it[closest_i],
                Fold.CONTINUE,
                Fold.LIMIT,
                Fold.PAUSE,
                b[b_on_line[-1][0]],
                Fold.CONTINUE,
            ]
        elif len(b_on_line) == 0 and len(ib_on_line) > 0:
            if len(ib_on_line) == 1:
                series = [
                    it[i],
                    Fold.PAUSE,
                    it[next_i],
                    Fold.CONTINUE,
                    ib[ib_on_line[0][0]],
                ]
            else:
                series = [
                    it[i],
                    Fold.PAUSE,
                    it[next_i],
                    Fold.CONTINUE,
                    ib[ib_on_line[0][0]],
                    Fold.PAUSE,
                    ib[ib_on_line[-1][0]],
                ]
        else:
            if last_radians < np.pi:
                series = [it[i], Fold.PAUSE, it[next_i], Fold.CONTINUE, Fold.LIMIT]
            else:
                series = [it[next_i], Fold.PAUSE, it[i], Fold.CONTINUE, Fold.LIMIT]

        folds.append(Fold(series))

        x1, y1 = ib[i]
        x2, y2 = ib[next_i]
        last_radians = regularize_radians(np.arctan2(y2 - y1, x2 - x1))

        t_on_line = select_points_on_line(
            points_list=t,
            start_point=ib[next_i],
            radians=last_radians,
        )
        it_on_line = select_points_on_line(
            points_list=it, start_point=ib[next_i], radians=last_radians
        )

        if len(t_on_line) > 0 and len(it_on_line) > 0:
            if last_radians > np.pi:
                series = [
                    ib[i],
                    Fold.PAUSE,
                    ib[next_i],
                    Fold.CONTINUE,
                    it[it_on_line[0][0]],
                    Fold.PAUSE,
                    it[it_on_line[-1][0]],
                    Fold.CONTINUE,
                ]
            else:
                series = [
                    ib[next_i],
                    Fold.PAUSE,
                    ib[i],
                    Fold.CONTINUE,
                    it[it_on_line[0][0]],
                    Fold.PAUSE,
                    it[it_on_line[-1][0]],
                    Fold.CONTINUE,
                ]
        elif len(t_on_line) > 0 and len(it_on_line) == 0:
            closest_i = closest_point_index(
                [ib[i], ib[next_i]], t[t_on_line[0][0]], indices=[i, next_i]
            )
            far_i = i if closest_i == next_i else next_i
            series = [
                ib[far_i],
                Fold.PAUSE,
                ib[closest_i],
                Fold.CONTINUE,
                Fold.LIMIT,
                Fold.PAUSE,
                t[t_on_line[-1][0]],
                Fold.CONTINUE,
            ]
        elif len(t_on_line) == 0 and len(it_on_line) > 0:
            if len(it_on_line) == 1:
                series = [
                    ib[i],
                    Fold.PAUSE,
                    ib[next_i],
                    Fold.CONTINUE,
                    it[it_on_line[0][0]],
                ]
            else:
                series = [
                    ib[i],
                    Fold.PAUSE,
                    ib[next_i],
                    Fold.CONTINUE,
                    it[it_on_line[0][0]],
                    Fold.PAUSE,
                    it[it_on_line[-1][0]],
                ]
        else:
            if last_radians > np.pi:
                series = [ib[i], Fold.PAUSE, ib[next_i], Fold.CONTINUE, Fold.LIMIT]
            else:
                series = [ib[next_i], Fold.PAUSE, ib[i], Fold.CONTINUE, Fold.LIMIT]

        folds.append(Fold(series))

    # draws all the folds.
    CORNERS = {"tl": (0, 0), "tr": (w, 0), "br": (w, h), "bl": (0, h)}
    LINE_COLORS = {
        "tl": TL_COLOR,
        "tr": TR_COLOR,
        "br": BR_COLOR,
        "bl": BL_COLOR,
        "t": ORTHO_LINE_COLOR,
        "b": ORTHO_LINE_COLOR,
        "l": ORTHO_LINE_COLOR,
        "r": ORTHO_LINE_COLOR,
    }
    LINE_PATTERNS = {
        "tl": "solid",
        "tr": "solid",
        "br": "solid",
        "bl": "solid",
        "t": "solid",
        "b": "solid",
        "l": "solid",
        "r": "solid",
    }
    for fold in folds:
        fill = LINE_COLORS[fold.corner]
        pattern = LINE_PATTERNS[fold.corner]
        light_fill = interpolate_rgb(fill, (255, 255, 255), factor=RGB_FADE)
        for a, b in fold.lines:
            draw_line(
                main_draw,
                a[:2],
                b[:2],
                pattern=pattern,
                fill=fill,
                width=line_width,
            )

        if fold.corner in CORNERS.keys():
            draw_circle(
                hints_draw,
                center=fold.marks[0],
                fill=(0, 0, 0),
                radius=fold_dot_outline,
            )
            draw_circle(
                hints_draw, center=fold.marks[0], fill=fill, radius=fold_dot_radius
            )

            # draws hint lines.
            if fold.hint_points is not None:
                for next_x, next_y in fold.hint_points:
                    draw_line(
                        hints_draw,
                        fold.marks[0][:2],
                        (next_x, next_y),
                        pattern=pattern,
                        fill=light_fill,
                        width=help_line_width,
                    )

        ORTHO_HINT_LENGTH_PX = 200
        if fold.corner in "bt":
            mark_y = fold.marks[0][1]
            start_x = 0
            end_x = w

            fill = interpolate_rgb(
                ORTHO_LINE_COLOR, (255, 255, 255), factor=ORTHO_RGB_FADE
            )
            draw_line(
                hints_draw,
                (start_x, mark_y),
                (start_x + ORTHO_HINT_LENGTH_PX, mark_y),
                pattern=pattern,
                fill=fill,
                width=help_line_width,
            )

            draw_line(
                hints_draw,
                (end_x - ORTHO_HINT_LENGTH_PX, mark_y),
                (end_x, mark_y),
                pattern=pattern,
                fill=fill,
                width=help_line_width,
            )

        elif fold.corner in "lr":
            mark_x = fold.marks[0][0]
            start_y = 0
            end_y = h

            fill = interpolate_rgb(
                ORTHO_LINE_COLOR, (255, 255, 255), factor=ORTHO_RGB_FADE
            )
            draw_line(
                hints_draw,
                (mark_x, start_y),
                (mark_x, start_y + ORTHO_HINT_LENGTH_PX),
                pattern=pattern,
                fill=fill,
                width=help_line_width,
            )

            draw_line(
                hints_draw,
                (mark_x, end_y - ORTHO_HINT_LENGTH_PX),
                (mark_x, end_y),
                pattern=pattern,
                fill=fill,
                width=help_line_width,
            )

    draw_circle(second_draw, center=(0, 0), fill=TL_COLOR, radius=corner_dot_radius)
    draw_circle(second_draw, center=(w, 0), fill=TR_COLOR, radius=corner_dot_radius)
    draw_circle(second_draw, center=(w, h), fill=BR_COLOR, radius=corner_dot_radius)
    draw_circle(second_draw, center=(0, h), fill=BL_COLOR, radius=corner_dot_radius)

    """
    Step 6) Draws beaming lines from star.
    """
    for i in range(num_sides):
        start = it[i]
        counter = (
            t[(i + num_sides // 2 + 1) % num_sides]
            if num_sides % 2 == 1
            else it[(i + num_sides // 2) % num_sides]
        )
        radians = np.arctan2(start[1] - counter[1], start[0] - counter[0])
        end = find_intersections(
            start, radians, right=w, bottom=h / 2, both_ways=False
        )[0]
        second_draw.line(
            (start[0], start[1], end[0], end[1]),
            fill=BEAM_LINE_COLOR,
            width=beam_line_width,
        )

        radians = regularize_radians(-radians)
        start = ib[i]
        end = find_intersections(
            start, radians, top=h / 2, right=w, bottom=h, both_ways=False
        )[0]
        second_draw.line(
            (start[0], start[1], end[0], end[1]),
            fill=BEAM_LINE_COLOR,
            width=beam_line_width,
        )

    """
    Step X) Pastes the transparent onto a solid white background 
            and finally saves it.
    """
    white_background = Image.new("RGBA", main_lines.size, (255, 255, 255, 255))

    # hints  ->  main_lines
    # image  ->  second_lines
    # image  ->  white background
    image = Image.alpha_composite(main_lines, hints)
    image = Image.alpha_composite(second_lines, image)
    image = Image.alpha_composite(white_background, image)

    # applies the print margin (if its specified).
    if (
        print_margin_left > 0
        or print_margin_right > 0
        or print_margin_top > 0
        or print_margin_bottom > 0
    ):
        image = image.crop(
            (
                print_margin_left,
                print_margin_top,
                w - print_margin_right,
                h - print_margin_bottom,
            )
        )

    return image
