import os
import numpy as np
from PIL import Image, ImageDraw
import reportlab.lib.pagesizes
from reportlab.pdfgen import canvas


DPI = 300

def _regularize_radians(radians):
    STEP = 2 * np.pi

    while radians < 0:
        radians += STEP
    while radians >= STEP:
        radians -= STEP

    return radians


def _calc_perp_radians(radians):
    """Returns the angle measure that's perpendicular to the given measure."""
    result = radians + np.pi / 2
    result = _regularize_radians(result)

    return result


def _gen_poly_points(center_x, top, bottom, num_sides: int=5, upward: bool=True):
    """
    Returns a list of points scaled to fit to the y-coords of top and bottom.
    """
    step = np.pi*2 / num_sides
    offset = 0 if num_sides % 2 == 1 else step / 2

    # calculates the points by rotating in a circle.
    points = []
    for i in range(num_sides):
        angle = step*i + offset - np.pi/2 + (0 if upward else np.pi)
        perp = _calc_perp_radians(angle)
        points.append((np.cos(angle), np.sin(angle), perp))

    min_y = np.min([p[1] for p in points])
    max_y = np.max([p[1] for p in points])

    # moves the points so the lowest point's Y-coord is at 0.
    points = [(p[0], p[1] - min_y, p[2]) for p in points]

    # scales the points to fit between the given top and bottom Y-coords,
    # then returns the results.
    span = bottom - top
    diff = max_y - min_y
    scale_factor = span / diff

    return [
        (
            center_x + p[0] * scale_factor,
            top + p[1] * scale_factor,
            p[2],
        )
        for p in points
    ]


def interpolate_color(color_a, color_b, factor=0.5):
    diff = (b - a for a, b in zip(color_a, color_b))
    result = tuple(round(a + d * factor) for a, d in zip(color_a, diff))

    return result


def _draw_circle(draw, center, radius=20, fill_color=(0, 200, 100)):
    left_up = (center[0] - radius, center[1] - radius)
    right_down = (center[0] + radius, center[1] + radius)
    draw.ellipse([left_up, right_down], fill=fill_color)


def find_edge_intersection(point, radians, max_x, max_y, min_x=0, min_y=0):
    radians = _regularize_radians(radians)
    going_right = radians > 3 * np.pi/2 or radians < np.pi/2
    going_down = 0 < radians < np.pi

    limit_x = max_x if going_right else min_x
    limit_y = max_y if going_down else min_y

    slope = np.tan(radians)
    hit_x = (limit_y - point[1]) / slope + point[0]
    hit_y = slope * (limit_x - point[0]) + point[1]
    hit = (hit_x, limit_y) if min_x <= hit_x < max_x else (limit_x, hit_y)

    return hit


def _calc_line(image_width, image_height, start, end, cap_start: bool=False):
    OOB = 100
    AXIS_PRECISION = np.pi / 180
    w, h = image_width, image_height
    angle = np.arctan2(end[1] - start[1], end[0] - start[0])  #  start to end.

    # handles occurrences where the line is practically vertical/horizontal.
    for i in range(4):
        current_axis = i * np.pi/2
        if angle - AXIS_PRECISION < current_axis < angle + AXIS_PRECISION:
            if i in [0, 2]: #  is horizontal.
                if not cap_start:
                    # extends the start point off the paper.
                    if start[0] < end[0]:
                        start = (-OOB, start[1])
                    else:
                        start = (w + OOB, start[1])
                
                # extends the end point off the paper.
                if end[0] < start[0]:
                    end = (-OOB, start[1])
                else:
                    end = (w + OOB, start[1])
                
                

            else: #  is vertical.
                if not cap_start:
                    # extends the start point off the paper.
                    if start[1] < end[1]:
                        start = (start[0], -OOB)
                    else:
                        start = (start[0], h + OOB)
                   
                # extends the end point off the paper.
                if end[1] < start[1]:
                    end = (start[0], -OOB)
                else:
                    end = (start[0], h + OOB)
                

            return (start, end)

    # by this point, it's known that the angle is a non-orthogonal angle.
    slope = np.tan(angle)

    def extend_point_off_paper(point, radians):
        hit = find_edge_intersection(point, radians, max_x=w, max_y=h)
        return (hit[0] + np.cos(radians)*OOB, hit[1] + np.sin(radians)*OOB)

    if not cap_start:
        flipped = _regularize_radians(angle + np.pi)
        start = extend_point_off_paper(start, flipped)

    end = extend_point_off_paper(end, angle)

    return (start, end)


def reflect_point(p, a, b, image_width, image_height):
    """
    Returns a tuple with the first element being 
    the point <p> reflect across the line defined by <a> to <b>
    and the second and third elements being
    the two points on the edge of the paper where
    the paper begins its fold.
    """
    w, h = image_width, image_height

    """ Step 1) The edges of the paper where fold will begin are found. """
    # strips extra info.
    p = p[:2]
    a = a[:2]
    b = b[:2]

    angle = np.arctan2(b[1] - a[1], b[0] - a[0]) #  start to end.
    flipped = _regularize_radians(angle + np.pi)
    hits = [
        find_edge_intersection(a, angle, max_x=w, max_y=h),
        find_edge_intersection(a, flipped, max_x=w, max_y=h),
    ]

    """ Step 2) The reflection is found. """
    p = np.array(p)
    a = np.array(a)
    b = np.array(b)

    ab = b - a # from <a> to <b>.    
    ap = p - a # from <a> to <p>.
    
    # projects the vector <ap> onto the vector <ab>.
    ab_normalized = ab / np.linalg.norm(ab)
    projection_length = np.dot(ap, ab_normalized)
    projection = a + projection_length * ab_normalized
    
    # reflects point <p> across the line.
    reflection = tuple(2 * projection - p)

    return (reflection, hits[0], hits[1])


def _draw_gradient_line(draw, start, end, start_color, end_color, line_width):
    dist = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    
    rise = end[1] - start[1]
    run = end[0] - start[0]
    slope = rise / run if run != 0 else 1000000
    
    num_units = int(dist) + 1
    step = 1 / (num_units - 1)

    for i in range(num_units):
        x = start[0] + i
        y = start[1] + slope * x
        color = interpolate_color(start_color, end_color, factor=step * i)
        _draw_circle(draw, center=(x, y), radius=line_width/2, fill_color=color)



_MAIN_LINE_COLOR = (0, 0, 0)
_TR_COLOR = (100, 60, 200)
_TL_COLOR = (200, 60, 100)
_BL_COLOR = (100, 200, 60)
_BR_COLOR = (200, 170, 30)

def _draw_fundamental_lines(
    draw,
    top_points,
    bottom_points,
    inner_top_points,
    inner_bottom_points,
    image_width,
    image_height,
    line_width,
):
    t, b = top_points, bottom_points
    it, ib = inner_top_points, inner_bottom_points
    w, h = image_width, image_height
    
    def draw_lines(
        p, inner_p, other_inner_p, h_left_color, v_left_color, h_right_color, v_right_color
    ):
        angle_43 = np.arctan2(p[3][1] - p[4][1], p[3][0] - p[4][0])
        angle_12 = np.arctan2(p[2][1] - p[1][1], p[2][0] - p[1][0])
        lines = [
            # left side.
            (_calc_line(w, h, start=p[0], end=p[4], cap_start=False), h_left_color),
            (_calc_line(w, h, start=other_inner_p[1], end=p[4], cap_start=True), v_left_color),
            (
                _calc_line(
                    w,
                    h,
                    start=other_inner_p[2],
                    end=(p[0][0] + np.cos(angle_43), h - p[0][1] + np.sin(angle_43)),
                    cap_start=True,
                ),
                v_left_color,
            ),

            # right side.
            (_calc_line(w, h, start=p[0], end=p[1], cap_start=False), h_right_color),
            (_calc_line(w, h, start=other_inner_p[4], end=p[1], cap_start=True), v_right_color),
            (
                _calc_line(
                    w,
                    h,
                    start=other_inner_p[3],
                    end=(p[0][0] + np.cos(angle_12), h - p[0][1] + np.sin(angle_12)),
                    cap_start=True,
                ),
                v_right_color,
            ),
        ]

        for line, color in lines:
            start, end = line
            draw.line(
                (start[0], start[1], end[0], end[1]),
                fill=color,
                width=line_width,
            )

        midway_color = interpolate_color(h_left_color, h_right_color)
        _draw_gradient_line(draw, (0, p[4][1]), inner_p[2], h_left_color, midway_color, line_width=line_width)
        _draw_gradient_line(draw, inner_p[3], (w, p[1][1]), midway_color, h_right_color, line_width=line_width)

    draw_lines(t, it, ib, _TL_COLOR, _BL_COLOR, _TR_COLOR, _BR_COLOR)
    draw_lines(b, ib, it, _BL_COLOR, _TL_COLOR, _BR_COLOR, _TR_COLOR)


def _draw_secondary_lines(
    draw,
    top_points,
    bottom_points,
    image_width,
    image_height,
    line_width,
):
    t, b = top_points, bottom_points


def create_star_diagram(
    print_width=8.5,
    print_height=11,
    print_margin_left=0.125,
    print_margin_right=0.125,
    print_margin_top=0.125,
    print_margin_bottom=0.125,
    poly_height=4.0625,
    is_metric: bool=False,
):
    """
    Writes an origami diagram to file that can be used to quickly
    turn the piece of paper into a 3D star.

    Parameters:
        out_path (str): where the .png diagram will be saved.
        print_width (num): the width of the paper (in inches/cm).
        print_height (num): the height of the paper (in inches/cm).
        poly_height (num): the distance from the top to bottom of the polygon.
        is_metric (bool): if True, all given measurements are treated as cm.
        DPI (num): the print resolution (pixels-per-inch).
    """

    MAIN_LINE_WIDTH_IN = 1 / 16
    HINT_LINE_WIDTH_IN = 1 / 32

    """
    Step 1) Initializes print and diagram settings.
    """
    # print settings. 
    print_width_in = print_width / 2.54 if is_metric else print_width
    print_height_in = print_height / 2.54 if is_metric else print_height
    print_margin_left_in = print_margin_left / 2.54 if is_metric else print_margin_left
    print_margin_right_in = print_margin_right / 2.54 if is_metric else print_margin_right
    print_margin_top_in = print_margin_top / 2.54 if is_metric else print_margin_top
    print_margin_bottom_in = print_margin_bottom / 2.54 if is_metric else print_margin_bottom

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

    """
    Step 2) Calculates the polygon points.
    """
    top_points = _gen_poly_points(
        center_x=w / 2,
        top=h / 2 - poly_height,
        bottom=h / 2,
    )
    bottom_points = [(p[0], h - p[1]) for p in top_points]

    t, b = top_points, bottom_points
    side_length = np.sqrt((t[1][0] - t[0][0])**2 + (t[1][1] - t[0][1])**2)
    inner_length = side_length * (np.sin(np.radians(36)) / np.sin(np.radians(108)))
    inner_top_points = _gen_poly_points(
        center_x=w / 2,
        top=t[1][1],
        bottom=t[2][1] - inner_length * np.sin(np.radians(36)),
        upward=False
    )
    inner_bottom_points = [(p[0], h - p[1]) for p in inner_top_points]


    """
    Step 3) Creates the template image and draws lines.
    """
    main_lines = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    second_lines = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    first_hints = Image.new("RGBA", (w, h), (255, 255, 255, 0))

    main_draw = ImageDraw.Draw(main_lines)
    second_draw = ImageDraw.Draw(second_lines)
    first_hints_draw = ImageDraw.Draw(first_hints)

    # draws the main horizontal line.
    main_draw.line(
        (0, h // 2, w, h // 2),
        fill=_MAIN_LINE_COLOR,
        width=line_width,
    )

    _draw_fundamental_lines(
        main_draw,
        top_points,
        bottom_points,
        inner_top_points,
        inner_bottom_points,
        w,
        h,
        line_width,
    )
    _draw_secondary_lines(second_draw, top_points, bottom_points, w, h, line_width)

    """
    Step 4) Draws dots.
    """
    TR, TL, BL, BR = (w, 0), (0, 0), (0, h), (w, h)
    _draw_circle(main_draw, center=TR, radius=100, fill_color=_TR_COLOR)
    _draw_circle(main_draw, center=TL, radius=100, fill_color=_TL_COLOR)
    _draw_circle(main_draw, center=BL, radius=100, fill_color=_BL_COLOR)
    _draw_circle(main_draw, center=BR, radius=100, fill_color=_BR_COLOR)


    def draw_fold_diagram(point, start, end, color):
        RADIUS = 30
        point, edge_a, edge_b = reflect_point(point, start, end, w, h) # TR, t[0], t[1]
        angle_a = np.arctan2(edge_a[1] - point[1], edge_a[0] - point[0])
        angle_b = np.arctan2(edge_b[1] - point[1], edge_b[0] - point[0])

        angle_a = _regularize_radians(angle_a)
        angle_b = _regularize_radians(angle_b)

        _draw_circle(first_hints_draw, center=point, radius=RADIUS, fill_color=color)
        
        dist_a = np.sqrt((edge_a[0] - point[0])**2 + (edge_a[1] - point[1])**2)
        dist_b = np.sqrt((edge_b[0] - point[0])**2 + (edge_b[1] - point[1])**2)

        ratio = dist_a / (dist_a + dist_b)

        help_a = (
            point[0] + np.cos(angle_a) * help_length * ratio,
            point[1] + np.sin(angle_a) * help_length * ratio,
        )
        help_b = (
            point[0] + np.cos(angle_b) * help_length * (1 - ratio),
            point[1] + np.sin(angle_b) * help_length * (1 - ratio),
        )
        factor = 0.65
        first_hints_draw.line(
            (point[0], point[1], help_a[0], help_a[1]),
            fill=interpolate_color(color, (255, 255, 255), factor=factor),
            width=help_line_width,
        )
        first_hints_draw.line(
            (point[0], point[1], help_b[0], help_b[1]),
            fill=interpolate_color(color, (255, 255, 255), factor=factor),
            width=help_line_width,
        )


    draw_fold_diagram(TR, t[0], t[1], _TR_COLOR)
    draw_fold_diagram(TL, t[0], t[4], _TL_COLOR)
    draw_fold_diagram(BL, b[0], b[4], _BL_COLOR)
    draw_fold_diagram(BR, b[0], b[1], _BR_COLOR)

    draw_fold_diagram(TR, t[0], t[2], _TR_COLOR)
    draw_fold_diagram(TL, t[0], t[3], _TL_COLOR)
    draw_fold_diagram(BL, b[0], b[3], _BL_COLOR)
    draw_fold_diagram(BR, b[0], b[2], _BR_COLOR)



    """
    Step 5) Pastes the transparent onto a solid white background 
            and finally saves it.
    """
    # 
    white_background = Image.new("RGBA", main_lines.size, (255, 255, 255, 255))
    image = Image.alpha_composite(main_lines, first_hints)
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


def _png_to_pdf(images: list, pdf_path, page_size, landscape: bool=False):
    
    if landscape:
        page_size = (max(page_size), min(page_size))
    else:
        page_size = (min(page_size), max(page_size))

    c = canvas.Canvas(pdf_path, pagesize=page_size)

    # creates a PDF canvas.
    pdf_width, pdf_height = page_size

    paths = []
    for i, image in enumerate(images):
        TEMP_PATH = f"temp-image-{i:03d}.png"
        paths.append(TEMP_PATH)
        img_width, img_height = image.size

        # calculates the scale factor to fit the image in the PDF.
        scale_x = pdf_width / img_width
        scale_y = pdf_height / img_height
        scale = min(scale_x, scale_y)

        image.save(TEMP_PATH)

        # scales the image and draw it on the PDF.
        c.drawImage(TEMP_PATH, 0, 0, width=img_width * scale, height=img_height * scale)
        
        if i < len(images) - 1:
            c.showPage()

    # saves the PDF.
    c.save()

    for path in paths:
        os.remove(path)



def main():
    page_size = reportlab.lib.pagesizes.letter

    star_diagram = create_star_diagram(
        print_width=11,
        print_height=8.5 * 2,
        poly_height=8,
        print_margin_left=0,
        print_margin_right=0,
        print_margin_top=0,
        print_margin_bottom=0,
    )

    star_diagram.save("stars.png")

    _png_to_pdf([star_diagram.crop((0, 0, 11 * DPI, 8.5 * DPI)), star_diagram.crop((0, 8.5 * DPI, 11 * DPI, 17 * DPI))], "star.pdf", page_size, landscape=True) 


if __name__ == "__main__":
    main()