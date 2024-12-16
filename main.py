import os
import numpy as np
from PIL import Image, ImageDraw
import reportlab.lib.pagesizes
from reportlab.pdfgen import canvas
from trig_math import *
from paint_diagram import *

MAIN_LINE_COLOR = (0, 0, 0)
HORIZ_LINE_COLOR = (128, 128, 128)
BEAM_LINE_COLOR = (223, 223, 223)
TR_COLOR = (100, 60, 200)
TL_COLOR = (200, 60, 100)
BL_COLOR = (100, 200, 60)
BR_COLOR = (200, 170, 30)
CORNER_DOT_RADIUS_IN = 1/2
FOLD_DOT_RADIUS_IN = 1/8
MAIN_LINE_WIDTH_IN = 3/64
HINT_LINE_WIDTH_IN = 1/32
BEAM_LINE_WIDTH_IN = 1/32

DPI = 300

def create_star_diagram(
    print_width=8.5,
    print_height=11,
    print_margin_left=0.125,
    print_margin_right=0.125,
    print_margin_top=0.125,
    print_margin_bottom=0.125,
    poly_height=4.0625,
    is_metric: bool=False,
    num_sides=5,
):
    """
    Writes an origami diagram to file that can be used to quickly
    turn the piece of paper into a 3D star.

    Parameters:
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
    beam_line_width = max(1, round(BEAM_LINE_WIDTH_IN * DPI))

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
        center_x=w/2,
        top=h/2 - poly_height,
        bottom=h/2,
        num_sides=num_sides,
    )
    bottom_points = [(p[0], h - p[1]) for p in top_points]

    poly_measure = (num_sides - 2) * np.pi / num_sides
    side_measure = (np.pi - poly_measure) / 2

    t, b = top_points, bottom_points
    side_length = point_dist(t[0], t[1])
    inner_length = side_length * (np.sin(side_measure) / np.sin(poly_measure))
    min_y = min([p[1] for p in t])
    inner_top_points = gen_poly_points(
        center_x=w / 2,
        top=t[1][1] if num_sides % 2 == 1 else (min_y + inner_length * np.sin(side_measure)), #
        bottom=t[num_sides//2][1] - inner_length * np.sin(side_measure),
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

    for p in top_points:
        draw_circle(main_draw, center=p, fill=TR_COLOR, radius=20)

    for p in inner_top_points:
        draw_circle(main_draw, center=p, fill=TL_COLOR, radius=12)

    for p in bottom_points:
        draw_circle(main_draw, center=p, fill=TR_COLOR, radius=20)

    for p in inner_bottom_points:
        draw_circle(main_draw, center=p, fill=TL_COLOR, radius=12)

    t, b = top_points, bottom_points
    it, ib = inner_top_points, inner_bottom_points

    class Fold:
        PAUSE = 0
        CONTINUE = 1
        LIMIT = 2
        def __init__(self, series):
            points = [s for s in series if s not in [Fold.PAUSE, Fold.CONTINUE, Fold.LIMIT]]
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            is_top = points[0][1] < h//2

            self.radians = regularize_radians(np.arctan2(y2 - y1, x2 - x1))

            #print(series)

            self.edges = find_intersections(points[0], self.radians, right=w, bottom=h, both_ways=True)

            self.series = []
            self.series.append(find_intersections(points[0], regularize_radians(self.radians + np.pi), right=w, bottom=h, both_ways=False)[0])
            self.series.append(Fold.CONTINUE)
            self.series.extend(series)

            if self.series[-1] == Fold.CONTINUE:
                self.series[-1] = find_intersections(points[0], self.radians, right=w, bottom=h, both_ways=False)[0]
            
            for i in range(len(self.series)):
                if self.series[i] == Fold.LIMIT:
                    self.series[i] = find_intersections(points[0], self.radians, top=0 if is_top else h//2, right=w, bottom=h//2 if is_top else h, both_ways=False)[0]

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

            #print(self.lines)
            #print(f"{np.degrees(self.radians):.1f}°")

    folds = []
    for i in range(num_sides):
        next_i = (i + 1) % num_sides
        second_next_i = (i + 2) % num_sides

        """
        Step 4) Looks for homo lines.
        """
        x1, y1 = t[i]
        x2, y2 = t[next_i]
        last_radians = regularize_radians(np.arctan2(y2-y1, x2-x1))

        b_on_line = select_points_on_line(
            points_list=b, start_point=t[next_i], radians=last_radians
        )
        ib_on_line = select_points_on_line(
            points_list=ib, start_point=t[next_i], radians=last_radians
        )

        if len(b_on_line) > 1 and len(ib_on_line) > 1:
            series = [ib[ib_on_line[0][0]], Fold.PAUSE, ib[ib_on_line[-1][0]], Fold.CONTINUE]
            #print(f"homo:\t--> ib[{ib_on_line[0][0]}]    ib[{ib_on_line[-1][0]}] -> ")
        elif (
            len(b_on_line) > 0 and len(ib_on_line) == 0 
            and t[next_i] != b[b_on_line[-1][0]]
        ):
            closest_i = closest_point_index([t[i], t[next_i]], b[b_on_line[0][0]], indices=[i, next_i])
            series = [t[closest_i], Fold.PAUSE, b[b_on_line[-1][0]], Fold.CONTINUE]
            #print(f"homo:\t-->  t[{closest_i}]     b[{b_on_line[-1][0]}] -> ")
        elif len(b_on_line) == 0 and len(ib_on_line) > 0:
            series = [ib[ib_on_line[0][0]], Fold.PAUSE, ib[ib_on_line[-1][0]], Fold.CONTINUE]
            #print(f"homo:\t--> ib[{ib_on_line[0][0]}]    ib[{ib_on_line[-1][0]}] -> ")
        else:
            if last_radians < np.pi:
                series = [t[i], Fold.CONTINUE, t[next_i], Fold.CONTINUE, Fold.LIMIT]
            else:
                series = [t[next_i], Fold.CONTINUE, t[i], Fold.CONTINUE, Fold.LIMIT]
            #print(f"homo:\t-->  t[{i}] ->  t[{next_i}] ->  (limit)")

        folds.append(Fold(series))

    
        x1, y1 = b[i]
        x2, y2 = b[next_i]
        last_radians = regularize_radians(np.arctan2(y2-y1, x2-x1))

        t_on_line = select_points_on_line(
            points_list=t, start_point=b[next_i], radians=last_radians
        )
        it_on_line = select_points_on_line(
            points_list=it, start_point=b[next_i], radians=last_radians
        )

        series = None
        if len(t_on_line) > 1 and len(it_on_line) > 1:
            series = [it[it_on_line[0][0]], Fold.PAUSE, it[it_on_line[-1][0]], Fold.CONTINUE]
        elif (
            len(t_on_line) > 0 and len(it_on_line) == 0 
            and b[next_i] != t[t_on_line[-1][0]]
        ):
            closest_i = closest_point_index([b[i], b[next_i]], t[t_on_line[0][0]], indices=[i, next_i])
            series = [b[closest_i], Fold.PAUSE, t[t_on_line[-1][0]], Fold.CONTINUE]
        elif len(t_on_line) == 0 and len(it_on_line) > 0:
            series = [it[it_on_line[0][0]], Fold.PAUSE, it[it_on_line[-1][0]], Fold.CONTINUE]
        else:
            if last_radians > np.pi:
                series = [b[i], Fold.CONTINUE, b[next_i], Fold.CONTINUE, Fold.LIMIT]
            else:
                series = [b[next_i], Fold.CONTINUE, b[i], Fold.CONTINUE, Fold.LIMIT]

        if series is not None:
            folds.append(Fold(series))

        """
        Step 5) Looks for hetero lines.
        """
        x1, y1 = it[i]
        x2, y2 = it[next_i]
        last_radians = regularize_radians(np.arctan2(y2-y1, x2-x1))

        b_on_line = select_points_on_line(
            points_list=b, start_point=it[next_i], radians=last_radians,
        )
        ib_on_line = select_points_on_line(
            points_list=ib, start_point=it[next_i], radians=last_radians
        )

        if len(b_on_line) > 0 and len(ib_on_line) > 0:
            if last_radians < np.pi:
                series = [it[i], Fold.PAUSE, it[next_i], Fold.CONTINUE, ib[ib_on_line[0][0]], Fold.PAUSE, ib[ib_on_line[-1][0]], Fold.CONTINUE]
            else:
                series = [it[next_i], Fold.PAUSE, it[i], Fold.CONTINUE, ib[ib_on_line[0][0]], Fold.PAUSE, ib[ib_on_line[-1][0]], Fold.CONTINUE]
            #print(f"hetero:\t--> it[{i}]    it[{next_i}] -> ib[{ib_on_line[0][0]}]    ib[{ib_on_line[-1][0]}] -> ")
        elif len(b_on_line) > 0 and len(ib_on_line) == 0:
            closest_i = closest_point_index([it[i], it[next_i]], b[b_on_line[-1][0]], indices=[i, next_i])
            far_i = i if closest_i == next_i else next_i
            series = [it[far_i], Fold.PAUSE, it[closest_i], Fold.CONTINUE, Fold.LIMIT, Fold.PAUSE, b[b_on_line[-1][0]], Fold.CONTINUE]
            #print(f"hetero:\t--> it[{far_i}]    it[{closest_i}] ->  (limit)   b[{b_on_line[-1][0]}] -> ")
        elif len(b_on_line) == 0 and len(ib_on_line) > 0:
            if len(ib_on_line) == 1:

                series = [it[i], Fold.PAUSE, it[next_i], Fold.CONTINUE, ib[ib_on_line[0][0]]]
                #print(f"hetero:\t--> it[{i}]    it[{next_i}] -> ib[{ib_on_line[0][0]}]")
            else:
                series = [it[i], Fold.PAUSE, it[next_i], Fold.CONTINUE, ib[ib_on_line[0][0]], Fold.PAUSE, ib[ib_on_line[-1][0]]]
                #print(f"hetero:\t--> it[{i}]    it[{next_i}] -> ib[{ib_on_line[0][0]}]    ib[{ib_on_line[-1][0]}]")
        else:
            if last_radians < np.pi:
                series = [it[i], Fold.PAUSE, it[next_i], Fold.CONTINUE, Fold.LIMIT]
            else:
                series = [it[next_i], Fold.PAUSE, it[i], Fold.CONTINUE, Fold.LIMIT]
            #print(f"hetero:\t--> it[{i}]    it[{next_i}] ->  (limit)")

        folds.append(Fold(series))

        x1, y1 = ib[i]
        x2, y2 = ib[next_i]
        last_radians = regularize_radians(np.arctan2(y2-y1, x2-x1))

        t_on_line = select_points_on_line(
            points_list=t, start_point=ib[next_i], radians=last_radians,
        )
        it_on_line = select_points_on_line(
            points_list=it, start_point=ib[next_i], radians=last_radians
        )

        if len(t_on_line) > 0 and len(it_on_line) > 0:
            if last_radians > np.pi:
                series = [ib[i], Fold.PAUSE, ib[next_i], Fold.CONTINUE, it[it_on_line[0][0]], Fold.PAUSE, it[it_on_line[-1][0]], Fold.CONTINUE]
            else:
                series = [ib[next_i], Fold.PAUSE, ib[i], Fold.CONTINUE, it[it_on_line[0][0]], Fold.PAUSE, it[it_on_line[-1][0]], Fold.CONTINUE]
        elif len(t_on_line) > 0 and len(it_on_line) == 0:
            closest_i = closest_point_index([ib[i], ib[next_i]], t[t_on_line[-1][0]], indices=[i, next_i])
            far_i = i if closest_i == next_i else next_i
            series = [ib[far_i], Fold.PAUSE, ib[closest_i], Fold.CONTINUE, Fold.LIMIT, Fold.PAUSE, t[t_on_line[-1][0]], Fold.CONTINUE]
        elif len(t_on_line) == 0 and len(it_on_line) > 0:
            if len(it_on_line) == 1:
                series = [ib[i], Fold.PAUSE, ib[next_i], Fold.CONTINUE, it[it_on_line[0][0]]]
            else:
                series = [ib[i], Fold.PAUSE, ib[next_i], Fold.CONTINUE, it[it_on_line[0][0]], Fold.PAUSE, it[it_on_line[-1][0]]]
        else:
            if last_radians > np.pi:
                series = [ib[i], Fold.PAUSE, ib[next_i], Fold.CONTINUE, Fold.LIMIT]
            else:
                series = [ib[next_i], Fold.PAUSE, ib[i], Fold.CONTINUE, Fold.LIMIT]

        folds.append(Fold(series))

    for fold in folds:
        for a, b in fold.lines:
            # draws the main horizontal line.
            main_draw.line(
                (a[0], a[1], b[0], b[1]),
                fill=MAIN_LINE_COLOR,
                width=line_width,
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


def png_to_pdf(images: list, pdf_path, page_size, landscape: bool=False):
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


def save_star_diagram(
    page_size,
    poly_height=7.65,
    two_page: bool=False,
    landscape: bool=False,
    two_page_margin=0,
    is_metric: bool=False,
    print_margin_left=0,
    print_margin_right=0,
    print_margin_top=0,
    print_margin_bottom=0,
    num_sides=5,
):
    if landscape:
        page_w, page_h = np.max(page_size), np.min(page_size)
    else:
        page_w, page_h = np.min(page_size), np.max(page_size)
   
    # reportlab uses 72 DPI, so that is converted to use this script's DPI.
    margin_in = two_page_margin / 2.54 if is_metric else two_page_margin
    image_w_in = page_w / 72
    image_h_in = 2*(page_h/72 - margin_in) if two_page else page_h / 72
    image_w_cm, image_h_cm = image_w_in * 2.54, image_h_in * 2.54

    star_diagram = create_star_diagram(
        print_width=image_w_cm if is_metric else image_w_in,
        print_height=image_h_cm if is_metric else image_h_in,
        poly_height=poly_height,
        print_margin_left=print_margin_left,
        print_margin_right=print_margin_right,
        print_margin_top=print_margin_top,
        print_margin_bottom=print_margin_bottom,
        is_metric=is_metric,
        num_sides=num_sides,
    )

    star_diagram.save("star.png")

    
    images = []
    if two_page:
        # first half.
        crop = star_diagram.crop((0, 0, page_w/72 * DPI, page_h/72 * DPI))
        
        # creates a new blank image with a white background.
        new_image = Image.new("RGB", (round(page_w/72 * DPI), round(page_h/72 * DPI)), (255, 255, 255))
        new_image.paste(crop, (0, 0))
        images.append(new_image)

        # second half.
        beam_line_width = max(1, round(BEAM_LINE_WIDTH_IN * DPI))
        crop = star_diagram.crop((0, (page_h/72 - margin_in) * DPI + max(1, beam_line_width/2), image_w_in * DPI, image_h_in * DPI))

        new_draw = ImageDraw.Draw(crop)
        width, height = crop.size
        new_draw.line(
            (0, height - beam_line_width/2, width, height - beam_line_width/2),
            fill=BEAM_LINE_COLOR,
            width=beam_line_width,
        )

        # creates a new blank image with a white background.
        new_image = Image.new("RGB", (round(page_w/72 * DPI), round(page_h/72 * DPI)), (255, 255, 255))
        new_image.paste(crop, (0, 0))

        images.append(new_image)
    else:
        images.append(star_diagram)

    for i, image in enumerate(images):
        image.save(f"test-{i}.png")

    png_to_pdf(images, f"{num_sides}-star.pdf", page_size, landscape=landscape)

def main():
    page_size = reportlab.lib.pagesizes.letter

    i = 9

    save_star_diagram(
        page_size,
        poly_height=4.6,
        two_page=False,
        landscape=False,
        two_page_margin=3/4,
        is_metric=False,
        print_margin_left=0,
        print_margin_right=0,
        print_margin_top=0,
        print_margin_bottom=0,
        num_sides=i,
    )


if __name__ == "__main__":
    main()