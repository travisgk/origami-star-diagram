import os
import tempfile
import numpy as np
from PIL import Image, ImageDraw
from reportlab.pdfgen import canvas
from .draw_diagram import *


def png_to_pdf(images: list, pdf_path, page_size, landscape: bool = False):
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
    num_sides: int = 5,
    poly_height=7.65,
    two_page: bool = False,
    landscape: bool = False,
    two_page_margin=0,
    print_margin_left=0,
    print_margin_right=0,
    print_margin_top=0,
    print_margin_bottom=0,
    is_metric: bool = False,
    out_path=None,
):
    """
    Saves an origami diagram to file that can be used to quickly
    turn the piece of paper into a 3D star.

    Parameters:
        page_size: a ReportLab page size. Or the page size in px with DPI being 72.
        num_sides (int): the number of sides the star will have.
        poly_height (num): the distance from the top to bottom of the polygon.
        two_page (bool): if True, the star will be made by glueing two sheets together.
        landscape (bool): if True, the paper will be oriented in landscape.
        two_mage_margin (num): in cases with two pages,
                               this is the length of overlap where the sheets will be pasted together.
        print_margin_left (num): the length cut off the left side of the output.
        print_margin_right (num): the length cut off the right side of the output.
        print_margin_top (num): the length cut off the top side of the output.
        print_margin_bottom (num): the length cut off the bottom side of the output.
        is_metric (bool): if True, all given measurements are treated as cm.
        out_path (str): output path for the PDF. if None, out path will become a program default.
    """

    if landscape:
        page_w, page_h = np.max(page_size), np.min(page_size)
    else:
        page_w, page_h = np.min(page_size), np.max(page_size)

    # reportlab uses 72 DPI, so that is converted to use this script's DPI.
    margin_in = two_page_margin / 2.54 if is_metric else two_page_margin
    image_w_in = page_w / 72
    image_h_in = 2 * (page_h / 72 - margin_in) if two_page else page_h / 72
    image_w_cm, image_h_cm = image_w_in * 2.54, image_h_in * 2.54

    poly_height_in = poly_height / 2.54 if is_metric else poly_height
    height_cap_in = image_h_in / 2
    height_cap_cm = height_cap_in * 2.54
    if poly_height_in >= height_cap_in:
        measure = (
            f"{height_cap_cm:.1f} cm" if is_metric else f"{height_cap_in:.1f} inches"
        )
        print(
            "The poly_height must fit within the bounds of the page.\n"
            f"With this paper size, it must be below {measure}.\n"
        )
        return

    star_diagram = create_star_diagram(
        num_sides=num_sides,
        print_width=image_w_cm if is_metric else image_w_in,
        print_height=image_h_cm if is_metric else image_h_in,
        poly_height=poly_height,
        print_margin_left=print_margin_left,
        print_margin_right=print_margin_right,
        print_margin_top=print_margin_top,
        print_margin_bottom=print_margin_bottom,
        is_metric=is_metric,
    )

    star_diagram.save("star.png")

    images = []
    if two_page:
        # first half.
        crop = star_diagram.crop((0, 0, page_w / 72 * DPI, page_h / 72 * DPI))

        # creates a new blank image with a white background.
        new_image = Image.new(
            "RGB", (round(page_w / 72 * DPI), round(page_h / 72 * DPI)), (255, 255, 255)
        )
        new_image.paste(crop, (0, 0))
        images.append(new_image)

        # second half.
        beam_line_width = max(1, round(BEAM_LINE_WIDTH_IN * DPI))
        crop = star_diagram.crop(
            (
                0,
                (page_h / 72 - margin_in) * DPI + max(1, beam_line_width / 2),
                image_w_in * DPI,
                image_h_in * DPI,
            )
        )

        # creates a new blank image with a white background.
        new_image = Image.new(
            "RGB", (round(page_w / 72 * DPI), round(page_h / 72 * DPI)), (255, 255, 255)
        )
        new_image.paste(crop, (0, 0))

        new_draw = ImageDraw.Draw(new_image)
        width, height = crop.size

        # the ending line on a two-pager should be cut off.
        new_draw.line(
            (0, height + beam_line_width / 2, width, height + beam_line_width / 2),
            fill=BEAM_LINE_COLOR,
            width=beam_line_width,
        )

        images.append(new_image)
    else:
        images.append(star_diagram)

    temp_paths = []
    for i, image in enumerate(images):
        with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
            temp_paths.append(temp_file.name)
        image.save(temp_paths[-1])

    if out_path is None:
        out_path = f"{num_sides}-star.pdf"
    png_to_pdf(images, out_path, page_size, landscape=landscape)

    for temp_path in temp_paths:
        os.remove(temp_path)

    print(
        f"An origami template for a star with {num_sides} "
        f"sides has been saved to {out_path}.\n"
    )
