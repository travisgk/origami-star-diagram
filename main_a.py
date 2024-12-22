"""
This creates a PDF for a 5-pointed star that uses one page.
"""

import reportlab.lib.pagesizes
from star_diagram import save_star_diagram


def main():
    page_size = reportlab.lib.pagesizes.letter  # US letter paper size.

    save_star_diagram(
        page_size,
        num_sides=5,
        poly_height=5.1,
        two_page=False,
        landscape=False,
        is_metric=False,
        save_png_copy=True,
    )


if __name__ == "__main__":
    main()
