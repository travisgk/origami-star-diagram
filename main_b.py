"""
This creates a PDF for a 7-pointed star that uses two pages.
Once printed, the second page is pasted right along the main horizontal line,
with the line not at all pasted over.

The second page should also have the end portion cut off too.
When this is cut, the thick line should be taken off with the cut.
"""

import reportlab.lib.pagesizes
from star_diagram import save_star_diagram


def main():
    page_size = reportlab.lib.pagesizes.letter  # US letter paper size.

    save_star_diagram(
        page_size,
        num_sides=7,
        poly_height=6.7,
        two_page=True,
        landscape=True,
        two_page_margin=3 / 4,
        is_metric=False,
    )


if __name__ == "__main__":
    main()
