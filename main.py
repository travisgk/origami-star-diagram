import reportlab.lib.pagesizes
from star_diagram import save_star_diagram

def main():
    page_size = reportlab.lib.pagesizes.letter

    save_star_diagram(
        page_size,
        poly_height=4.9,
        two_page=False,
        landscape=False,
        two_page_margin=3/4,
        is_metric=False,
        print_margin_left=0,
        print_margin_right=0,
        print_margin_top=0,
        print_margin_bottom=0,
        num_sides=7,
    )


if __name__ == "__main__":
    main()