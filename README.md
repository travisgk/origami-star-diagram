# origami-star-diagram
This little toy script generates a PDF that can be printed out and followed to created a 3D paper star box.
![](https://github.com/travisgk/origami-star-diagram/blob/main/example-outputs/stars-1.jpg?raw=true)
![](https://github.com/travisgk/origami-star-diagram/blob/main/example-outputs/stars-3.jpg?raw=true)
See an example PDF [here](https://github.com/travisgk/origami-star-diagram/blob/main/example-outputs/5-star.pdf).

<br>

# Setup
```
pip install numpy pillow reportlab
```

<br>

# "How do I make a diagram?"
1) Generate a PDF to print out.
   ```
   import reportlab.lib.pagesizes
   from star_diagram import save_star_diagram
   
   page_size = reportlab.lib.pagesizes.letter  # US letter paper size.
   save_star_diagram(
       page_size,
       num_sides=5,
       poly_height=5.1,
       two_page=False,
       landscape=False,
       is_metric=False,
   )
   ```
2) Open up the generated `5-star.pdf` in your preferred printing application. I highly recommend Chrome.
3) Set the scaling option to match the paper size and print.
4) If you're using two pages, paste the two pages together so that the second page's overlap runs along the main horizontal line (the midway line).
5) Trim off the extra paper on the second page, removing the thick bottom line with the final cut.

<br>

# "Okay, now how do I make this thing?"
1) Pull a corner to a dot of that has the same color, lining up the pulled corner of the paper with the thinner, lighter guide lines coming from the point.
2) Make the crease along where the thick line is shown (a light table can be very helpful here).
3) Repeat this for all the colored points on the diagram. For very large diagrams, some of these points might be outside of the diagram itself. In this case, look for any guide lines running off the page that match the corner's color and reasonably corresponds to the fold you're trying to make.
4) Fold the paper in half to get the main horizontal line that bisects the paper.
5) For vertical/horizontal folds, pull and match sides up with the short gray notches to make the fold along the thick vertical/horizontal line.
6) Fold the faint "beaming" lines coming from the star. These do not have any hint markers.
7) Assemble the star. More detailed instructions will be added later. [See this video to understand how it comes together](https://www.youtube.com/watch?v=gcgAhG46NYM).

<br>

# Notes
- Folds are often drawn so they stop and don't go over the halfway line.
- For glueing, it seems to work best to find the side of the star that exists along the hinge (one solid side, no overlapping), and then go around, skipping every other side, and glueing those sides together. Note the video for further technique.
