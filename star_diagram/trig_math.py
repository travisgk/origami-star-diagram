import numpy as np

AXIS_PRECISION = np.pi / 180

def regularize_radians(radians):
    STEP = 2 * np.pi

    while radians < 0:
        radians += STEP
    while radians >= STEP:
        radians -= STEP

    return radians


def calc_perp_radians(radians):
    """Returns the angle measure that's perpendicular to the given measure."""
    result = radians + np.pi / 2
    result = regularize_radians(result)
    return result


def gen_poly_points(center_x, top, bottom, num_sides: int=5, upward: bool=True):
    """
    Returns a list of points scaled to fit to the y-coords of top and bottom.
    
    Parameters:
        center_x (num): the X-coordinate that the polygon will be centered at.
        top (num): the Y-coordinate that will be the min Y-coord of all points.
        bottom (num): the Y-coordinate that will be the max Y-coord of all ponits.
        upward (bool): if True, the polygon is drawn with its top point upward.
    
    Returns:
        list: 2D tuples of points to construct a perfect polygon.
    """

    step = np.pi*2 / num_sides
    offset = 0 if num_sides % 2 == 1 else step / 2

    # calculates the points by rotating in a circle.
    points = []
    for i in range(num_sides):
        angle = step*i + offset - np.pi/2 + (0 if upward else step/2)
        points.append((np.cos(angle), np.sin(angle)))

    min_y = np.min([p[1] for p in points])
    max_y = np.max([p[1] for p in points])

    # moves the points so the lowest point's Y-coord is at 0.
    points = [(p[0], p[1] - min_y) for p in points]

    # scales the points to fit between the given top and bottom Y-coords,
    # then returns the results.
    span = bottom - top
    diff = max_y - min_y
    scale_factor = span / diff

    return [
        (
            center_x + p[0] * scale_factor,
            top + p[1] * scale_factor,
        )
        for p in points
    ]


def point_dist(a, b):
    """ Returns the 2D distance between 2D points <a> and <b>. """
    return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)


def angle_is_vertical(radians):
    return (
        np.pi/2 - AXIS_PRECISION < radians < np.pi/2 + AXIS_PRECISION
        or 3*np.pi/2 - AXIS_PRECISION < radians < 3*np.pi/2 + AXIS_PRECISION
    )


def select_points_on_line(
    points_list: list,
    start_point,
    radians,
    both_ways: bool=True,
    get_indices: bool=True,
):
    MATCH_DIST = 1.0 # within 1 px.

    if both_ways:
        angles = [radians, regularize_radians(radians + np.pi)]
    else:
        angles = [radians]

    hits = []

    for angle in angles:
        radians = regularize_radians(angle)
        if np.pi/2 - AXIS_PRECISION < radians < np.pi/2 + AXIS_PRECISION:
            # vertical line with increasing Y. around 90°.
            hits.extend([
                (i, radians) if get_indices else (p, radians) 
                for i, p in enumerate(points_list) if p[1] < start_point[1] 
                and abs(start_point[0] - p[0]) < MATCH_DIST
            ])

        elif 3*np.pi/2 - AXIS_PRECISION < radians < 3*np.pi/2 + AXIS_PRECISION:
            # vertical line with decreasing Y. around 270°.
            hits.extend(
                [
                    (i, radians) if get_indices else (p, radians) 
                    for i, p in enumerate(points_list) if p[1] > start_point[1] 
                    and abs(start_point[0] - p[0]) < MATCH_DIST
                ]
            )
        
        elif np.pi - AXIS_PRECISION < radians < np.pi + AXIS_PRECISION:
            # horizontal line with decreasing X. around 180°.
            hits.extend(
                [
                    (i, radians) if get_indices else (p, radians) 
                    for i, p in enumerate(points_list) if p[0] > start_point[0] 
                    and abs(start_point[1] - p[1]) < MATCH_DIST
                ]
            )
                
        elif 2*np.pi - AXIS_PRECISION < radians or radians < AXIS_PRECISION:
            # horizontal line with increasing X. around 0°.
            hits.extend(
                [
                    (i, radians) if get_indices else (p, radians) 
                    for i, p in enumerate(points_list) if p[0] < start_point[0] 
                    and abs(start_point[1] - p[1]) < MATCH_DIST
                ]
            )

        else:
            x_is_increasing = radians > 3 * np.pi/2 or radians < np.pi/2 # going right
            y_is_increasing = 0 < radians < np.pi # going "down".

            slope = np.tan(radians)
            s_x, s_y = start_point
            for i, p in enumerate(points_list):
                if (
                    (
                        (x_is_increasing and p[0] > s_x) 
                        or (not x_is_increasing and p[0] < s_x)
                    )
                    and (
                        (y_is_increasing and p[1] > s_y) 
                        or (not y_is_increasing and p[1] < s_y)
                    )
                ):
                    x, point_y = p
                    line_y = slope * (x - s_x) + s_y
                    residual = point_y - line_y

                    if abs(residual) < MATCH_DIST:
                        hits.append((i, radians) if get_indices else (p, radians))

    if get_indices:
        # sorts the list with the tuples 
        # having their first element being an index.
        hits.sort(key=lambda x: point_dist(start_point, points_list[x[0]]))
    else:
        # sorts the list with the tuples having
        # their first element being a 2D point.
        hits.sort(key=lambda x: point_dist(start_point, x[0]))

    return hits


def find_intersections(
    p, radians, left=0, top=0, right=None, bottom=None, both_ways: bool=True
):
    """
    Returns a list of points along a rectangle that are hit 
    by a ray/line coming out of point <p> at measure of <radians>,
    where the ray/line doesn't continue to go through the rectangle.

    Parameters:
        p (2D tuple): the 2D point that lies on the ray/line.
        radians (num): the angle measure of the ray/line in radians.
        left (num): the leftward bound of the rectangle (X-minimum).
        top (num): the upward bound of the rectangle (Y-minimum).
        right (num): the rightward bound of the rectangle (X-maximum).
        bottom (num): the downward bound of the rectangle (Y-maximum).
        both_ways (bool): if True, the 180° flip of the angle 
                          will be checked too.
    """

    # determines which angles to check. 
    x, y = p
    in_rectangle = (left <= x <= right) and (top <= y <= bottom)

    angles = [regularize_radians(radians)]
    if both_ways:
        angles.append(regularize_radians(radians + np.pi))

    hits = []

    for angle in angles:
        radians = regularize_radians(angle)
        x_is_increasing = radians > 3 * np.pi/2 or radians < np.pi/2 # going right
        y_is_increasing = 0 < radians < np.pi # going "down".

        """ Step 2) Finds the axes the line is expected to intersect. """
        if np.pi/2 - AXIS_PRECISION < radians < np.pi/2 + AXIS_PRECISION:
            # vertical line with increasing Y. around 90°.
            if left <= x <= right:
                hits.append((x, top if y < top else bottom))

        elif 3*np.pi/2 - AXIS_PRECISION < radians < 3*np.pi/2 + AXIS_PRECISION:
            # vertical line with decreasing Y. around 270°.
            if left <= x <= right:
                hits.append((x, bottom if y > bottom else top))
        
        elif np.pi - AXIS_PRECISION < radians < np.pi + AXIS_PRECISION:
            # horizontal line with decreasing X. around 180°.
            if top <= y <= bottom:
                hits.append((right if x > right else left, y))
                
        elif 2*np.pi - AXIS_PRECISION < radians or radians < AXIS_PRECISION:
            # horizontal line with increasing X. around 0°.
            if top <= y <= bottom:
                hits.append((left if x < left else right, y))
             
        else:
            # line is not orthogonal.
            # ---
            # the X-coord that could be collided against is found. 
            if x < left:
                x_limit = left
            elif x >= right:
                x_limit = right
            else:
                x_limit = right if x_is_increasing else left

            # the Y-coord that could be collided against is found. 
            if y < top:
                y_limit = top
            elif y >= bottom:
                y_limit = bottom
            else:
                y_limit = bottom if y_is_increasing else top

            # finds the intercepts with each X/Y limit.
            slope = np.tan(radians)
            result_x = (y_limit - p[1]) / slope + p[0] # a; y = m(x - x1) + y1
            result_y = slope * (x_limit - p[0]) + p[1] # b; x = (y - y1)/m + x1

            a, b = (result_x, y_limit), (x_limit, result_y)

            a_is_valid = left <= result_x <= right
            b_is_valid = top <= result_y <= bottom

            if a_is_valid and b_is_valid:
                a_dist, b_dist = point_dist(p, a), point_dist(p, b)
                hits.append(a if a_dist < b_dist else b)
            elif a_is_valid:
                hits.append(a)
            elif b_is_valid:
                hits.append(b)

    return hits


"""def calc_line(image_width, image_height, start, end, cap_start: bool=False):
    OOB = 100 # how far a line is extended off paper.
    w, h = image_width, image_height
    angle = np.arctan2(end[1] - start[1], end[0] - start[0])  #  start to end.

    # handles occurrences where the line is practically vertical/horizontal.
    for i in range(4):
        current_axis = i * np.pi/2
        if angle - AXIS_PRECISION < current_axis < angle + AXIS_PRECISION:
            # this angle measure is close enough to be considered orthogonal.
            if i in [0, 2]:
                # line is horizontal.
                if not cap_start:
                    # extends the start point off the paper.
                    start = (-OOB if start[0] < end[0] else w + OOB, start[1])
                
                # extends the end point off the paper.
                end = (-OOB if end[0] < start[0] else w + OOB, start[1])

            else:
                # line is vertical.
                if not cap_start:
                    # extends the start point off the paper.
                    start = (start[0], -OOB if start[1] < end[1] else h + OOB)
                   
                # extends the end point off the paper.
                end = (start[0], -OOB if end[1] < start[1] else h + OOB)
            
            # returns the calculated orthogonal line.
            return (start, end)

    # by this point, it's known that the angle is a non-orthogonal angle.
    slope = np.tan(angle)

    def extend_point_off_paper(point, radians):
        hit = find_intersections(point, radians, right=w, bottom=h)
        return (hit[0] + np.cos(radians)*OOB, hit[1] + np.sin(radians)*OOB)

    if not cap_start:
        flipped = regularize_radians(angle + np.pi)
        start = extend_point_off_paper(start, flipped)

    end = extend_point_off_paper(end, angle)

    return (start, end)"""


def reflect_across_line(p, a, b, image_width, image_height):
    """
    Returns the point <p> reflected by 
    the line defined by 2D points <a> and <b>.
    """
    w, h = image_width, image_height

    # strips extra info.
    p = np.array(p[:2])
    a = np.array(a[:2])
    b = np.array(b[:2])

    ab = b - a # from <a> to <b>.    
    ap = p - a # from <a> to <p>.

    # projects the vector <ap> onto the vector <ab>.
    ab_normalized = ab / np.linalg.norm(ab)
    projection_length = np.dot(ap, ab_normalized)
    projection = a + projection_length * ab_normalized
    
    # reflects point <p> across the line.
    reflection = tuple(2 * projection - p)

    return reflection


def closest_point_index(points, main_point, indices=None):
    """
    Finds the index of the point in the list that is closest to the main_point.

    Parameters:
        points (list of tuple): List of 2D points represented as (x, y).
        main_point (tuple): The reference 2D point represented as (x, y).

    Returns:
        int: Index of the closest point in the list.
    """
    if not points:
        raise ValueError("The points list cannot be empty.")
    
    closest_index = 0
    min_distance = point_dist(points[0], main_point)

    for i in range(1, len(points)):
        distance = point_dist(points[i], main_point)
        if distance < min_distance:
            min_distance = distance
            closest_index = i

    return closest_index if indices is None else indices[closest_index]


def interpolate_rgb(color_a, color_b, factor=0.5):
    return tuple([round(a + factor*(b - a)) for a, b in zip(color_a, color_b)])
