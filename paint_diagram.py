from trig_math import point_dist

def blend_color(color_a, color_b, factor=0.5):
    diff = (b - a for a, b in zip(color_a, color_b))
    return tuple(round(a + d*factor) for a, d in zip(color_a, diff))


def draw_circle(draw, center, radius=20, fill=(0, 200, 100)):
    left_up = (center[0] - radius, center[1] - radius)
    right_down = (center[0] + radius, center[1] + radius)
    draw.ellipse([left_up, right_down], fill=fill)


def draw_gradient_line(draw, start, end, start_color, end_color, line_width):
    dist = point_dist(start, end)
    rise = end[1] - start[1]
    run = end[0] - start[0]
    slope = rise / run if run != 0 else 1000000
    
    num_units = int(dist) + 1
    step = 1 / (num_units - 1)

    for i in range(num_units):
        x, y = start[0] + i, start[1] + slope * x
        color = blend_color(start_color, end_color, factor=step * i)
        draw_circle(draw, center=(x, y), radius=line_width/2, fill_color=color)