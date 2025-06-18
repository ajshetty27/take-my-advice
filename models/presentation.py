def build_impress_slides(sections):
    """
    Builds Impress.js-compatible slides with transformations.
    Each slide gets a unique position along a spiral path.
    """
    slides_html = ""
    x, y, z, rotate = 0, 0, 0, 0
    dx, dy, dz, dr = 1500, 1000, -500, 30

    for i, (title, bullets) in enumerate(sections):
        slide = f"""
        <div class="step" data-x="{x}" data-y="{y}" data-z="{z}" data-rotate="{rotate}">
            <h2>{title}</h2>
            <ul>
                {''.join(f'<li>{b}</li>' for b in bullets)}
            </ul>
        </div>
        """
        slides_html += slide
        x += dx
        y += dy if i % 2 == 0 else -dy
        z += dz
        rotate += dr

    return slides_html


def generate_impress_html(sections):
    slides = build_impress_slides(sections)
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Impress.js Presentation</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">
        <style>
            body {{
                background: #111;
                color: #f0f0f0;
                font-family: 'Inter', sans-serif;
            }}
            .step {{
                padding: 2em;
                width: 800px;
                height: 600px;
                box-sizing: border-box;
                border-radius: 12px;
                background: rgba(0,0,0,0.4);
                box-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
            }}
            h2 {{
                color: #f5f5f5;
                font-size: 2.5em;
                margin-bottom: 0.5em;
            }}
            ul {{
                font-size: 1.3em;
                line-height: 1.6;
            }}
        </style>
    </head>
    <body>
        <div id="impress">
            {slides}
        </div>

        <script src="https://cdn.jsdelivr.net/npm/impress.js@1.1.0/js/impress.min.js"></script>
        <script>impress().init();</script>
    </body>
    </html>
    """
