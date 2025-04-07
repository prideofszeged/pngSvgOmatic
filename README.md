# PNG to Vector SVG Converter

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](<Your Streamlit Cloud URL - Optional>) <!-- Optional: Add link if deployed -->

A web-based tool to convert raster PNG images into vector SVG files using color segmentation and contour tracing. Ideal for reducing manual tracing work in vector editing software.

## Features

*   **Simple Web UI:** Built with Streamlit for ease of use.
*   **PNG Upload:** Upload your source images directly.
*   **Color Segmentation:** Identifies dominant color regions using K-Means.
*   **Contour Tracing:** Creates vector paths from color boundaries using OpenCV.
*   **Output Modes:** Generate SVGs with `fill` (solid shapes) or `stroke` (outlines).
*   **Adjustable Parameters:** Control number of colors, minimum shape area, stroke width, and optional pre-blurring.
*   **Live Preview:** See the generated SVG directly in the app.
*   **Direct Downloads:** Save results as `.svg` or `.zip` files.

<!-- Optional: Add a screenshot
## Screenshot

![App Screenshot](link/to/your/screenshot.png)
-->

## Requirements

*   Python 3.8+
*   Git

## Installation & Setup

Use Python's `venv` for environment isolation and `uv` for fast package installation.

1.  **Clone Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory-name>
    ```

2.  **Create & Activate Virtual Environment:**
    ```bash
    # Create environment
    python -m venv .venv

    # Activate (choose one based on your OS/shell)
    # Windows (cmd): .venv\Scripts\activate.bat
    # Windows (PowerShell): .venv\Scripts\Activate.ps1
    # macOS/Linux (bash/zsh): source .venv/bin/activate
    ```
    *(Your terminal prompt should now show `(.venv)`)*

3.  **Install `uv` Package Manager:**
    ```bash
    # Run inside the activated environment
    pip install uv
    ```

4.  **Install Dependencies:**
    ```bash
    # Run inside the activated environment
    uv pip install -r requirements.txt
    ```
    *(If `requirements.txt` is missing, run `uv pip install streamlit numpy opencv-python scikit-learn Pillow svgwrite` then `uv pip freeze > requirements.txt`)*

## Usage

1.  **Ensure Environment is Active:** If needed, reactivate using the command from Step 2 above.
2.  **Run Streamlit App:**
    ```bash
    streamlit run app.py
    ```
3.  **Open in Browser:** Access the local URL provided (e.g., `http://localhost:8501`).
4.  **Convert Images:**
    *   Upload a PNG file.
    *   Adjust settings in the sidebar.
    *   Click "Convert to SVG".
    *   Preview the result.
    *   Download the `.svg` or `.zip` file using the sidebar buttons.
5.  **Stop the App:** Press `Ctrl+C` in the terminal.
6.  **(Optional) Deactivate Environment:** `deactivate`

## Parameters Explained

*   **Number of Colors:** Controls color detail vs. complexity. More colors = more detail & paths.
*   **Output Mode:**
    *   `fill`: Creates solid, filled vector shapes.
    *   `stroke`: Creates outlined vector shapes.
*   **Stroke Width:** Thickness of lines in `stroke` mode (pixels).
*   **Minimum Area:** Ignores detected color regions smaller than this (pixels) to reduce noise.
*   **Gaussian Blur Radius:** Pre-blurs the image (0 = off). Helps smooth jagged edges or noise before vectorization.

## Usage Guide & Tips

*   **Choosing the Right Method:**
    *   Use **Color Segmentation** for images with distinct, flat color areas (logos, cartoons, simple illustrations, icons). It works best when colors are clearly separated.
    *   Use **Edge Detection** for line art, sketches, technical drawings, or when you primarily need the outlines of shapes, regardless of their fill color.
*   **Input Image Quality:**
    *   **Prefer PNG:** Lossless PNGs generally work best, especially for segmentation. JPG compression artifacts can sometimes create unwanted noise or fuzzy boundaries.
    *   **Clear Boundaries:** Images with well-defined edges and less noise/gradients will convert more cleanly.
    *   **GIFs:** Only the first frame of an animated GIF is processed.
*   **Preprocessing with Blur:**
    *   A small **Gaussian Blur Radius** (e.g., 0.5-1.5) applied *before* vectorization can significantly smooth jagged pixel edges or reduce minor noise, leading to cleaner vector paths.
    *   Experiment carefully: too much blur will merge details and reduce accuracy. Start with 0 and increase slightly if needed.
*   **Performance with Downscaling:**
    *   Keep **Downscale Large Images** enabled for faster processing, especially with high-resolution inputs. The default `Max Dimension` (e.g., 1024px) is often sufficient.
    *   Disable downscaling only if you absolutely need to preserve every detail from a very large image and are prepared for longer processing times.
*   **Color Segmentation Tuning:**
    *   **Number of Colors:** Start with a moderate number (e.g., 8-16). Increase it if important color details are missed. Decrease it if the result is too complex or noisy. Finding the right balance is key.
    *   **Mode:** `fill` is standard for recreating colored shapes. `stroke` can be used for artistic outlining based on color regions.
    *   **Min Region Area:** Increase this value to eliminate small, isolated "speckle" artifacts, particularly common in noisy images or complex gradients.
*   **Edge Detection Tuning:**
    *   **Canny Thresholds:** These significantly impact edge detection. Lower `Threshold 1` and `Threshold 2` values detect more (potentially weaker) edges. Higher values detect only stronger edges. A common starting point is T2 ≈ 2*T1 or 3*T1. Experiment to find the best balance for your image.
    *   **Min Edge Length Area:** Increase this to discard very short, potentially irrelevant edge segments.
    *   **Stroke Width/Color:** Adjust for desired line thickness and visibility.
*   **Iterate and Refine:**
    *   Vectorization is often an iterative process. Don't expect perfection on the first attempt.
    *   Adjust one parameter at a time (e.g., blur, number of colors/thresholds, min area) and reconvert to see the effect.
*   **Post-Processing in Vector Software:**
    *   The generated SVG is a starting point. Use vector editing software (like Inkscape, Adobe Illustrator, Figma, Affinity Designer) to:
        *   Simplify or smooth paths further.
        *   Combine overlapping shapes.
        *   Correct colors or add gradients.
        *   Remove unwanted small elements.

MIT License (See LICENSE file for details)
