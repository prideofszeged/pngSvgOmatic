# Image to Vector SVG Converter

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](<Your Streamlit Cloud URL - Optional>) <!-- Optional: Add link if deployed -->

A web-based tool to convert raster images (PNG, JPG, GIF) into vector SVG files using either **Color Segmentation** or **Edge Detection**. Ideal for reducing manual tracing work in vector editing software.

## Features

*   **Simple Web UI:** Built with Streamlit for ease of use.
*   **Multi-Format Upload:** Supports PNG, JPG/JPEG, and GIF (first frame) uploads.
*   **Dual Vectorization Methods:**
    *   **Color Segmentation:** Identifies dominant color regions using K-Means (MiniBatch variant).
    *   **Edge Detection:** Creates outlines based on detected edges using the Canny algorithm.
*   **Contour Tracing:** Creates vector paths from boundaries using OpenCV.
*   **Output Modes (Color Segmentation):** Generate SVGs with `fill` (solid shapes) or `stroke` (color outlines).
*   **Configurable Parameters:** Fine-tune the conversion with controls for:
    *   Number of colors (segmentation)
    *   Canny thresholds (edge detection)
    *   Minimum shape/edge area
    *   Stroke width and color
    *   Optional pre-blurring
    *   Optional image downscaling
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
    git clone <your-repository-url> # Replace with your actual repo URL
    cd <repository-directory-name> # Navigate into the cloned project folder
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
    *(If `requirements.txt` is missing: first run `uv pip install streamlit numpy opencv-python scikit-learn Pillow svgwrite`, then generate the file with `uv pip freeze > requirements.txt`)*

## Usage

1.  **Ensure Environment is Active:** If needed, reactivate using the command from Step 2 above.
2.  **Run Streamlit App:**
    ```bash
    streamlit run app.py
    ```
3.  **Open in Browser:** Access the local URL provided (e.g., `http://localhost:8501`).
4.  **Convert Images:**
    *   Upload an image file (PNG, JPG, GIF) using the file uploader.
    *   In the sidebar, select the desired `Vectorization Method` ('Color Segmentation' or 'Edge Detection').
    *   Adjust the general settings (`Gaussian Blur Radius`, `Downscale Large Images`).
    *   Configure the method-specific settings that appear below (e.g., `Number of Colors` and `Output Mode` for Color Segmentation, or `Canny Thresholds` and `Edge Stroke Color` for Edge Detection).
    *   Click the "Convert to SVG" button.
    *   Wait for processing and view the preview.
    *   Download the resulting `.svg` or `.zip` file using the sidebar buttons.
5.  **Stop the App:** Press `Ctrl+C` in the terminal.
6.  **(Optional) Deactivate Environment:** `deactivate`

## Usage Guide & Tips

*   **Choosing the Right Method:**
    *   Use **Color Segmentation** for images with distinct, flat color areas (logos, cartoons, simple illustrations, icons). It preserves color information and creates filled or stroked regions.
    *   Use **Edge Detection** for line art, sketches, technical drawings, or when you primarily need the outlines of shapes, regardless of their fill color. This method ignores color and focuses purely on lines.
*   **Input Image Quality:**
    *   **Prefer Lossless:** PNGs generally work best, especially for Color Segmentation, as JPG compression artifacts can create unwanted noise or fuzzy boundaries.
    *   **Clear Boundaries:** Images with well-defined edges and less noise/gradients will convert more cleanly using either method.
    *   **GIFs:** Only the first frame of an animated GIF is processed. Ensure it's converted to RGBA internally.
*   **Preprocessing with Blur (`Gaussian Blur Radius`):**
    *   Applies blur *before* vectorization. A small radius (e.g., 0.5-1.5) can significantly smooth jagged pixel edges or reduce minor noise, leading to cleaner vector paths in both methods.
    *   Experiment carefully: too much blur will merge details and reduce accuracy. Start with 0 and increase slightly if needed.
*   **Performance (`Downscale Large Images`, `Max Dimension`):**
    *   Keep **Downscale Large Images** enabled for faster processing, especially with high-resolution inputs. Reducing the `Max Dimension` (largest side in pixels) speeds things up considerably.
    *   Disable downscaling only if you absolutely need maximum detail from a very large image and accept longer processing times.
*   **Color Segmentation Tuning:**
    *   **`Number of Colors`:** Start moderately (e.g., 8-16). Increase to capture more color details; decrease if the result is too noisy or overly complex. Finding the right balance is key.
    *   **`Output Mode`:** `fill` creates standard solid color shapes. `stroke` creates colored outlines based on the detected color regions.
    *   **`Min Region Area`:** Increase this pixel value to eliminate small, isolated "speckle" artifacts from noise or complex gradients.
    *   **`Stroke Width` (Stroke Mode):** Sets line thickness when `Output Mode` is `stroke`.
*   **Edge Detection Tuning:**
    *   **`Canny Threshold 1 / Threshold 2`:** These significantly impact edge detection sensitivity. Lower values detect more (potentially weaker) edges; higher values detect only stronger edges. (Tip: T2 ≈ 2*T1 or 3*T1 is often a good starting range). Experiment!
    *   **`Min Edge Length Area`:** Increase this value (approx. pixel area of the contour) to discard very short, potentially irrelevant edge segments.
    *   **`Stroke Width` / `Edge Stroke Color`:** Adjust for desired line thickness and color in the final SVG outline.
*   **Iterate and Refine:**
    *   Vectorization often requires experimentation. Adjust one parameter at a time (e.g., blur, number of colors/thresholds, min area) and reconvert to observe the effect.
*   **Post-Processing in Vector Software:**
    *   The generated SVG is often a great starting point. Use vector editing software (Inkscape, Illustrator, Figma, Affinity Designer) to:
        *   Simplify or smooth paths further.
        *   Combine or merge overlapping shapes.
        *   Correct or change colors/strokes.
        *   Remove unwanted small elements manually.

## License

MIT License (See LICENSE file for details)
