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


MIT License (See LICENSE file for details)
