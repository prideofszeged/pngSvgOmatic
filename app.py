#!/usr/bin/env python3
"""
Streamlit frontend for converting PNG images to vector SVGs using
color segmentation and contour tracing.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
import svgwrite
import zipfile
import warnings
from typing import Tuple, List, Optional, Union
import io # Used for in-memory file handling
import streamlit as st

# --- Constants (from backend script) ---
DEFAULT_N_COLORS = 16
DEFAULT_MIN_AREA = 20
DEFAULT_STROKE_WIDTH = 1.0
DEFAULT_MODE = 'fill'
DEFAULT_BLUR_RADIUS = 0.0
KMEANS_N_INIT = 10
RANDOM_STATE = 42

# --- Backend Logic Functions (Copied/adapted from previous script) ---
# Suppress KMeans ConvergenceWarning for cleaner output
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def contour_to_svg_path(contour: np.ndarray) -> str:
    if not isinstance(contour, np.ndarray) or contour.ndim != 3 or contour.shape[1] != 1 or contour.shape[2] != 2:
        st.warning(f"Unexpected contour format encountered: shape={contour.shape}. Skipping.")
        return ""
    if len(contour) < 2: return ""
    path_data = f"M {contour[0, 0, 0]},{contour[0, 0, 1]}"
    if len(contour) > 1:
        path_data += " L " + " ".join(f"{pt[0][0]},{pt[0][1]}" for pt in contour[1:])
    path_data += " Z"
    return path_data

def apply_optional_blur(image: Image.Image, blur_radius: float) -> Image.Image:
    if blur_radius > 0:
        return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return image

def segment_image(image: Image.Image, n_colors: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    try:
        img_np = np.array(image)
        if img_np.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        elif img_np.shape[2] == 3:
            img_rgb = img_np
        else:
            raise ValueError(f"Unsupported image format: needs RGB or RGBA, got shape {img_np.shape}")

        original_shape = img_rgb.shape
        pixels = img_rgb.reshape((-1, 3))

        kmeans = KMeans(n_clusters=n_colors, n_init=KMEANS_N_INIT, random_state=RANDOM_STATE)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_
        centers_int = centers.round().astype(np.uint8)
        label_map = labels.reshape(original_shape[:2])

        return label_map, centers_int, original_shape

    except Exception as e:
        st.error(f"Error during KMeans segmentation: {e}")
        raise # Re-raise to stop processing

def create_svg_content(
    label_map: np.ndarray,
    centers: np.ndarray,
    shape: Tuple[int, int, int],
    mode: str,
    min_area: int,
    stroke_width: float
) -> str:
    """Generates SVG content as a string."""
    height, width = shape[:2]
    # Use io.StringIO to build the SVG string in memory
    svg_io = io.StringIO()
    dwg = svgwrite.Drawing(profile='tiny', size=(f"{width}px", f"{height}px"))

    num_paths_added = 0
    for i, center_color in enumerate(centers):
        mask = np.zeros(label_map.shape, dtype=np.uint8)
        mask[label_map == i] = 255
        try:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue

            contour_index = 0
            while contour_index >= 0:
                contour = contours[contour_index]
                area = cv2.contourArea(contour)

                if area >= min_area:
                    path_data = contour_to_svg_path(contour)
                    if path_data:
                        svg_color = svgwrite.rgb(*center_color)
                        path_attrs = {'d': path_data}
                        if mode == 'fill':
                            path_attrs['fill'] = svg_color
                            path_attrs['stroke'] = 'none'
                        else: # stroke mode
                            path_attrs['fill'] = 'none'
                            path_attrs['stroke'] = svg_color
                            path_attrs['stroke-width'] = stroke_width
                            path_attrs['stroke-linecap'] = 'round'
                            path_attrs['stroke-linejoin'] = 'round'

                        dwg.add(dwg.path(**path_attrs))
                        num_paths_added += 1
                # Move to next contour at the same level
                contour_index = hierarchy[0, contour_index, 0] if hierarchy is not None else -1

        except cv2.error as e:
            st.warning(f"OpenCV error finding contours for cluster {i}: {e}")
            continue
        except Exception as e:
            st.error(f"Error during contour processing for cluster {i}: {e}")
            raise # Re-raise critical errors

    # Use dwg.tostring() to get the SVG XML as a string
    svg_string = dwg.tostring()
    st.info(f"Generated SVG with {num_paths_added} paths.")
    return svg_string


def create_zip_content(svg_content: str, svg_filename: str) -> bytes:
    """Creates a ZIP archive in memory containing the SVG content."""
    zip_buffer = io.BytesIO()
    try:
        with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            # Add SVG content to the zip file using arcname
            zipf.writestr(svg_filename, svg_content)
        st.info(f"Created ZIP containing '{svg_filename}'.")
        # Return the ZIP content as bytes
        return zip_buffer.getvalue()
    except Exception as e:
        st.error(f"Error creating ZIP file in memory: {e}")
        raise


# --- Streamlit UI ---

st.set_page_config(page_title="PNG to SVG Converter", layout="wide")

st.title("🖼️ PNG to Vector SVG Converter")
st.markdown("Upload a PNG image to convert it into an SVG file using color segmentation. "
            "Adjust the parameters to control the output.")

# --- Initialize Session State ---
# Store results so download buttons work after reruns
if 'svg_content' not in st.session_state:
    st.session_state['svg_content'] = None
if 'zip_content' not in st.session_state:
    st.session_state['zip_content'] = None
if 'svg_filename' not in st.session_state:
    st.session_state['svg_filename'] = "output.svg"
if 'zip_filename' not in st.session_state:
    st.session_state['zip_filename'] = "output.zip"
if 'error_message' not in st.session_state:
    st.session_state['error_message'] = None


# --- Input File Upload ---
uploaded_file = st.file_uploader("1. Upload PNG Image", type=['png'])

# --- Sidebar for Parameters ---
st.sidebar.header("⚙️ Conversion Parameters")

n_colors = st.sidebar.slider(
    "Number of Colors (Clusters)", min_value=2, max_value=64, value=DEFAULT_N_COLORS, step=1,
    help="Controls the level of color detail captured. More colors = more detail, but more complex SVG."
)

mode = st.sidebar.radio(
    "Output Mode", ('fill', 'stroke'), index=0 if DEFAULT_MODE == 'fill' else 1,
    help=" 'fill' creates solid color shapes. 'stroke' creates outlines."
)

# Conditionally show stroke width slider only if mode is 'stroke'
stroke_width = DEFAULT_STROKE_WIDTH
if mode == 'stroke':
    stroke_width = st.sidebar.slider(
        "Stroke Width", min_value=0.1, max_value=10.0, value=DEFAULT_STROKE_WIDTH, step=0.1,
        help="Line thickness for 'stroke' mode."
    )

min_area = st.sidebar.slider(
    "Minimum Area", min_value=0, max_value=1000, value=DEFAULT_MIN_AREA, step=5,
    help="Ignore small color regions (noise reduction). Area is in pixels."
)

blur_radius = st.sidebar.slider(
    "Gaussian Blur Radius", min_value=0.0, max_value=5.0, value=DEFAULT_BLUR_RADIUS, step=0.1,
    help="Apply blur before processing to smooth edges (0 = no blur)."
)

# --- Action Button ---
st.sidebar.markdown("---") # Separator
run_button = st.sidebar.button("🚀 Convert to SVG", disabled=(uploaded_file is None))
st.sidebar.markdown("---") # Separator

# --- Processing Logic ---
if run_button and uploaded_file is not None:
    # Reset previous results/errors
    st.session_state['svg_content'] = None
    st.session_state['zip_content'] = None
    st.session_state['error_message'] = None

    # Generate default filenames based on uploaded file
    base_filename = os.path.splitext(uploaded_file.name)[0]
    st.session_state['svg_filename'] = f"{base_filename}_vectorized.svg"
    st.session_state['zip_filename'] = f"{base_filename}_vectorized.zip"

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(uploaded_file, caption=f"Input: {uploaded_file.name}", use_column_width=True)

    with col2:
        st.subheader("Processing...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.info("Loading image...")
            try:
                image = Image.open(uploaded_file).convert("RGBA")
            except UnidentifiedImageError:
                raise ValueError("Uploaded file is not a valid PNG image.")
            progress_bar.progress(10)

            status_text.info(f"Applying blur (radius: {blur_radius})...")
            image_processed = apply_optional_blur(image, blur_radius)
            progress_bar.progress(20)

            status_text.info(f"Segmenting into {n_colors} colors...")
            label_map, centers, shape = segment_image(image_processed, n_colors)
            progress_bar.progress(50)

            status_text.info(f"Generating SVG paths (mode: {mode}, min_area: {min_area})...")
            svg_data = create_svg_content(
                label_map, centers, shape, mode, min_area, stroke_width
            )
            st.session_state['svg_content'] = svg_data # Store SVG data
            progress_bar.progress(90)

            # Optional: Generate ZIP content if needed later for download
            # We generate it here to ensure it's available if checkbox is ticked later
            try:
                st.session_state['zip_content'] = create_zip_content(
                    svg_data, st.session_state['svg_filename']
                    )
            except Exception as zip_e:
                 st.warning(f"Could not create ZIP file: {zip_e}")
                 st.session_state['zip_content'] = None # Ensure it's None if zipping fails

            progress_bar.progress(100)
            status_text.success("✅ Conversion Successful!")

            # Display SVG preview (Streamlit has limited native SVG rendering, show as text/code)
            st.subheader("Generated SVG Preview")
            st.markdown("_(Browser rendering might differ from vector software)_")
            # Render SVG using st.image which handles SVG data URIs
            st.image(st.session_state['svg_content'])


        except Exception as e:
            st.session_state['error_message'] = f"❌ Conversion Failed: {e}"
            status_text.error(st.session_state['error_message'])
            # Ensure progress bar shows completion even on error
            progress_bar.progress(100)


# --- Display Download Buttons ---
if st.session_state['svg_content']:
    st.sidebar.header("⬇️ Download Results")
    st.sidebar.download_button(
        label="Download SVG",
        data=st.session_state['svg_content'],
        file_name=st.session_state['svg_filename'],
        mime="image/svg+xml",
    )

    # Optional ZIP download
    create_zip = st.sidebar.checkbox("Create ZIP file", value=False)
    if create_zip and st.session_state['zip_content']:
        st.sidebar.download_button(
            label="Download ZIP",
            data=st.session_state['zip_content'],
            file_name=st.session_state['zip_filename'],
            mime="application/zip",
        )
    elif create_zip and not st.session_state['zip_content']:
         st.sidebar.warning("ZIP creation failed earlier.")

elif st.session_state['error_message']:
    # Optionally show persistent error message if conversion failed
    # st.error(st.session_state['error_message']) # Already shown during processing
    pass

st.sidebar.markdown("---")
st.sidebar.info("Tip: Experiment with the parameters in the sidebar and re-run the conversion.")