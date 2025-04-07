#!/usr/bin/env python3
"""
Streamlit frontend for converting raster images (PNG, JPG, GIF)
to vector SVGs using either color segmentation or edge detection.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError, ImageSequence
# Import MiniBatchKMeans
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.exceptions import ConvergenceWarning
import svgwrite
import zipfile
import warnings
from typing import Tuple, List, Optional, Union, Dict
import io
import time
import streamlit as st

# --- Constants ---
DEFAULT_N_COLORS = 16
DEFAULT_MIN_AREA = 20
DEFAULT_STROKE_WIDTH = 1.0
DEFAULT_MODE = 'fill'
DEFAULT_BLUR_RADIUS = 0.0
DEFAULT_MAX_DIMENSION = 1024 # For scaling
DEFAULT_CANNY_T1 = 100
DEFAULT_CANNY_T2 = 200
DEFAULT_EDGE_COLOR = 'black'
DEFAULT_EDGE_MIN_AREA = 10 # Edges can be smaller than regions

KMEANS_N_INIT = 10
RANDOM_STATE = 42
MINIBATCH_SIZE = 1024 * 3 # Adjust based on typical image size/memory

# --- Backend Logic Functions ---
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn') # Suppress MiniBatchKMeans UserWarning

# --- Contour/SVG Helpers ---

def contour_to_svg_path(contour: np.ndarray) -> Optional[str]:
    """Converts an OpenCV contour to SVG path data string, returns None if invalid."""
    if not isinstance(contour, np.ndarray) or contour.ndim != 3 or contour.shape[1] != 1 or contour.shape[2] != 2:
        st.warning(f"Unexpected contour format: shape={contour.shape}. Skipping.")
        return None # Return None for invalid format
    if len(contour) < 2:
        return None # Return None for too short contour

    path_data = f"M {contour[0, 0, 0]},{contour[0, 0, 1]}"
    if len(contour) > 1:
        path_data += " L " + " ".join(f"{pt[0][0]},{pt[0][1]}" for pt in contour[1:])
    path_data += " Z"
    return path_data

def _generate_svg_path_attributes(
    contour: np.ndarray,
    mode: str,
    min_area: int,
    stroke_width: float,
    color: Union[Tuple[int, int, int], str] = 'black' # Can be RGB tuple or color name/hex
) -> Optional[Dict[str, Union[str, float]]]:
    """
    Generates SVG path attributes dictionary for a contour based on mode.
    Returns None if contour area is too small or path is invalid.
    """
    area = cv2.contourArea(contour)
    if area < min_area:
        return None

    path_data = contour_to_svg_path(contour)
    if path_data is None: # Check for None return from contour_to_svg_path
        return None

    # Determine fill/stroke based on mode and color type
    is_rgb_color = isinstance(color, tuple)
    svg_color_val = svgwrite.rgb(*color) if is_rgb_color else color

    path_attrs = {'d': path_data}
    if mode == 'fill':
        if not is_rgb_color:
             st.warning("Fill mode typically expects RGB colors from segmentation.")
             # Fallback: use the provided color string anyway, might be invalid SVG
             path_attrs['fill'] = color
        else:
            path_attrs['fill'] = svg_color_val
        path_attrs['stroke'] = 'none'
    elif mode == 'stroke':
        path_attrs['fill'] = 'none'
        path_attrs['stroke'] = svg_color_val
        path_attrs['stroke-width'] = stroke_width
        path_attrs['stroke-linecap'] = 'round'
        path_attrs['stroke-linejoin'] = 'round'
    else:
        st.error(f"Invalid mode '{mode}' encountered in _generate_svg_path_attributes")
        return None

    return path_attrs

# --- Image Processing ---

def apply_optional_blur(image: Image.Image, blur_radius: float) -> Image.Image:
    """Applies Gaussian blur if radius > 0."""
    if blur_radius > 0:
        return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return image

def resize_image(image: Image.Image, max_dimension: int) -> Image.Image:
    """Resizes image if its largest dimension exceeds max_dimension, maintains aspect ratio."""
    max_dim = max(image.width, image.height)
    if max_dim > max_dimension:
        scale = max_dimension / max_dim
        new_size = (int(image.width * scale), int(image.height * scale))
        st.info(f"Downscaling image from {image.size} to {new_size}...")
        # Use ANTIALIAS (Pillow >= 9) or LANCZOS (older Pillow)
        resample_method = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
        return image.resize(new_size, resample_method=resample_method)
    return image

# --- Vectorization Methods ---

def segment_image_kmeans(image: Image.Image, n_colors: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """Segments image using MiniBatchKMeans."""
    st.info(f"Segmenting image into {n_colors} colors using MiniBatchKMeans...")
    start_time = time.time()
    try:
        # Ensure image is RGB for clustering
        img_rgb = image.convert("RGB")
        img_np = np.array(img_rgb)
        original_shape = img_np.shape
        pixels = img_np.reshape((-1, 3)).astype(np.float32) # Kmeans prefers float

        # Use MiniBatchKMeans for potentially faster clustering
        kmeans = MiniBatchKMeans(
            n_clusters=n_colors,
            n_init=KMEANS_N_INIT,
            random_state=RANDOM_STATE,
            batch_size=MINIBATCH_SIZE,
            max_iter=100 # Usually converges faster than standard KMeans
        )
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_
        centers_int = centers.round().astype(np.uint8)
        label_map = labels.reshape(original_shape[:2])

        duration = time.time() - start_time
        st.info(f"Segmentation complete ({duration:.2f}s).")
        return label_map, centers_int, original_shape

    except Exception as e:
        st.error(f"Error during KMeans segmentation: {e}")
        raise

def detect_edges_canny(image: Image.Image, threshold1: int, threshold2: int) -> np.ndarray:
    """Detects edges using Canny algorithm."""
    st.info(f"Detecting edges using Canny (T1={threshold1}, T2={threshold2})...")
    start_time = time.time()
    try:
        # Convert to grayscale for Canny
        img_gray = np.array(image.convert('L')) # 'L' mode is grayscale
        edges = cv2.Canny(img_gray, threshold1, threshold2)
        duration = time.time() - start_time
        st.info(f"Edge detection complete ({duration:.2f}s).")
        return edges
    except Exception as e:
        st.error(f"Error during Canny edge detection: {e}")
        raise

# --- SVG Creation ---

def create_svg_from_clusters(
    label_map: np.ndarray,
    centers: np.ndarray,
    shape: Tuple[int, int, int],
    mode: str,
    min_area: int,
    stroke_width: float
) -> str:
    """Generates SVG content string from clustered image regions."""
    st.info(f"Generating SVG from color clusters (mode: {mode}, min_area: {min_area})...")
    height, width = shape[:2]
    svg_io = io.StringIO()
    dwg = svgwrite.Drawing(profile='tiny', size=(f"{width}px", f"{height}px"))
    num_paths_added = 0

    for i, center_color in enumerate(centers):
        mask = np.zeros(label_map.shape, dtype=np.uint8)
        mask[label_map == i] = 255
        try:
            # Find contours - RETR_CCOMP finds external and holes
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if not contours or hierarchy is None: continue

            # Iterate through top-level contours (no parent)
            contour_index = 0
            while contour_index >= 0:
                contour = contours[contour_index]
                # Generate attributes (includes area check and path conversion)
                path_attrs = _generate_svg_path_attributes(
                    contour, mode, min_area, stroke_width, tuple(center_color)
                )
                if path_attrs is not None:
                    dwg.add(dwg.path(**path_attrs))
                    num_paths_added += 1

                # Move to next contour at the same level
                contour_index = hierarchy[0, contour_index, 0]

        except cv2.error as e:
            st.warning(f"OpenCV error processing cluster {i}: {e}")
            continue
        except Exception as e:
            st.error(f"Error during contour processing for cluster {i}: {e}")
            raise # Re-raise critical errors

    svg_string = dwg.tostring()
    st.info(f"Generated SVG with {num_paths_added} paths from clusters.")
    return svg_string

def create_svg_from_edges(
    edge_map: np.ndarray,
    shape: Tuple[int, int], # Shape is now (height, width) from grayscale
    min_area: int,
    stroke_width: float,
    edge_color: str
) -> str:
    """Generates SVG content string from edge map contours."""
    st.info(f"Generating SVG from edges (min_area: {min_area}, color: {edge_color})...")
    height, width = shape[:2]
    svg_io = io.StringIO()
    dwg = svgwrite.Drawing(profile='tiny', size=(f"{width}px", f"{height}px"))
    num_paths_added = 0

    try:
        # Find contours on the edge map - RETR_EXTERNAL is suitable for outlines
        contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            st.warning("No contours found on the edge map.")
            return dwg.tostring() # Return empty drawing

        for contour in contours:
            # Generate attributes for stroke mode
            path_attrs = _generate_svg_path_attributes(
                contour, 'stroke', min_area, stroke_width, edge_color
            )
            if path_attrs is not None:
                dwg.add(dwg.path(**path_attrs))
                num_paths_added += 1

    except cv2.error as e:
        st.warning(f"OpenCV error processing edges: {e}")
    except Exception as e:
        st.error(f"Error during edge contour processing: {e}")
        raise # Re-raise critical errors

    svg_string = dwg.tostring()
    st.info(f"Generated SVG with {num_paths_added} paths from edges.")
    return svg_string

# --- ZIP Helper ---

def create_zip_content(svg_content: str, svg_filename: str) -> bytes:
    """Creates a ZIP archive in memory containing the SVG content."""
    zip_buffer = io.BytesIO()
    try:
        with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr(svg_filename, svg_content)
        # Return the ZIP content as bytes
        return zip_buffer.getvalue()
    except Exception as e:
        st.error(f"Error creating ZIP file in memory: {e}")
        raise


# --- Streamlit UI ---

st.set_page_config(page_title="Image to SVG Converter", layout="wide")

st.title("🖼️ Image to Vector SVG Converter")
st.markdown("Upload a raster image (PNG, JPG, GIF) to convert it into an SVG file "
            "using either **Color Segmentation** or **Edge Detection**.")

# --- Initialize Session State ---
if 'svg_content' not in st.session_state: st.session_state['svg_content'] = None
if 'zip_content' not in st.session_state: st.session_state['zip_content'] = None
if 'svg_filename' not in st.session_state: st.session_state['svg_filename'] = "output.svg"
if 'zip_filename' not in st.session_state: st.session_state['zip_filename'] = "output.zip"
if 'error_message' not in st.session_state: st.session_state['error_message'] = None


# --- Input File Upload ---
uploaded_file = st.file_uploader(
    "1. Upload Image",
    type=['png', 'jpg', 'jpeg', 'gif'], # Added more types
    help="Upload PNG, JPG, or GIF files."
)

# --- Sidebar for Parameters ---
st.sidebar.header("⚙️ Conversion Parameters")

# --- General Parameters ---
st.sidebar.subheader("General Settings")
vectorization_method = st.sidebar.radio(
    "Vectorization Method",
    ('Color Segmentation', 'Edge Detection'),
    index=0, # Default to Color Segmentation
    key='vector_method', # Unique key
    help="**Color Segmentation:** Creates regions based on dominant colors. \n"
         "**Edge Detection:** Creates outlines based on detected edges (ignores color)."
)

blur_radius = st.sidebar.slider(
    "Gaussian Blur Radius", min_value=0.0, max_value=10.0, value=DEFAULT_BLUR_RADIUS, step=0.1,
    help="Apply blur *before* vectorization to smooth edges/noise (0 = no blur)."
)

enable_downscaling = st.sidebar.checkbox(
    "Downscale Large Images", value=True,
    help=f"Reduce image size if larger than max dimension to improve performance."
)
max_dimension = st.sidebar.slider(
    "Max Dimension (px)", min_value=256, max_value=4096, value=DEFAULT_MAX_DIMENSION, step=128,
    disabled=not enable_downscaling,
    help="Largest side length allowed if downscaling is enabled."
)

st.sidebar.markdown("---")

# --- Method-Specific Parameters ---

# Parameters for Color Segmentation
if vectorization_method == 'Color Segmentation':
    st.sidebar.subheader("Color Segmentation Settings")
    n_colors = st.sidebar.slider(
        "Number of Colors", min_value=2, max_value=64, value=DEFAULT_N_COLORS, step=1,
        help="Number of dominant colors (clusters) to detect."
    )
    mode = st.sidebar.radio(
        "Output Mode", ('fill', 'stroke'), index=0 if DEFAULT_MODE == 'fill' else 1,
        help=" 'fill': solid color shapes. 'stroke': color outlines."
    )
    min_area_color = st.sidebar.slider(
        "Min Region Area", min_value=0, max_value=1000, value=DEFAULT_MIN_AREA, step=5, key='min_area_color',
        help="Ignore small color regions (pixels)."
    )
    # Conditionally show stroke width only if mode is 'stroke'
    stroke_width_color = DEFAULT_STROKE_WIDTH
    if mode == 'stroke':
        stroke_width_color = st.sidebar.slider(
            "Stroke Width", min_value=0.1, max_value=10.0, value=DEFAULT_STROKE_WIDTH, step=0.1, key='stroke_width_color',
            help="Line thickness for 'stroke' mode."
        )

# Parameters for Edge Detection
elif vectorization_method == 'Edge Detection':
    st.sidebar.subheader("Edge Detection Settings")
    canny_threshold1 = st.sidebar.slider(
        "Canny Threshold 1", min_value=0, max_value=500, value=DEFAULT_CANNY_T1, step=5,
        help="Lower threshold for Canny edge detection. Affects edge sensitivity."
    )
    canny_threshold2 = st.sidebar.slider(
        "Canny Threshold 2", min_value=0, max_value=500, value=DEFAULT_CANNY_T2, step=5,
        help="Higher threshold for Canny edge detection. Affects edge linking."
    )
    min_area_edge = st.sidebar.slider(
        "Min Edge Length Area", min_value=0, max_value=500, value=DEFAULT_EDGE_MIN_AREA, step=1, key='min_area_edge',
        help="Ignore very short edge contours (approx. pixel area)."
    )
    stroke_width_edge = st.sidebar.slider(
        "Stroke Width", min_value=0.1, max_value=10.0, value=DEFAULT_STROKE_WIDTH, step=0.1, key='stroke_width_edge',
        help="Line thickness for edge strokes."
    )
    edge_color = st.sidebar.color_picker(
        "Edge Stroke Color", value=f"#{DEFAULT_EDGE_COLOR}" if DEFAULT_EDGE_COLOR=='black' else DEFAULT_EDGE_COLOR , # Use hex for default black
        help="Color of the detected edge lines in the SVG."
    )


# --- Action Button ---
st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Convert to SVG", disabled=(uploaded_file is None))
st.sidebar.markdown("---")

# --- Processing Logic ---
if run_button and uploaded_file is not None:
    # Reset previous results/errors
    st.session_state['svg_content'] = None
    st.session_state['zip_content'] = None
    st.session_state['error_message'] = None

    base_filename = os.path.splitext(uploaded_file.name)[0]
    method_suffix = "colors" if vectorization_method == 'Color Segmentation' else "edges"
    st.session_state['svg_filename'] = f"{base_filename}_{method_suffix}.svg"
    st.session_state['zip_filename'] = f"{base_filename}_{method_suffix}.zip"

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image Preview")
        try:
            # Load image and display immediately
            # Handle GIF: use the first frame
            preview_image = Image.open(uploaded_file)
            if preview_image.format == "GIF":
                 # Seek to the first frame for display/processing
                preview_image.seek(0)
            # Display using Streamlit (handles RGBA conversion for display)
            st.image(preview_image, caption=f"Input: {uploaded_file.name}", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading image preview: {e}")
            st.stop() # Stop execution if preview fails


    with col2:
        st.subheader("Processing...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.info("Loading and Preparing Image...")
            # Load again for processing, ensuring RGBA for consistency downstream
            # Reset file pointer for Pillow
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
            # Handle GIF: use the first frame
            if image.format == "GIF":
                image.seek(0)
                # Important: Convert GIF frame to RGBA (might be indexed)
                image = image.convert("RGBA")

            # Ensure RGBA base format for consistency before processing
            if image.mode != 'RGBA':
                 image = image.convert("RGBA")

            progress_bar.progress(10)

            # 1. Optional Downscaling
            image_to_process = image # Start with original loaded image
            if enable_downscaling:
                 status_text.info("Applying Downscaling (if necessary)...")
                 image_to_process = resize_image(image_to_process, max_dimension)
            progress_bar.progress(20)

            # 2. Optional Blur
            status_text.info(f"Applying Blur (radius: {blur_radius})...")
            image_processed = apply_optional_blur(image_to_process, blur_radius)
            progress_bar.progress(30)

            # 3. Vectorization (Method Dependent)
            if vectorization_method == 'Color Segmentation':
                 # 3a. Segment Image using KMeans
                 label_map, centers, shape = segment_image_kmeans(image_processed, n_colors)
                 progress_bar.progress(60)

                 # 4a. Create SVG from Clusters
                 svg_data = create_svg_from_clusters(
                     label_map, centers, shape, mode, min_area_color, stroke_width_color
                 )
                 progress_bar.progress(90)

            elif vectorization_method == 'Edge Detection':
                 # 3b. Detect Edges using Canny
                 # Pass image_processed which includes potential blur
                 edge_map = detect_edges_canny(image_processed, canny_threshold1, canny_threshold2)
                 progress_bar.progress(60)

                 # 4b. Create SVG from Edges
                 svg_data = create_svg_from_edges(
                     edge_map, edge_map.shape, min_area_edge, stroke_width_edge, edge_color
                 )
                 progress_bar.progress(90)

            else:
                 # Should not happen due to radio button choices
                 raise ValueError(f"Invalid vectorization method: {vectorization_method}")


            st.session_state['svg_content'] = svg_data # Store SVG data

            # 5. Optional ZIP creation (do it here for download button readiness)
            try:
                st.session_state['zip_content'] = create_zip_content(
                    svg_data, st.session_state['svg_filename']
                    )
            except Exception as zip_e:
                 st.warning(f"Could not create ZIP file: {zip_e}")
                 st.session_state['zip_content'] = None

            progress_bar.progress(100)
            status_text.success("✅ Conversion Successful!")

            # Display SVG preview
            st.subheader("Generated SVG Preview")
            st.markdown("_(Browser rendering might differ from vector software. Best viewed by downloading.)_")
            st.image(st.session_state['svg_content'])


        except Exception as e:
            st.session_state['error_message'] = f"❌ Conversion Failed: {e}"
            import traceback
            st.error(f"{st.session_state['error_message']}\n```\n{traceback.format_exc()}\n```") # Show traceback for debugging
            # Ensure progress bar shows completion even on error
            progress_bar.progress(100)


# --- Display Download Buttons ---
if st.session_state.get('svg_content'): # Use .get for safer access
    st.sidebar.header("⬇️ Download Results")
    st.sidebar.download_button(
        label="Download SVG",
        data=st.session_state['svg_content'],
        file_name=st.session_state['svg_filename'],
        mime="image/svg+xml",
    )

    # Optional ZIP download
    create_zip = st.sidebar.checkbox("Create ZIP file", value=False)
    if create_zip:
        if st.session_state.get('zip_content'):
            st.sidebar.download_button(
                label="Download ZIP",
                data=st.session_state['zip_content'],
                file_name=st.session_state['zip_filename'],
                mime="application/zip",
            )
        else:
             st.sidebar.warning("ZIP creation failed or not available.")


st.sidebar.markdown("---")
st.sidebar.info("Tip: Experiment with parameters and methods!")

# --- Display Error if process failed ---
if st.session_state.get('error_message'):
    # Error already displayed in the main area during processing
    pass