import streamlit as st
import cv2
import numpy as np
import io

# Helper to convert hex color string to BGR tuple for OpenCV
def hex_to_bgr(hex_color_code):
    """Converts a hex color string (e.g., '#FF0000') to an OpenCV BGR tuple."""
    hex_color_code = hex_color_code.lstrip('#')
    # Convert hex to RGB, then reverse to BGR for OpenCV
    return tuple(int(hex_color_code[i:i+2], 16) for i in (4, 2, 0))

# Load the Haar Cascade classifier for face detection
# Ensure 'haarcascade_frontalface_default.xml' is accessible.
# cv2.data.haarcascades provides the path to OpenCV's installed cascade files.
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        st.error("Error loading Haar cascade classifier. Make sure 'haarcascade_frontalface_default.xml' is present.")
        st.stop() # Stop the app if the cascade fails to load
except Exception as e:
    st.error(f"Failed to load cascade classifier: {e}")
    st.stop()

def main():
    st.title("💡 Interactive Face Detection App")

    # --- 1. Add instructions to the Streamlit app interface ---
    st.markdown("""
    Welcome to the **Interactive Face Detection App**!
    
    Upload an image below to automatically detect faces using the Viola-Jones algorithm.
    
    You can customize the detection sensitivity and the color of the bounding boxes using the controls in the **sidebar** on the left.
    Once faces are detected, you'll have the option to download the image with the bounding boxes.
    """)
    st.write("---") # Horizontal line for separation
    # -----------------------------------------------------------------

    uploaded_file = st.file_uploader("📸 Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image bytes and decode into an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1) # 1 means load as color image

        st.subheader("🖼️ Original Image")
        st.image(img, channels="BGR", use_column_width=True)
        st.write("---")

        # --- 2. Allow user to choose the color of the rectangles ---
        st.sidebar.subheader("🎨 Customize Detection")
        selected_color_hex = st.sidebar.color_picker("Choose rectangle color", "#FF0000") # Default to Red
        rectangle_color_bgr = hex_to_bgr(selected_color_hex)
        # ----------------------------------------------------------

        # --- 3. Adjust minNeighbors parameter ---
        st.sidebar.markdown("**Detection Sensitivity**")
        min_neighbors = st.sidebar.slider(
            "Minimum Neighbors (higher = stricter detection, fewer false positives)",
            min_value=1, max_value=15, value=5, step=1,
            help="Higher values require more 'neighbors' (overlapping detections) for a face to be confirmed, reducing false positives but potentially missing some faces."
        )
        # ----------------------------------------

        # --- 4. Adjust scaleFactor parameter ---
        scale_factor = st.sidebar.slider(
            "Scale Factor (lower = more thorough search, slower, detects smaller faces)",
            min_value=1.01, max_value=1.5, value=1.1, step=0.01, format="%.2f",
            help="This parameter specifies how much the image size is reduced at each image scale. A lower value (e.g., 1.05) means a finer scan and might detect smaller or more distant faces, but takes longer."
        )
        st.write("---")
        # ---------------------------------------

        # Convert to grayscale for face detection (Viola-Jones works best on grayscale)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Perform face detection using the Haar Cascade
        # Uses the user-adjusted parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30) # Minimum possible object size to be detected. Objects smaller than that are ignored.
        )

        # Draw rectangles around the detected faces on a copy of the original image
        img_with_faces = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(img_with_faces, (x, y), (x + w, y + h), rectangle_color_bgr, 2)
            # Add text label (optional)
            # cv2.putText(img_with_faces, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rectangle_color_bgr, 2)

        st.subheader("👤 Detected Faces")
        if len(faces) > 0:
            st.write(f"Found {len(faces)} face(s) in the image.")
            st.image(img_with_faces, channels="BGR", use_column_width=True)
        else:
            st.info("No faces detected with the current settings. Try adjusting the parameters in the sidebar!")
        st.write("---")

        # --- 5. Add a feature to save the images with detected faces ---
        # Convert the image with faces back to bytes (PNG format) for download
        is_success, buffer = cv2.imencode(".png", img_with_faces)
        if is_success:
            st.download_button(
                label="⬇️ Download Image with Faces",
                data=buffer.tobytes(),
                file_name="detected_faces.png",
                mime="image/png",
                help="Click to download the image with the detected faces highlighted."
            )
        # -----------------------------------------------------------------

if __name__ == '__main__':
    main()