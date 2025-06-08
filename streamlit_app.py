import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="🏷️ Image Classification Task",
    page_icon="🏷️",
    layout="wide"
)

# Set paths relative to the script location (following the pattern from app.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")  # Adjust this path as needed
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "annotations")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the 10 labels
LABELS = [
    "(a) Code",
    "(b) Run Time Error", 
    "(c) Menus and Preferences",
    "(d) Dialog Box",
    "(e) Steps and Processes",
    "(f) Program Input",
    "(g) Desired Output",
    "(h) Program Output",
    "(i) CPU/GPU Performance",
    "(j) Algorithm/Concept Description"
]

# Special option for when no labels apply
NONE_OPTION = "None of the above labels can apply"

# Initialize session state
if 'current_image_index' not in st.session_state:
    st.session_state.current_image_index = 0
if 'annotations' not in st.session_state:
    st.session_state.annotations = {}
if 'images_data' not in st.session_state:
    st.session_state.images_data = None

class DataLoader:
    """Data loader class following the pattern from app.py"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def load_csv(self, filename):
        """Load CSV file from data directory"""
        file_path = os.path.join(self.data_dir, filename)
        try:
            if not os.path.exists(file_path):
                st.error(f"❌ File not found: {file_path}")
                st.info(f"💡 Please make sure '{filename}' exists in the data directory: {self.data_dir}")
                return None
            
            df = pd.read_csv(file_path)
            st.success(f"✅ Successfully loaded {len(df)} rows from {filename}")
            return df
            
        except Exception as e:
            st.error(f"❌ Error loading {filename}: {str(e)}")
            return None
    
    def get_image_path(self, image_filename):
        """Get full path to image file"""
        if pd.isna(image_filename) or not image_filename:
            return None
        
        # Try different possible locations for images
        possible_paths = [
            os.path.join(self.data_dir, str(image_filename)),
            os.path.join(self.data_dir, "images", str(image_filename)),
            os.path.join(SCRIPT_DIR, "images", str(image_filename)),
            os.path.join(SCRIPT_DIR, str(image_filename))
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None

@st.cache_data
def load_images_data():
    """Load image data using the improved DataLoader"""
    loader = DataLoader(DATA_DIR)
    return loader.load_csv('post_img.csv')

def display_image_with_proper_path(current_row, loader):
    """Display image with proper path handling"""
    
    # Debug info
    with st.expander("🔍 Debug: Available Columns"):
        st.write("**All columns in CSV:**")
        st.write(list(current_row.index))
        
        # Show first row of data
        st.write("**Current row data:**")
        for col in current_row.index:
            st.write(f"• **{col}**: {current_row[col]}")
    
    # Look for image columns more broadly
    image_columns = []
    for col in current_row.index:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['image', 'img', 'url', 'path', 'screenshot', 'pic', 'photo', 'src', 'file']):
            image_columns.append(col)
    
    if image_columns:
        st.write(f"**Detected potential image columns:** {image_columns}")
        
        for col in image_columns:
            image_identifier = current_row[col]
            
            if pd.isna(image_identifier):
                st.warning(f"⚠️ {col} is empty for this row")
                continue
                
            # Handle different types of image identifiers
            image_identifier_str = str(image_identifier).strip()
            
            if image_identifier_str.startswith(('http://', 'https://')):
                # It's a URL
                try:
                    st.image(image_identifier_str, 
                            caption=f"Image from {col}", 
                            use_column_width=True)
                    return True
                except Exception as e:
                    st.error(f"❌ Failed to load image from URL: {e}")
                    st.code(f"URL: {image_identifier_str}")
            else:
                # It's likely a filename or path
                image_path = loader.get_image_path(image_identifier_str)
                
                if image_path:
                    try:
                        st.image(image_path, 
                                caption=f"Image: {os.path.basename(image_path)}", 
                                use_column_width=True)
                        return True
                    except Exception as e:
                        st.error(f"❌ Failed to load image: {e}")
                        st.code(f"Path: {image_path}")
                else:
                    st.warning(f"⚠️ Image file not found: {image_identifier_str}")
                    st.info("💡 Searched in:")
                    search_paths = [
                        os.path.join(DATA_DIR, image_identifier_str),
                        os.path.join(DATA_DIR, "images", image_identifier_str),
                        os.path.join(SCRIPT_DIR, "images", image_identifier_str),
                        os.path.join(SCRIPT_DIR, image_identifier_str)
                    ]
                    for path in search_paths:
                        st.code(path)
    else:
        st.info("💡 No image columns detected. Looking for columns containing: image, img, url, path, screenshot, pic, photo, src, file")
    
    return False

def save_annotations_to_json():
    """Save annotations to JSON format for export"""
    annotations_export = []
    
    for img_index, labels in st.session_state.annotations.items():
        if st.session_state.images_data is not None:
            # Get image info from the dataframe
            img_row = st.session_state.images_data.iloc[int(img_index)]
            
            # Only keep specific columns and convert to native Python types
            keep_columns = ['post_id', 'title', 'link']
            img_info = {}
            for col in keep_columns:
                if col in img_row.index:
                    value = img_row[col]
                    # Convert pandas/numpy types to native Python types
                    if pd.isna(value):
                        img_info[col] = None
                    elif hasattr(value, 'item'):  # numpy/pandas scalar
                        img_info[col] = value.item()
                    else:
                        img_info[col] = value
            
            annotation_entry = {
                'image_index': int(img_index),
                'image_info': img_info,
                'selected_labels': labels
            }
            annotations_export.append(annotation_entry)
    
    return json.dumps(annotations_export, indent=2, ensure_ascii=False)

def main():
    st.title("🏷️ Image Annotation Tool")
    st.markdown("---")
    
    # Display current directories for debugging
    with st.expander("📁 File System Info"):
        st.write(f"**Script directory:** {SCRIPT_DIR}")
        st.write(f"**Data directory:** {DATA_DIR}")
        st.write(f"**Output directory:** {OUTPUT_DIR}")
        
        # List files in data directory
        if os.path.exists(DATA_DIR):
            st.write(f"**Files in data directory:**")
            for file in os.listdir(DATA_DIR):
                st.write(f"• {file}")
        else:
            st.warning(f"⚠️ Data directory does not exist: {DATA_DIR}")
    
    # Load image data
    if st.session_state.images_data is None:
        st.session_state.images_data = load_images_data()
    
    if st.session_state.images_data is None:
        st.stop()
    
    total_images = len(st.session_state.images_data)
    
    if total_images == 0:
        st.error("❌ No images found in the CSV file.")
        st.stop()
    
    # Progress indicator
    progress = (st.session_state.current_image_index + 1) / total_images
    st.progress(progress)
    st.write(f"**Image {st.session_state.current_image_index + 1} of {total_images}**")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display current image
        current_row = st.session_state.images_data.iloc[st.session_state.current_image_index]
        loader = DataLoader(DATA_DIR)
        
        st.subheader("Current Image")
        
        # Display image with proper path handling
        image_displayed = display_image_with_proper_path(current_row, loader)
        
        if not image_displayed:
            st.warning("⚠️ No image could be displayed for this row")
        
        # Display other information
        st.write("**Additional Information:**")
        display_columns = ['post_id', 'title', 'link']
        for column in display_columns:
            if column in current_row.index:
                st.write(f"• **{column}**: {current_row[column]}")
    
    with col2:
        st.subheader("🏷️ Select Labels")
        st.write("Choose 1-3 labels OR select 'None' if no labels apply:")
        
        # Get current annotations for this image
        current_annotations = st.session_state.annotations.get(
            str(st.session_state.current_image_index), []
        )
        
        # Check if "None" option is currently selected
        none_selected = NONE_OPTION in current_annotations
        
        # Create checkboxes for each label
        selected_labels = []
        for i, label in enumerate(LABELS):
            checkbox_key = f"label_{label}_{st.session_state.current_image_index}"
            disabled = none_selected
            if st.checkbox(label, value=(label in current_annotations and not none_selected), key=checkbox_key, disabled=disabled):
                selected_labels.append(label)
        
        # Add separator
        st.markdown("---")
        
        # "None of the above" option
        none_checkbox_key = f"none_option_{st.session_state.current_image_index}"
        none_disabled = len(selected_labels) > 0
        none_checked = st.checkbox(NONE_OPTION, value=none_selected, key=none_checkbox_key, disabled=none_disabled)
        
        # Determine final selection
        if none_checked:
            final_selection = [NONE_OPTION]
            selection_valid = True
        elif selected_labels:
            final_selection = selected_labels
            selection_valid = len(selected_labels) <= 3
        else:
            final_selection = []
            selection_valid = False
        
        # Validate selection
        if len(selected_labels) > 3:
            st.error("❌ You can select maximum 3 labels!")
        elif not selection_valid:
            st.warning("⚠️ Please select at least one option (labels or 'None') to continue!")
        
        # Save current selections
        st.session_state.annotations[str(st.session_state.current_image_index)] = final_selection
        
        # Display current selection
        if final_selection:
            if NONE_OPTION in final_selection:
                st.info("ℹ️ None of the labels apply to this image")
            else:
                st.success(f"✅ Selected {len(final_selection)} label(s):")
                for label in final_selection:
                    st.write(f"• {label}")
        else:
            st.info("ℹ️ No selection made for this image")
        
        st.markdown("---")
        
        # Navigation
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            prev_disabled = (st.session_state.current_image_index == 0) or not selection_valid
            if st.button("⬅️ Previous", disabled=prev_disabled):
                st.session_state.current_image_index -= 1
                st.rerun()
        
        with col_nav2:
            next_disabled = (st.session_state.current_image_index >= total_images - 1) or not selection_valid
            if st.button("➡️ Next", disabled=next_disabled):
                st.session_state.current_image_index += 1
                st.rerun()
        
        # Jump to specific image
        target_image = st.number_input(
            "Go to image:", 
            min_value=1, 
            max_value=total_images, 
            value=st.session_state.current_image_index + 1,
            key=f"nav_input_{st.session_state.current_image_index}"
        )
        go_disabled = not selection_valid
        if st.button("🎯 Go", disabled=go_disabled):
            st.session_state.current_image_index = target_image - 1
            st.rerun()
        
        st.markdown("---")
        
        # Export annotations
        st.subheader("💾 Export Annotations")
        if st.session_state.annotations:
            annotations_json = save_annotations_to_json()
            filename = f"image_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            st.download_button(
                label="💾 Download Annotations",
                data=annotations_json,
                file_name=filename,
                mime="application/json",
                help="Download your annotations as a JSON file"
            )
        else:
            st.button("💾 Download Annotations", disabled=True, help="No annotations to export yet")
    
    # Rest of your code remains the same...
    st.markdown("---")
    st.subheader("📊 Annotation Summary")
    
    annotated_count = len(st.session_state.annotations)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Images", total_images)
    with col2:
        st.metric("Annotated Images", annotated_count)
    with col3:
        completion_rate = (annotated_count / total_images) * 100 if total_images > 0 else 0
        st.metric("Completion Rate", f"{completion_rate:.1f}%")
    
    # Show annotation details
    if st.session_state.annotations:
        with st.expander("📋 View All Annotations"):
            for img_idx, labels in st.session_state.annotations.items():
                if labels:
                    st.write(f"**Image {int(img_idx) + 1}**: {', '.join(labels)}")

# Instructions sidebar
with st.sidebar:
    st.header("📖 Instructions")
    st.markdown("""
    ### How to use this tool:
    
    1. **View Image**: Each page shows one image, please read the image carefully.
    
    2. **Select Labels**: Choose 1-3 labels from the 10 available options which can best describe the image content, OR select "None of the above labels can apply" if no labels fit
    
    3. **Navigate**: Use Previous/Next buttons or jump to specific images (disabled until you make a selection)
    
    4. **Export**: Download your annotations as a JSON file when done
    
    ### Available Labels:
    """)
    
    for i, label in enumerate(LABELS, 1):
        st.write(f"{i}. {label}")
    
    st.markdown("---")
    st.info("💡 Your progress is automatically saved when you export!")

if __name__ == "__main__":
    main()