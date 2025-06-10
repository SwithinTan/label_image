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

# Supported image extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif']

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
        # Define possible image directories
        self.image_directories = [
            os.path.join(data_dir, "images"),
            os.path.join(data_dir),
            os.path.join(SCRIPT_DIR, "images"),
            SCRIPT_DIR
        ]
    
    def load_csv(self, filename):
        """Load CSV file from data directory"""
        file_path = os.path.join(self.data_dir, filename)
        try:
            if not os.path.exists(file_path):
                st.error(f"❌ File not found: {file_path}")
                return None
            
            df = pd.read_csv(file_path)
            return df
            
        except Exception as e:
            st.error(f"❌ Error loading {filename}: {str(e)}")
            return None
    
    def find_image_by_post_id(self, post_id):
        """Find image file by post_id with various extensions"""
        if pd.isna(post_id) or not post_id:
            return None
        
        # Convert post_id to string and clean it
        post_id_str = str(post_id).strip()
        
        # Try each image directory
        for img_dir in self.image_directories:
            if not os.path.exists(img_dir):
                continue
                
            # Try each extension
            for ext in IMAGE_EXTENSIONS:
                # Try both with and without extension (in case post_id already includes extension)
                possible_filenames = [
                    f"{post_id_str}{ext}",
                    f"{post_id_str.lower()}{ext}",
                    f"{post_id_str.upper()}{ext}"
                ]
                
                # If post_id already has an extension, also try it as-is
                if '.' in post_id_str:
                    possible_filenames.append(post_id_str)
                
                for filename in possible_filenames:
                    full_path = os.path.join(img_dir, filename)
                    if os.path.exists(full_path):
                        return full_path
        
        return None
    
    def get_available_images_info(self):
        """Get information about available images in all directories"""
        available_images = {}
        
        for img_dir in self.image_directories:
            if not os.path.exists(img_dir):
                continue
                
            dir_name = os.path.basename(img_dir) if os.path.basename(img_dir) else "root"
            available_images[dir_name] = []
            
            try:
                for file in os.listdir(img_dir):
                    if any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                        available_images[dir_name].append(file)
            except PermissionError:
                available_images[dir_name] = ["Permission denied"]
        
        return available_images

@st.cache_data
def load_images_data():
    """Load image data using the improved DataLoader"""
    loader = DataLoader(DATA_DIR)
    return loader.load_csv('post_img.csv')

def display_image_with_post_id(current_row, loader):
    """Display image using post_id to find local image file"""
    
    # Check if post_id column exists
    if 'post_id' not in current_row.index:
        st.error("❌ 'post_id' column not found in the CSV file")
        return False
    
    post_id = current_row['post_id']
    
    # Try to find and display the image
    image_path = loader.find_image_by_post_id(post_id)
    
    if image_path:
        try:
            st.image(image_path, 
                    caption=f"Post ID: {post_id}", 
                    use_container_width=True)
            return True
        except Exception as e:
            st.error(f"❌ Failed to load image for Post ID: {post_id}")
            return False
    else:
        st.error(f"❌ No image found for Post ID: {post_id}")
        st.info("💡 Please ensure the image file exists in the images directory")
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

def check_current_selection_valid():
    """Check if current selection is valid (has at least one option selected)"""
    current_annotations = st.session_state.annotations.get(
        str(st.session_state.current_image_index), []
    )
    return len(current_annotations) > 0

def main():
    st.title("🏷️ Image Annotation Tool")
    st.markdown("---")
    
    # Load image data
    if st.session_state.images_data is None:
        with st.spinner("Loading image data..."):
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
        
        # Display image using post_id
        image_displayed = display_image_with_post_id(current_row, loader)
        
        if not image_displayed:
            st.error("❌ No image could be displayed for this row")
        
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
        
        # Navigation - Modified to allow Previous/Next only with valid selection
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
        
        # Jump to specific image - Modified to allow jumping without requiring current answer
        st.markdown("---")
        st.write("**🎯 Jump to Specific Question:**")
        target_image = st.number_input(
            "Question number:", 
            min_value=1, 
            max_value=total_images, 
            value=st.session_state.current_image_index + 1,
            key=f"nav_input_{st.session_state.current_image_index}",
            help="You can jump to any question without answering the current one"
        )
        
        # Remove the validation requirement for jumping
        if st.button("🎯 Go to Question", key="jump_button"):
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
    
    # Statistics and summary
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
    
    1. **View Image**: Each page shows one image loaded from local directory using post_id as filename
    
    2. **Select Labels**: Choose 1-3 labels from the 10 available options which can best describe the image content, OR select "None of the above labels can apply" if no labels fit
    
    3. **Navigate**: 
       - Use Previous/Next buttons (requires answering current question)
       - **Jump to any question** using the question number input (no answer required)
    
    4. **Export**: Download your annotations as a JSON file when done
    
    ### Navigation Options:
    - **Previous/Next**: Requires answering the current question
    - **Jump to Question**: Can jump to any question without answering current one
    
    ### Image Loading:
    - Images are loaded using **post_id** as filename
    - Supported formats: **jpg, jpeg, png, gif, bmp, webp, tiff, tif**
    - Searches in: data/images/, data/, images/, script directory
    
    ### Available Labels:
    """)
    
    for i, label in enumerate(LABELS, 1):
        st.write(f"{i}. {label}")
    
    st.markdown("---")
    st.info("💡 Your progress is will NOT be saved, please export!")

if __name__ == "__main__":
    main()