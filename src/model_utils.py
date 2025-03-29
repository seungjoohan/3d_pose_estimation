import os
import tensorflow as tf
import numpy as np
import logging
import cv2
import matplotlib
# Set Matplotlib to use a non-GUI backend to avoid thread issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tempfile
import os
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('model_utils')

# General utility functions
def get_project_root():
    """Get the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_model_directory():
    """Get the absolute path to the model directory."""
    return os.path.join(get_project_root(), 'model')

def ensure_directory_exists(directory_path):
    """Ensure that a directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")
    return directory_path

def get_model_path():
    """Get the absolute path to the model file."""
    if os.environ.get('K_SERVICE'):
        from google.cloud import storage
        temp_dir = tempfile.mkdtemp()
        destination = os.path.join(temp_dir, 'keypoint_estimation')
        os.makedirs(destination, exist_ok=True)

        client = storage.Client()
        bucket = client.bucket(f"{os.environ.get('GOOGLE_CLOUD_PROJECT')}-models")
        blobs = list(bucket.list_blobs(prefix='keypoint_estimation'))
        for blob in blobs:
            if blob.name.endswith('/'):
                continue
            rel_path = blob.name.replace('keypoint_estimation/', '')
            if not rel_path:
                continue

            dest_path = os.path.join(destination, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            blob.download_to_filename(dest_path)

        return destination
    else:
        return os.path.join(get_project_root(), 'model', 'keypoint_estimation')

# Global variables
MODEL = None
MODEL_PATH = get_model_path()
logger.info(f"Using model path: {MODEL_PATH}")
print(MODEL_PATH)

com_weights = np.array([
        0.081,
        0,
        0.140042,
        0.019204,
        0.015004,
        0.140042,
        0.019204,
        0.015004,
        0.18095,
        0.067334,
        0.036966,
        0.18095,
        0.067334,
        0.036966,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

class KeyPoints(Enum):
    TOP = 0
    NECK = 1
    RIGHT_SHOULDER = 2
    RIGHT_ELBOW = 3
    RIGHT_WRIST = 4
    LEFT_SHOULDER = 5
    LEFT_ELBOW = 6
    LEFT_WRIST = 7
    RIGHT_HIP = 8
    RIGHT_KNEE = 9
    RIGHT_ANKLE = 10
    LEFT_HIP = 11
    LEFT_KNEE = 12
    LEFT_ANKLE = 13
    NOSE = 14
    RIGHT_EYE = 15
    RIGHT_EAR = 16
    LEFT_EYE = 17
    LEFT_EAR = 18
    SPINE = 19
    RIGHT_FINGER = 20
    RIGHT_TOE = 21
    LEFT_FINGER = 22
    LEFT_TOE = 23
    STERNUM = 24
    SACRUM = 25

class KeyPointConnections:
    links = [
        {"from": KeyPoints.TOP, "to": KeyPoints.NECK, "color": (25, 175, 25)},
        {"from": KeyPoints.NECK, "to": KeyPoints.RIGHT_SHOULDER, "color": (25, 25, 175)},
        {"from": KeyPoints.NECK, "to": KeyPoints.LEFT_SHOULDER, "color": (255, 25, 25)},
        {"from": KeyPoints.LEFT_WRIST, "to": KeyPoints.LEFT_ELBOW, "color": (255, 25, 25)},
        {"from": KeyPoints.LEFT_ELBOW, "to": KeyPoints.LEFT_SHOULDER, "color": (255, 25, 25)},
        {"from": KeyPoints.LEFT_SHOULDER, "to": KeyPoints.RIGHT_SHOULDER, "color": (25, 175, 25)},
        {"from": KeyPoints.RIGHT_SHOULDER, "to": KeyPoints.RIGHT_ELBOW, "color": (25, 25, 175)},
        {"from": KeyPoints.RIGHT_ELBOW, "to": KeyPoints.RIGHT_WRIST, "color": (25, 25, 175)},
        {"from": KeyPoints.SPINE, "to": KeyPoints.STERNUM, "color": (25, 175, 25)},
        {"from": KeyPoints.LEFT_HIP, "to": KeyPoints.RIGHT_HIP, "color": (25, 175, 25)},
        {"from": KeyPoints.SPINE, "to": KeyPoints.SACRUM, "color": (25, 175, 25)},
        {"from": KeyPoints.LEFT_HIP, "to": KeyPoints.LEFT_KNEE, "color": (255, 25, 25)},
        {"from": KeyPoints.LEFT_KNEE, "to": KeyPoints.LEFT_ANKLE, "color": (255, 25, 25)},
        {"from": KeyPoints.RIGHT_HIP, "to": KeyPoints.RIGHT_KNEE, "color": (25, 25, 175)},
        {"from": KeyPoints.RIGHT_KNEE, "to": KeyPoints.RIGHT_ANKLE, "color": (25, 25, 175)},
        {"from": KeyPoints.NOSE, "to": KeyPoints.TOP, "color": (25, 175, 25)},
        {"from": KeyPoints.NOSE, "to": KeyPoints.NECK, "color": (25, 175, 25)},
        {"from": KeyPoints.NOSE, "to": KeyPoints.RIGHT_EYE, "color": (25, 25, 175)},
        {"from": KeyPoints.NOSE, "to": KeyPoints.LEFT_EYE, "color": (255, 25, 25)},
        {"from": KeyPoints.RIGHT_EYE, "to": KeyPoints.RIGHT_EAR, "color": (25, 25, 175)},
        {"from": KeyPoints.LEFT_EYE, "to": KeyPoints.LEFT_EAR, "color": (255, 25, 25)},
        {"from": KeyPoints.STERNUM, "to": KeyPoints.RIGHT_SHOULDER, "color": (25, 25, 175)},
        {"from": KeyPoints.STERNUM, "to": KeyPoints.LEFT_SHOULDER, "color": (255, 25, 25)},
        {"from": KeyPoints.SACRUM, "to": KeyPoints.RIGHT_HIP, "color": (25, 25, 175)},
        {"from": KeyPoints.SACRUM, "to": KeyPoints.LEFT_HIP, "color": (255, 25, 25)},
        {"from": KeyPoints.RIGHT_WRIST, "to": KeyPoints.RIGHT_FINGER, "color": (25, 25, 175)},
        {"from": KeyPoints.LEFT_WRIST, "to": KeyPoints.LEFT_FINGER, "color": (255, 25, 25)},
        {"from": KeyPoints.RIGHT_ANKLE, "to": KeyPoints.RIGHT_TOE, "color": (25, 25, 175)},
        {"from": KeyPoints.LEFT_ANKLE, "to": KeyPoints.LEFT_TOE, "color": (255, 25, 25)}
    ]

# Model loading functions
def load_model(model_path=None):
    """
    Load the TensorFlow model.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        The loaded TensorFlow model
    """
    global MODEL, MODEL_PATH
    
    if model_path:
        MODEL_PATH = model_path
    
    # If model is already loaded, return it
    if MODEL is not None:
        return MODEL
    
    try:
        # Check if model path exists
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model path does not exist: {MODEL_PATH}")
            # For testing or development, provide mock model
            if os.getenv('FLASK_ENV') == 'development':
                logger.warning("Development environment detected. Using mock model data.")
                return create_mock_model()
            else:
                raise FileNotFoundError(f"Model path not found: {MODEL_PATH}")
        
        # Load the model
        logger.info(f"Loading model from {MODEL_PATH}")
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        
        # Warm up the model with a dummy prediction
        warmup_model(MODEL)
        
        logger.info("Model loaded successfully")
        return MODEL
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        if os.getenv('FLASK_ENV') == 'development':
            logger.warning("Development environment detected. Using mock model data.")
            return create_mock_model()
        raise

def create_mock_model():
    """
    Creates a mock model for testing or development purposes.
    
    Returns:
        A simple mock model object that can be used for testing
    """
    # This is a very basic mock that can be expanded based on your needs
    class MockModel:
        def __init__(self):
            self.input_shape = [None, 384, 384, 3]
            self.output_shape = {
                0: [None, 48, 48, 24],  # Heatmap
                1: [None, 24]  # 3D keypoints
            }
        
        def predict(self, input_data, verbose=0):
            # Create random outputs that match the expected shapes
            batch_size = input_data.shape[0]
            heatmap = np.random.normal(0, 0.1, (batch_size, 48, 48, 24))
            keypoints_3d = np.random.normal(0, 0.1, (batch_size, 24))
            return [heatmap, keypoints_3d]
    
    return MockModel()

def warmup_model(model):
    """
    Warm up the model with a dummy prediction to initialize weights.
    
    Args:
        model: The TensorFlow model
    """
    try:
        # Create a dummy input tensor
        # Adjust the shape based on your model's input requirements
        dummy_input = np.zeros((1, 384, 384, 3), dtype=np.float32)
        
        # Make a prediction
        _ = model.predict(dummy_input, verbose=0)
            
        logger.info("Model warmed up successfully")
    
    except Exception as e:
        logger.error(f"Error warming up model: {e}")


def load_image(image_path, target_size=(384, 384), normalize=False):
    """
    Load and preprocess an image.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for the image (height, width)
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Preprocessed image (model input) as a numpy array
        original image
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Resize
        if target_size:
            load_img = cv2.resize(img, target_size)
        
        # Normalize to [0, 1]
        if normalize:
            load_img = load_img.astype(np.float32) / 255.0
        
        return np.array(load_img, dtype=np.float32), img
    
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise

def preprocess_image(image_path, target_size=(384, 384)):
    """
    Preprocess an image for the model.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for the image (height, width)
        
    Returns:
        Preprocessed image as a numpy array with batch dimension.
        org_img_size: Original image size
    """
    try:
        # Load and preprocess image
        load_img, org_img = load_image(image_path, target_size=target_size)
        input_data = np.expand_dims(load_img, axis=0)
        
        return input_data, org_img
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def convert_to_tensor(image, dtype=tf.float32):
    """
    Convert a numpy array to a TensorFlow tensor.
    
    Args:
        image: Numpy array
        dtype: TensorFlow data type
        
    Returns:
        TensorFlow tensor
    """
    if len(image.shape) == 3:  # Add batch dimension if needed
        image = np.expand_dims(image, axis=0)
    
    return tf.convert_to_tensor(image, dtype=dtype)

def argmax_ind(heatmap):
    """
    Find the index of the maximum value in a heatmap.
    """
    heatmap_squeezed = np.squeeze(heatmap)
    ind = np.unravel_index(np.argmax(heatmap_squeezed), heatmap_squeezed.shape)
    return ind[0], ind[1], heatmap_squeezed[ind[0], ind[1]]

def weighted_location(heatmap, image_size=(48, 48)):
    """
    Get weighted location of the maximum value in a heatmap for a single keypoint.
    """
    heatmap = np.squeeze(heatmap)
    heatmap[heatmap < 0] = 0
    rough_y, rough_x, score = argmax_ind(heatmap)
    padding = 2
    min_x = max(0, rough_x - padding)
    min_y = max(0, rough_y - padding)
    max_x = min(heatmap.shape[1], rough_x + padding + 1)
    max_y = min(heatmap.shape[0], rough_y + padding + 1)
    cropped_heatmap = heatmap[min_y:max_y, min_x:max_x]
    loc_x = np.sum((0.5 + np.arange(min_x, max_x)) * np.sum(cropped_heatmap, axis=0)) / np.sum(cropped_heatmap)
    loc_y = np.sum((0.5 + np.arange(min_y, max_y)) * np.sum(cropped_heatmap, axis=1)) / np.sum(cropped_heatmap)

    loc_x = loc_x / heatmap.shape[1] * image_size[1]
    loc_y = loc_y / heatmap.shape[0] * image_size[0]

    return loc_x, loc_y, score

def convert_heatmap_to_keypoints(heatmap, image_size=(48, 48)):
    """
    Convert a heatmap to a keypoints.
    """
    kp_num = heatmap.shape[-1]
    return [weighted_location(heatmap[:, :, :, i], image_size) for i in range(kp_num)]

def scale_keypoints(keypoints, org_height, org_width, d_height, d_width):
    """
    Scale keypoints to the original image size.
    """
    w_scale = float(d_width) / float(org_width)
    h_scale = float(d_height) / float(org_height)

    return [(((x[0]) * h_scale), ((x[1]) * w_scale), x[2]) for x in keypoints]

def adjust_camera_view(img, keypoints_2d, keypoints_3d):
    """
    Adjust keypoints to the camera view.
    """
    w_h_ratio = img.shape[1] / img.shape[0]
    d_to_camera = np.max((1, np.max(np.abs(keypoints_3d[keypoints_3d < 0] * 1.1))))
    center = np.zeros((len(keypoints_2d), 2)) + 0.5
    x_coords = [i[0] for i in keypoints_2d]
    y_coords = [i[1] for i in keypoints_2d]

    if w_h_ratio >= 1:
        x_coords = np.array(x_coords) * w_h_ratio
        y_coords = np.array(y_coords)
        center[:, 1] = 1 / (2 * w_h_ratio)
    else:
        x_coords = np.array(x_coords) / w_h_ratio
        y_coords = np.array(y_coords)
        center[:, 0] = w_h_ratio / 2

    x_adj = (x_coords - center[:, 0]) * (keypoints_3d + d_to_camera) / d_to_camera
    y_adj = (y_coords - center[:, 1]) * (keypoints_3d + d_to_camera) / d_to_camera
    keypoints = np.array([x_adj * img.shape[1], y_adj * img.shape[0], keypoints_3d * np.max(img.shape)]).T

    return keypoints

def add_sternum_sacrum(keypoints):
    """
    Add sternum and sacrum to the keypoints.
    """
    newshape = list(keypoints.shape)
    newshape[0] = 26
    kp_sternum_sacrum = np.zeros(newshape)
    kp_sternum_sacrum[:24, :] = keypoints
    shoulder_mid = (kp_sternum_sacrum[KeyPoints.RIGHT_SHOULDER.value, :] + kp_sternum_sacrum[KeyPoints.LEFT_SHOULDER.value, :]) / 2
    kp_sternum_sacrum[24, :] = 0.06 * kp_sternum_sacrum[KeyPoints.SPINE.value, :] + 0.94 * shoulder_mid
    hips_mid = (kp_sternum_sacrum[KeyPoints.LEFT_HIP.value, :] + kp_sternum_sacrum[KeyPoints.RIGHT_HIP.value, :]) / 2
    kp_sternum_sacrum[25, :] = 0.25 * kp_sternum_sacrum[KeyPoints.SPINE.value, :] + 0.75 * hips_mid

    return kp_sternum_sacrum

def predict(image_path, model=None):
    """
    Make a prediction on an image.
    
    Args:
        image_path: Path to the image file
        model: The model to use (loads the global model if None)
        
    Returns:
        The model's prediction
    """
    try:
        # Load model if not provided
        if model is None:
            model = load_model()

        # Preprocess the image
        input_shape = model.input_shape
        input_data, org_img = preprocess_image(image_path, target_size=(input_shape[1], input_shape[2]))
        org_width = org_img.shape[1]
        org_height = org_img.shape[0]

        # Make prediction
        if hasattr(model, 'predict'):
            prediction = model.predict(input_data, verbose=0)

            heatmap_2d = [prediction[i] for i in range(len(model.output_shape)) if len(model.output_shape[i]) == 4][-1]
            keypoints_3d = [prediction[i] for i in range(len(model.output_shape)) if len(model.output_shape[i]) == 2]
            keypoints_3d = np.squeeze(keypoints_3d[-1])[:24]
    
            # Get 2d keypoints from heatmap
            keypoints_2d = convert_heatmap_to_keypoints(heatmap_2d)
            real_coords = scale_keypoints(keypoints_2d, model.output_shape[1][1], model.output_shape[1][2], org_height, org_width)
            normalized_coords = [(((x[0]) / org_height), ((x[1]) / org_width), x[2]) for x in real_coords]
            
            z_coords = keypoints_3d - np.sum(keypoints_3d * com_weights)
            x_std = np.std([i[0] for i in real_coords]) / np.max((org_height, org_width))
            y_std = np.std([i[1] for i in real_coords]) / np.max((org_height, org_width))
            scale = (x_std + y_std) / 2
            z_scaled = z_coords * scale

            return org_img, real_coords, normalized_coords, z_scaled
        else:
            return model(input_tensor)
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise

def process_video(video_path, model=None, frame_interval=5):
    """
    Process a video file frame by frame.
    
    Args:
        video_path: Path to the video file
        model: The model to use (loads the global model if None)
        frame_interval: Process every Nth frame
        
    Returns:
        List of frame results with keypoints and visualization
    """
    try:
        # Load model if not provided
        if model is None:
            model = load_model()
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {fps} fps, {frame_width}x{frame_height}, {total_frames} total frames")
        
        frame_count = 0
        results = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame
            if frame_count % frame_interval == 0:
                logger.info(f"Processing frame {frame_count}/{total_frames}")
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Save frame temporarily to use predict function
                temp_frame_path = f"/tmp/frame_{frame_count}.jpg"
                cv2.imwrite(temp_frame_path, frame)
                
                try:
                    # Use the same prediction logic as for images
                    org_img, real_coords, normalized_coords, z_scaled = predict(temp_frame_path, model)
                    
                    # Generate plot for this frame
                    plot_base64 = plot_keypoints_to_base64(org_img, normalized_coords, z_scaled)
                    
                    # Convert keypoints to the format expected by the frontend with keypoint names
                    keypoints = []
                    for i, point in enumerate(real_coords):
                        # Get keypoint name from enum if within range
                        keypoint_name = KeyPoints(i).name if i < len(KeyPoints) else f"KEYPOINT_{i}"
                        
                        # Create keypoint dict with coordinates and confidence
                        keypoint = {
                            'id': i,
                            'name': keypoint_name,
                            'x': float(point[0]),
                            'y': float(point[1]),
                            'z': float(z_scaled[i]) if i < len(z_scaled) else 0.0,
                            'confidence': float(point[2]),
                            'frame': frame_count
                        }
                        keypoints.append(keypoint)
                    
                    # Add result for this frame
                    results.append({
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        'keypoints': keypoints,
                        'plot': plot_base64
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {str(e)}")
                
                # Clean up temp file
                try:
                    os.remove(temp_frame_path)
                except:
                    pass
            
            frame_count += 1
            
            # For development/testing, limit to 20 frames
            if os.getenv('FLASK_ENV') == 'development' and len(results) >= 20:
                logger.info("Development mode: Limiting to 20 processed frames")
                break
        
        cap.release()
        
        # Get video metadata
        video_info = {
            'fps': fps,
            'width': frame_width,
            'height': frame_height,
            'total_frames': total_frames,
            'processed_frames': len(results)
        }
        
        return results, video_info
    
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        raise

def plot_view(fig, grid_position, image, view_name, keypoints, color='blue'):
        """
        Plot a view of the keypoints on an image.
        
        Args:
            fig: matplotlib figure
            grid_position: tuple of (nrows, ncols, index) for subplot
            image: Image as numpy array
            view_name: Title for the view
            keypoints: 3D keypoints
            color: Color for the keypoints and lines
            
        Returns:
            matplotlib axes object
        """
        # Unpack grid position to ensure correct format for add_subplot
        rows, cols, idx = grid_position
        ax = fig.add_subplot(rows, cols, idx, projection='3d')
        ax.set_title(view_name)
        ax.set_box_aspect((1, 1, 1))

        if view_name.lower().startswith('side'):
            keypoints = keypoints[:, [2, 1, 0]]
        elif view_name.lower().startswith('back'):
            new_keypoints = keypoints.copy()
            new_keypoints[:, 0] = -keypoints[:, 0]
            new_keypoints[:, 2] = -keypoints[:, 2]
            keypoints = new_keypoints

        ax.scatter3D(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], color=color)
        
        for line in KeyPointConnections.links:
            ax.plot3D((keypoints[line["from"].value, 0], keypoints[line["to"].value, 0]),
                    (keypoints[line["from"].value, 1], keypoints[line["to"].value, 1]),
                    (keypoints[line["from"].value, 2], keypoints[line["to"].value, 2]), color=color)
        ax.set_xlim(-image.shape[1]/2, image.shape[1]/2)
        ax.set_ylim(-image.shape[0]/2, image.shape[0]/2)
        ax.set_zlim(-np.max(image.shape)/2, np.max(image.shape)/2)
        
        ax.view_init(elev=-90, azim=-90)
        ax.set_zticks([])
        
        return ax

def plot_keypoints_to_base64(image, keypoints_2d, keypoints_3d=None, figsize=(10, 10), color='blue'):
    """
    Plot keypoints on an image and convert to base64 for web display.
    
    Args:
        image: Image as numpy array
        keypoints_2d: Array of 2d keypoints [x, y]
        keypoints_3d: Array of 3d keypoints [z]
        figsize: Figure size
        color: Color of the keypoints
        
    Returns:
        base64 encoded string of the plot image
    """
    import io
    import base64
    
    try:
        # Create the plot figure
        fig = plot_keypoints(image, keypoints_2d, keypoints_3d, figsize, color)
        
        # Save the figure to a BytesIO object
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        
        # Encode the image to base64
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Close the figure to free memory
        plt.close(fig)
        
        return img_str
    except Exception as e:
        logger.error(f"Error creating 3D plot: {e}", exc_info=True)
        # Return None to indicate failure
        return None

def plot_keypoints(image, keypoints_2d, keypoints_3d, figsize=(10, 10), color='blue'):
    """
    Plot keypoints on an image using matplotlib.
    
    Args:
        image: Image as numpy array
        keypoints_2d: Array of 2d keypoints [x, y]
        keypoints_3d: Array of 3d keypoints [z]
        figsize: Figure size
        color: Color for plotting
        
    Returns:
        Matplotlib figure
    """
    keypoints = adjust_camera_view(image, keypoints_2d, keypoints_3d)
    kp_sternum_sacrum = add_sternum_sacrum(keypoints)

    # Create a figure with 3 subplots for different views
    fig = plt.figure(figsize=(figsize[0]*3, figsize[1]))
    
    # 1. Front view (original)
    plot_view(fig, (1, 3, 1), image, 'Front View', kp_sternum_sacrum, color=color)
    
    # 2. Side view (90 degrees rotated along x-axis)
    plot_view(fig, (1, 3, 2), image, 'Side View (90° X-Rotation)', kp_sternum_sacrum, color=color)
    
    # 3. Back view (180 degrees rotated along x-axis)
    plot_view(fig, (1, 3, 3), image, 'Back View (180° X-Rotation)', kp_sternum_sacrum, color=color)
    
    plt.tight_layout()
    return fig
