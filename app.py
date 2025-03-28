from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import time
import atexit
import base64
from datetime import datetime
import uuid
import math

# Import our utility functions
import src.app_utils as app_utils
import src.model_utils as model_utils

# # Set up environment for development if needed
# if os.environ.get('FLASK_ENV') != 'production':
#     os.environ['FLASK_ENV'] = 'development'
os.environ['FLASK_ENV'] = 'production'

app = Flask(__name__)

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'webm', 'ogg', 'mov', 'MOV'}
MAX_CONTENT_LENGTH = 40 * 1024 * 1024  # 40MB max upload size

# Set maximum content length
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Get file extension
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        print(f"File extension: {file_extension}")
        # Save the file to temp directory
        unique_filename = app_utils.generate_unique_filename(extension=file_extension)
        file_data = file.read()
        _, file_path = app_utils.save_file_to_temp(file_data, unique_filename)
        
        # Determine if it's an image or video
        file_type = 'image' if file_extension in ['png', 'jpg', 'jpeg'] else 'video'
        
        return jsonify({
            'success': True,
            'filename': unique_filename,
            'file_type': file_type,
            'message': f'{file_type.capitalize()} uploaded successfully'
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/save-capture', methods=['POST'])
def save_capture():
    if 'data' not in request.json:
        return jsonify({'error': 'No data provided'}), 400
    
    data_url = request.json['data']
    file_type = request.json['type']  # 'image' or 'video'
    
    # Process based on file type
    if file_type == 'image':
        # Format: data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/...
        header, encoded = data_url.split(",", 1)
        data = base64.b64decode(encoded)
        
        # Save to temp directory
        unique_filename = app_utils.generate_unique_filename(extension='jpg')
        _, file_path = app_utils.save_file_to_temp(data, unique_filename)
        
    else:  # video
        # For video, the data is already a Blob that was converted to base64
        data = base64.b64decode(data_url)
        
        # Save to temp directory
        unique_filename = app_utils.generate_unique_filename(extension='webm')
        _, file_path = app_utils.save_file_to_temp(data, unique_filename)
    
    return jsonify({
        'success': True,
        'filename': unique_filename,
        'file_type': file_type,
        'message': f'{file_type.capitalize()} saved successfully'
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Serve files from the temporary directory
    return send_from_directory(app_utils.get_temp_directory(), filename)

@app.route('/process', methods=['POST'])
def process_media():
    # Get request data
    filename = request.json.get('filename')
    file_type = request.json.get('file_type')
    
    app.logger.info(f"Processing media: filename={filename}, file_type={file_type}")
    
    if not filename:
        app.logger.error("No filename provided in request")
        return jsonify({'error': 'No filename provided'}), 400
    
    # Get the full file path
    file_path = app_utils.get_file_path(filename)
    
    if not os.path.exists(file_path):
        app.logger.error(f"File not found at path: {file_path}")
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Process based on file type
        if file_type == 'image':
            # Process image
            try:
                # Try to use the model if available
                app.logger.info(f"Predicting with model on {file_path}")
                
                # First check if we can load the model
                try:
                    model = model_utils.load_model()
                    if model is None:
                        raise ValueError("Model could not be loaded")
                except Exception as model_load_error:
                    app.logger.error(f"Error loading model: {str(model_load_error)}", exc_info=True)
                    if os.environ.get('FLASK_ENV') == 'development':
                        # In development, continue with mock data
                        app.logger.warning("Using mock data for development")
                        from src.model_utils import KeyPoints
                        keypoints = []
                        for i in range(24):  # Generate 24 keypoints
                            keypoint_name = KeyPoints(i).name if i < len(KeyPoints) else f"KEYPOINT_{i}"
                            keypoints.append({
                                'id': i,
                                'name': keypoint_name,
                                'x': float(100 + i*10),
                                'y': float(200 + i*5),
                                'z': float(i * 0.1),
                                'confidence': float(0.9 - i*0.02)
                            })
                        plot_base64 = None
                        return jsonify({
                            'success': True,
                            'results': {
                                'message': f'Processed {file_type} {filename} with mock data (model not available)',
                                'keypoints': keypoints,
                                'plot': plot_base64
                            }
                        })
                    else:
                        # In production, report the error
                        raise ValueError(f"Failed to load pose estimation model: {str(model_load_error)}")
                
                # Now make predictions
                org_img, real_coords, normalized_coords, z_scaled = model_utils.predict(file_path, model)
                
                # Convert keypoints to the format expected by the frontend with keypoint names and z-coordinates
                from src.model_utils import KeyPoints
                
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
                        'confidence': float(point[2])
                    }
                    keypoints.append(keypoint)
                
                # Generate 3D plot as base64 image
                app.logger.info("Generating 3D plot")
                plot_base64 = model_utils.plot_keypoints_to_base64(org_img, normalized_coords, z_scaled)
                
                if plot_base64 is None:
                    app.logger.warning("Could not generate 3D plot, but keypoints were detected")
                else:
                    app.logger.info("3D plot generated successfully")
                
            except Exception as model_error:
                app.logger.error(f"Error using model: {str(model_error)}", exc_info=True)
                # Fallback to placeholder data if model fails
                from src.model_utils import KeyPoints
                keypoints = []
                for i in range(24):  # Generate 24 keypoints
                    keypoint_name = KeyPoints(i).name if i < len(KeyPoints) else f"KEYPOINT_{i}"
                    keypoints.append({
                        'id': i,
                        'name': keypoint_name,
                        'x': float(100 + i*10),
                        'y': float(200 + i*5),
                        'z': float(i * 0.1),
                        'confidence': float(0.9 - i*0.02)
                    })
                plot_base64 = None
                return jsonify({
                    'success': False,
                    'error': f"Error processing image: {str(model_error)}"
                }), 500
        else:  # video
            # Process video
            try:
                # Try to use the model if available
                app.logger.info(f"Processing video: {file_path}")
                
                # First check if we can load the model
                try:
                    model = model_utils.load_model()
                    if model is None:
                        raise ValueError("Model could not be loaded")
                except Exception as model_load_error:
                    app.logger.error(f"Error loading model for video: {str(model_load_error)}", exc_info=True)
                    if os.environ.get('FLASK_ENV') == 'development':
                        # In development, continue with mock data
                        app.logger.warning("Using mock data for video in development")
                        from src.model_utils import KeyPoints
                        # Generate mock keypoints for multiple frames
                        keypoints = []
                        video_info = {
                            'fps': 30,
                            'width': 640,
                            'height': 480,
                            'total_frames': 100,
                            'processed_frames': 10
                        }
                        
                        # Create mock data for 10 frames
                        for frame_num in range(0, 100, 10):  # Every 10th frame
                            frame_keypoints = []
                            for i in range(24):  # Generate 24 keypoints per frame
                                keypoint_name = KeyPoints(i).name if i < len(KeyPoints) else f"KEYPOINT_{i}"
                                # Add a little movement to simulate motion
                                x_offset = math.sin(frame_num * 0.1) * 20
                                y_offset = math.cos(frame_num * 0.1) * 10
                                frame_keypoints.append({
                                    'id': i,
                                    'name': keypoint_name,
                                    'x': float(100 + i*10 + x_offset),
                                    'y': float(200 + i*5 + y_offset),
                                    'z': float(i * 0.1),
                                    'confidence': float(0.9 - i*0.02),
                                    'frame': frame_num
                                })
                            
                            keypoints.extend(frame_keypoints)
                        
                        return jsonify({
                            'success': True,
                            'results': {
                                'message': f'Processed {file_type} {filename} with mock data (model not available)',
                                'keypoints': keypoints,
                                'video_info': video_info,
                                'plot': None  # No plot for mock data
                            }
                        })
                    else:
                        # In production, report the error
                        raise ValueError(f"Failed to load pose estimation model for video: {str(model_load_error)}")
                
                # Process the video frames
                frame_interval = 5  # Process every 5th frame
                frame_results, video_info = model_utils.process_video(file_path, model, frame_interval=frame_interval)
                
                # Prepare keypoints data from all frames
                keypoints = []
                plots = {}  # Dictionary of frame_num -> plot_base64
                
                for result in frame_results:
                    frame_num = result['frame']
                    keypoints.extend(result['keypoints'])
                    if result.get('plot'):
                        plots[frame_num] = result['plot']
                
                app.logger.info(f"Processed {len(frame_results)} frames from video")
                app.logger.info(f"Total keypoints: {len(keypoints)}")
                
                return jsonify({
                    'success': True,
                    'results': {
                        'message': f'Processed {file_type} {filename}',
                        'keypoints': keypoints,
                        'video_info': video_info,
                        'plots': plots
                    }
                })
                
            except Exception as model_error:
                app.logger.error(f"Error processing video: {str(model_error)}", exc_info=True)
                # Fallback to placeholder data if model fails
                from src.model_utils import KeyPoints
                keypoints = []
                for i in range(24):  # Just one frame of mock data
                    keypoint_name = KeyPoints(i).name if i < len(KeyPoints) else f"KEYPOINT_{i}"
                    keypoints.append({
                        'id': i,
                        'name': keypoint_name,
                        'x': float(100 + i*10),
                        'y': float(200 + i*5),
                        'z': float(i * 0.1),
                        'confidence': float(0.9 - i*0.02),
                        'frame': 0
                    })
                
                return jsonify({
                    'success': False,
                    'error': f"Error processing video: {str(model_error)}"
                }), 500
        
        return jsonify({
            'success': True,
            'results': {
                'message': f'Processed {file_type} {filename}',
                'keypoints': keypoints,
                'plot': plot_base64
            }
        })
    
    except Exception as e:
        app.logger.error(f"Error processing {file_type}: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f"Error processing {file_type}: {str(e)}"
        }), 500

# Register cleanup function to run when the app exits
atexit.register(app_utils.delete_temp_directory)

if __name__ == '__main__':
    # Log cleanup settings
    app.logger.info(f"Temporary files will be automatically deleted after {app_utils.MAX_FILE_AGE} seconds")
    app.logger.info(f"Cleanup runs every {app_utils.CLEANUP_INTERVAL} seconds")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)