import os
import shutil
import uuid
import logging
import threading
import time
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('app_utils')

# Global variable to store the path to the temp directory
TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'temp')

# Cleanup settings
MAX_FILE_AGE = 900  # Maximum file age in seconds (15 minutes)
CLEANUP_INTERVAL = 300  # Cleanup interval in seconds (5 minutes)
cleanup_thread = None

def initialize_temp_directory():
    """Initialize a temporary directory for file uploads."""
    global TEMP_DIR
    
    # Create the temp directory if it doesn't exist
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        logger.info(f"Temporary directory created at: {TEMP_DIR}")
    else:
        logger.info(f"Using existing temporary directory at: {TEMP_DIR}")
    
    # Start the scheduled cleanup thread
    start_scheduled_cleanup()
    
    return TEMP_DIR

def get_temp_directory():
    """Get the temporary directory path, creating it if it doesn't exist."""
    global TEMP_DIR
    
    if not os.path.exists(TEMP_DIR):
        initialize_temp_directory()
    
    return TEMP_DIR

def generate_unique_filename(original_filename=None, extension=None):
    """Generate a unique filename for an uploaded file."""
    if original_filename and not extension:
        # Extract extension from original filename
        _, ext = os.path.splitext(original_filename)
        extension = ext[1:] if ext else 'bin'
    elif not extension:
        extension = 'bin'
    
    return f"{uuid.uuid4().hex}.{extension}"

def save_file_to_temp(file_data, filename=None, extension=None):
    """Save file data to the temporary directory."""
    temp_dir = get_temp_directory()
    
    # Generate a unique filename if not provided
    if not filename:
        filename = generate_unique_filename(extension=extension)
    
    # Create the full file path
    file_path = os.path.join(temp_dir, filename)
    
    # Save the file
    with open(file_path, 'wb') as f:
        f.write(file_data)
    
    logger.info(f"File saved to temporary location: {file_path}")
    return filename, file_path

def get_file_path(filename):
    """Get the full path to a file in the temporary directory."""
    return os.path.join(get_temp_directory(), filename)

def cleanup_temp_directory():
    """Delete all files in the temporary directory."""
    global TEMP_DIR
    
    if os.path.exists(TEMP_DIR):
        try:
            # Count files before deletion
            file_count = len([name for name in os.listdir(TEMP_DIR) if os.path.isfile(os.path.join(TEMP_DIR, name))])
            
            # Remove all files in the directory
            for filename in os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            
            logger.info(f"Cleaned up {file_count} files from temporary directory")
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory: {e}")

def delete_temp_directory():
    """Delete the entire temporary directory."""
    global TEMP_DIR, cleanup_thread
    
    # Stop the cleanup thread if it's running
    if cleanup_thread and cleanup_thread.is_alive():
        # We can't really stop the thread, but we can set a flag
        # In this case, we'll just let it finish naturally
        logger.info("Waiting for cleanup thread to finish...")
    
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
            logger.info(f"Temporary directory removed: {TEMP_DIR}")
        except Exception as e:
            logger.error(f"Error removing temporary directory: {e}")

def cleanup_old_files(max_age=MAX_FILE_AGE):
    """Clean up files older than max_age seconds."""
    global TEMP_DIR
    
    if not os.path.exists(TEMP_DIR):
        return
    
    try:
        now = datetime.now()
        count = 0
        
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            
            if not os.path.isfile(file_path):
                continue
            
            # Get file modification time
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            # If file is older than max_age, delete it
            if now - file_time > timedelta(seconds=max_age):
                try:
                    os.unlink(file_path)
                    count += 1
                except Exception as e:
                    logger.error(f"Error deleting old file {file_path}: {e}")
        
        if count > 0:
            logger.info(f"Cleaned up {count} old files")
    
    except Exception as e:
        logger.error(f"Error during old file cleanup: {e}")

def scheduled_cleanup_task():
    """Task that runs periodically to clean up old files."""
    while True:
        try:
            # Sleep first to allow the server to start up properly
            time.sleep(CLEANUP_INTERVAL)
            
            # Clean up old files
            cleanup_old_files()
            logger.info(f"Scheduled cleanup ran - files older than {MAX_FILE_AGE} seconds will be removed")
        
        except Exception as e:
            logger.error(f"Error in scheduled cleanup: {e}")
            # Sleep a bit before retrying
            time.sleep(60)

def start_scheduled_cleanup():
    """Start the scheduled cleanup thread."""
    global cleanup_thread
    
    if cleanup_thread is None or not cleanup_thread.is_alive():
        cleanup_thread = threading.Thread(target=scheduled_cleanup_task, daemon=True)
        cleanup_thread.start()
        logger.info(f"Started scheduled cleanup thread (interval: {CLEANUP_INTERVAL}s, max age: {MAX_FILE_AGE}s)")

# Initialize temp directory when module is imported
initialize_temp_directory() 