"""
Flask server for uploading video files and streaming processed video with car detection.
"""

from flask import Flask, request, redirect, render_template, Response
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Array, Queue, Value
import json
import cv2
from cv_detect import process_video, IMG_FORMATS, VID_FORMATS

# Initialize Flask application
app = Flask(__name__)

# Configuration settings
app.config['UPLOAD_FOLDER'] = Path('./files')
app.config['CONTENT_FOLDER'] = Path('./pages')
app.config['ALLOWED_EXTENSIONS'] = IMG_FORMATS + VID_FORMATS

# Ensure upload folder exists
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Shared memory and multiprocessing variables
tracking_array = Array('i', [0, 0])  # Shared array for tracking car presence (left, right)
video_processing = mp.Process()  # Process for video processing
video_processing_stop_signal = Value('i', 0)  # Signal to stop video processing
frame_queue = Queue()  # Queue for video frames


def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Handle file uploads and redirect to prediction page."""
    if request.method == 'POST':
        if 'file' not in request.files:
            # Assert file is present in the request
            return redirect(request.url)
        file = request.files['file']

        # Assert file has a valid name
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(app.config['UPLOAD_FOLDER'] / filename)
            return redirect(f'/complete_{filename}')
        
    return render_template('download_page.html')

@app.route('/complete_<string:path>')
def complete_download(path: str):
    """
    Start video processing for the uploaded file and render completion page.
    """
    file_path = app.config['UPLOAD_FOLDER'] / path
    assert file_path.exists(), f"File {path} not found"
    
    global video_processing
    video_processing = mp.Process(
        target=process_video,
        args=(str(file_path), tracking_array, frame_queue, video_processing_stop_signal)
    )
    video_processing.start()
    return render_template('complete_page.html')

@app.route('/state')
def tracking_state():
    response = app.response_class(
        response=json.dumps(
            {
            'is_car_leftside'   :   tracking_array[0],
            'is_car_rightside'  :    tracking_array[1]
            }
        ),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/stop')
def stop_processing():
    video_processing_stop_signal.value = 1
    if video_processing.is_alive():
        video_processing.terminate()
    return render_template('stop.html')


class generator_video_feed():
    def __init__(self):
        self.q = frame_queue
        self.prev_image = None

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        """Generate the next video frame for streaming."""
        output_frame = self.q.get()
        (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
        if flag:
            self.prev_image = encodedImage
            return(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                    bytearray(encodedImage) + b'\r\n')
        else:
            return(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                    bytearray(self.prev_image) + b'\r\n')

	
generator = generator_video_feed()
@app.route("/video_feed")
def video_feed():
    """Stream video feed with processed frames."""
    return Response(generator, mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video")
def video_page():
    """Render the video feed page."""
    return render_template('video.html')


if __name__ == "__main__":
    app.run(debug=False, port=8080)