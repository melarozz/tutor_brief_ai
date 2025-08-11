import os
import threading
import time
from flask import Flask, request, render_template, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
from diarizer.diarizer import Diarizer
from pathlib import Path
import tempfile
import uuid

app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}
VOSK_MODEL_DIR = "./models/vosk-ru"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# In-memory dictionary to track job status: {job_id: {"status": "processing"|"done"|"error", "output": {...} }}
jobs = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def diarize_async(job_id, video_path, output_json):
    try:
        diarizer = Diarizer(VOSK_MODEL_DIR)
        diarizer.diarize(video_path, output_json)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["output"] = {
            "json_file": output_json,
            "srt_file": str(Path(output_json).with_suffix(".srt")),
            "summary_file": str(Path(output_json).with_suffix(".summary.txt"))
        }
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video_file' not in request.files:
            flash('No video file part')
            return redirect(request.url)
        video_file = request.files['video_file']

        if video_file.filename == '':
            flash('No selected video file')
            return redirect(request.url)

        if video_file and allowed_file(video_file.filename):
            filename = secure_filename(video_file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(video_path)

            output_json = os.path.join(app.config['UPLOAD_FOLDER'], filename + '_output.json')

            job_id = str(uuid.uuid4())
            jobs[job_id] = {"status": "processing"}

            # Run diarization in background thread
            thread = threading.Thread(target=diarize_async, args=(job_id, video_path, output_json))
            thread.start()

            # Redirect to processing page
            return redirect(url_for('processing', job_id=job_id))
        else:
            flash('Invalid file type')
            return redirect(request.url)
    return render_template('index.html')

@app.route('/processing/<job_id>')
def processing(job_id):
    job = jobs.get(job_id)
    if not job:
        flash("Invalid job ID")
        return redirect(url_for('index'))

    if job["status"] == "done":
        return redirect(url_for('result', job_id=job_id))
    elif job["status"] == "error":
        flash(f"Error during diarization: {job.get('error', 'Unknown error')}")
        return redirect(url_for('index'))

    # Still processing
    return render_template('processing.html', job_id=job_id)

@app.route('/result/<job_id>')
def result(job_id):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        flash("Result not ready or invalid job ID")
        return redirect(url_for('index'))

    output = job["output"]
    return render_template('result.html',
                           json_file=output["json_file"],
                           srt_file=output["srt_file"],
                           summary_file=output["summary_file"])

@app.route('/download/<path:filename>')
def download_file(filename):
    try:
        return send_file(filename, as_attachment=True)
    except Exception:
        flash('File not found')
        return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
