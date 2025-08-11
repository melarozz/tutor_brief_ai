# TutorBriefAI - Lesson Transcription and Summarization Tool

**TutorBriefAI** is a powerful tool to transcribe videos of lessons and generate concise summaries using AI — designed to help tutors save time and focus on what matters!

---

## Features

- Transcribes audio from lesson videos using Vosk speech recognition  
- Speaker diarization and segmentation  
- AI-powered summarization of transcripts  
- Clean, user-friendly web interface powered by Flask

---

## Instructions

1. **Download the Vosk Russian model** (e.g., [vosk-ru](https://alphacephei.com/vosk/models))  
2. Place the extracted model folder inside the `./models` directory (ensure path is `./models/vosk-ru`)  
3. Install dependencies:

   ```bash
   pip install -r requirements.txt

4. **Install FFmpeg** on your system:

   * **Ubuntu/Debian:**

     ```bash
     sudo apt update && sudo apt install ffmpeg
     ```

   * **macOS (using Homebrew):**

     ```bash
     brew install ffmpeg
     ```

   * **Windows:**

     Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) and add it to your system PATH.

5. **Set your MISTRAL API key** as an environment variable:

   * On Linux/macOS:

     ```bash
     export MISTRAL_API_KEY="your_api_key_here"
     ```

   * On Windows (PowerShell):

     ```powershell
     setx MISTRAL_API_KEY "your_api_key_here"
     ```

     *(Restart your terminal after setting this.)*

6. Run the web app:

   ```bash
   python app.py
   ```

7. Open your browser and go to [http://localhost:5000](http://localhost:5000)

8. Upload your lesson video and get transcription & summary

---

## Usage Options

* **Web UI:** Run `python app.py` and use the browser interface.
* **CLI:** Run `python diarize_transcribe_py.py <video_path> <vosk_model_dir> <output_json>`
* **Terminal input mode:** Run `python main.py` and follow interactive prompts.

---

## To Do

* Add a `requirements.txt` file listing all dependencies
* Improve UI/UX
* Add progress bar for processing status
* Optimize performance and error handling
* Fix bug with MISTRAL\_API\_KEY
* Move to React

---

Happy tutoring! 😊
