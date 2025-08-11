from diarizer.diarizer import Diarizer

def terminal_input():
    print("Enter video path:")
    video_path = input().strip()
    print("Enter Vosk model directory (e.g., ./models):")
    vosk_model_dir = input().strip()
    print("Enter output JSON path (e.g., ./output.json):")
    output_json = input().strip()
    return video_path, vosk_model_dir, output_json

def main():
    video_path, vosk_model_dir, output_json = terminal_input()
    diarizer = Diarizer(vosk_model_dir)
    diarizer.diarize(video_path, output_json)

if __name__ == "__main__":
    main()
