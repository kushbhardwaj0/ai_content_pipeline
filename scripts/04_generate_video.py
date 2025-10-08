# 04_generate_video.py
import whisper
import pysrt
import subprocess
import os

def generate_video(audio_path, video_path, output_dir="output"):
    """
    Generates a video with subtitles from the audio and a background video.
    """
    srt_file = os.path.join(output_dir, "subtitles.srt")
    output_video = os.path.join(output_dir, "final_video.mp4")
    new_video_file = os.path.join(output_dir, 'new_final_video.mp4')

    # Load Whisper model
    model_audio = whisper.load_model("base")

    # Transcribe the audio
    print("Transcribing audio...")
    result = model_audio.transcribe(audio_path, task='transcribe')
    segments = result['segments']

    # Helper function to convert seconds to SubRipTime
    def seconds_to_subriptime(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int(round((seconds - int(seconds)) * 1000))
        return pysrt.SubRipTime(hours=hours, minutes=minutes, seconds=secs, milliseconds=milliseconds)

    # Create SRT file
    subs = pysrt.SubRipFile()
    for i, segment in enumerate(segments, start=1):
        start_time = seconds_to_subriptime(segment['start'])
        end_time = seconds_to_subriptime(segment['end'])
        text = segment['text'].strip().replace('--', '-')
        sub = pysrt.SubRipItem(index=i, start=start_time, end=end_time, text=text)
        subs.append(sub)
    subs.save(srt_file, encoding='utf-8')
    print(f"Subtitles saved to {srt_file}")

    # Function to get duration of a media file
    def get_media_duration(file_path):
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        return float(result.stdout)

    # Get durations
    audio_duration = get_media_duration(audio_path)
    video_duration = get_media_duration(video_path)
    shortest_duration = min(audio_duration, video_duration)

    # FFmpeg command to burn subtitles and trim video
    command = [
        'ffmpeg', '-y', '-i', video_path, '-i', audio_path,
        '-vf', f"subtitles={srt_file}:force_style='FontName=Arial,FontSize=20,PrimaryColour=&HFFFFFF&'",
        '-c:v', 'libx264', '-c:a', 'aac', '-b:a', '192k', '-t', str(shortest_duration), output_video
    ]
    print("Burning subtitles and setting audio...")
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        print("FFmpeg error (burning subtitles):", process.stderr)
    else:
        print(f"Intermediate video created: {output_video}")

    # Combine final video with original audio
    ffmpeg_command = [
        'ffmpeg', '-y', '-i', output_video, '-i', audio_path,
        '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', new_video_file
    ]
    print("Combining video and audio...")
    process = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        print("FFmpeg error (combining audio):", process.stderr)
    else:
        print(f"Final video created: {new_video_file}")

    return new_video_file

if __name__ == "__main__":
    audio_file = "output/output.mp3"
    video_file = "data/brain_number_trimmed.mp4"  # Make sure this path is correct
    generate_video(audio_file, video_file)
