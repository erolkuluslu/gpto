from dotenv import load_dotenv
from openai import OpenAI
import cv2
import os
import base64
from moviepy.editor import VideoFileClip

class ChatAssistant:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI()

    def get_joke(self):
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a short joke about AI."}
            ]
        )
        return completion.choices[0].message.content

    def solve_math_problem(self):
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that responds in Markdown. Help me with my math homework!"},
                { "role": "user", "content": [
                    {"type": "text", "text": "What's the area of the triangle?"},
                    {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/e/e2/The_Algebra_of_Mohammed_Ben_Musa_-_page_82b.png"}}
                ]}
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    def generate_video_summary(self, base64Frames, transcription_text):
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are generating a video summary. Please provide a summary of the video. Respond in Markdown."},
                {
                    "role": "user", 
                    "content": [
                        "These are the frames from the video.",
                        *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames),
                        {"type": "text", "text": f"The audio transcription is: {transcription_text}"}
                    ],
                }
            ],
            temperature=0,
        )
        return response.choices[0].message.content

class VideoProcessor:
    @staticmethod
    def process_video(video_path, seconds_per_frame=2):
        base64Frames = []
        base_video_path, _ = os.path.splitext(video_path)

        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frames_to_skip = int(fps * seconds_per_frame)
        curr_frame = 0

        while curr_frame < total_frames - 1:
            video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
            curr_frame += frames_to_skip
        video.release()

        audio_path = f"{base_video_path}.mp3"
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, bitrate="32k")
        clip.audio.close()
        clip.close()

        return base64Frames, audio_path

class TranscriptionProcessor:
    def __init__(self, client):
        self.client = client

    def generate_transcript_summary(self, audio_path):
        transcription = self.client.audio.transcriptions.create(
            model="whisper-1", 
            file=open(audio_path,"rb"),
        )
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are generating a transcript summary. Create a summary of the provided transcription. Respond in Markdown."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"The audio transcription is: {transcription.text}"}
                ]}
            ],
            temperature=0,
        )
        return response.choices[0].message.content

# Example usage:
if __name__ == "__main__":
    assistant = ChatAssistant()
    video_processor = VideoProcessor()
    transcription_processor = TranscriptionProcessor(assistant.client)

    joke = assistant.get_joke()
    print("Assistant:", joke)

    math_solution = assistant.solve_math_problem()
    print("Assistant:", math_solution)

    base64Frames, audio_path = video_processor.process_video("data/bison.mp4", seconds_per_frame=1)

    video_summary = assistant.generate_video_summary(base64Frames, transcription_processor.generate_transcript_summary(audio_path))
    print(video_summary)
