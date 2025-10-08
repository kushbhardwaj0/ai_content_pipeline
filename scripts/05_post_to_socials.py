# 05_post_to_socials.py
from instagrapi import Client
import os
import random
import time
import httplib2
import argparse
from apiclient.discovery import build
from apiclient.errors import HttpError
from apiclient.http import MediaFileUpload
from oauth2client.client import flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client.tools import run_flow

def post_to_instagram(video_path, caption):
    """
    Posts a video to Instagram as a Reel.
    """
    cl = Client()
    username = os.environ.get("INSTA_USERNAME")
    password = os.environ.get("INSTA_PASSWORD")

    if not username or not password:
        print("Instagram credentials not found in environment variables.")
        return

    try:
        cl.login(username, password)
        media = cl.clip_upload(path=video_path, caption=caption)
        print(f"Reel posted to Instagram! Media ID: {media.dict()['id']}")
    except Exception as e:
        print(f"An error occurred while posting to Instagram: {e}")

def post_to_youtube(video_path, title, description, category, keywords, privacy_status):
    """
    Uploads a video to YouTube.
    """
    CLIENT_SECRETS_FILE = "client_web.json" # Should be in the root directory
    YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"

    def get_authenticated_service():
        flow = flow_from_clientsecrets(CLIENT_SECRETS_FILE, scope=YOUTUBE_UPLOAD_SCOPE)
        storage = Storage(f"{os.path.splitext(CLIENT_SECRETS_FILE)[0]}-oauth2.json")
        credentials = storage.get()

        if not credentials or credentials.invalid:
            flags = argparse.Namespace(
                auth_host_name='localhost',
                auth_host_port=[8080, 8090],
                logging_level='ERROR',
                noauth_local_webserver=True,
            )
            credentials = run_flow(flow, storage, flags=flags)
        return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, credentials=credentials)

    def initialize_upload(youtube, options):
        tags = options.keywords.split(",") if options.keywords else None
        body = {
            "snippet": {
                "title": options.title,
                "description": options.description,
                "tags": tags,
                "categoryId": options.category
            },
            "status": {
                "privacyStatus": options.privacyStatus
            }
        }
        insert_request = youtube.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=MediaFileUpload(options.file, chunksize=-1, resumable=True)
        )
        resumable_upload(insert_request)

    def resumable_upload(insert_request):
        response = None
        error = None
        retry = 0
        while response is None:
            try:
                print("Uploading file to YouTube...")
                status, response = insert_request.next_chunk()
                if response is not None:
                    if "id" in response:
                        print(f"Video id '{response['id']}' was successfully uploaded to YouTube.")
                        return
                    else:
                        raise Exception(f"Unexpected response: {response}")
            except HttpError as e:
                if e.resp.status in [500, 502, 503, 504]:
                    error = f"A retriable HTTP error {e.resp.status} occurred:\n{e.content}"
                else:
                    raise
            except (httplib2.HttpLib2Error, IOError) as e:
                error = f"A retriable error occurred: {e}"

            if error:
                print(error)
                retry += 1
                if retry > 10:
                    raise Exception("No longer attempting to retry.")
                sleep_seconds = random.random() * (2 ** retry)
                print(f"Sleeping {sleep_seconds:.2f} seconds and retrying...")
                time.sleep(sleep_seconds)

    class Args:
        def __init__(self):
            self.file = video_path
            self.title = title
            self.description = description
            self.category = category
            self.keywords = keywords
            self.privacyStatus = privacy_status
    
    args = Args()

    if not os.path.exists(args.file):
        raise FileNotFoundError(f"Video file '{args.file}' not found.")

    youtube = get_authenticated_service()
    try:
        initialize_upload(youtube, args)
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")

if __name__ == "__main__":
    video_file = "output/new_final_video.mp4"
    insta_caption = "havent posted in over a month mb. oh wow, you like pink floyd? congrats on sitting through 20 full minutes of 50 year old whale noises and pretentious guitar snoozefests. nothing says 'Im intellectually superior' quite like zoning out to grandpas 'experimental' psychedelic lullabies while pretending youve unlocked lifes secrets. Oh, is that the genius of roger waters profound lyrics or is it just another 5-minute long keyboard chord? stop acting like deciphering Floyd lyrics is your phd thesis. go touch grass instead of worshipping album covers painted by a bored art student. youre not deep, just annoying. #ai #brainrot #reddit #fyp #aita #aitah"
    yt_title = "Self-tuned AI Generated Brainrot Content"
    yt_description = "Dutch's final speech copypasta: I got a plan John. This is a good one. We can’t always fight nature, John. We can’t fight change. We can’t fight gravity. We can’t fight nothing. My whole life, all I ever did was fight. But I can’t give up, neither. I can’t fight my own nature. That’s a paradox, John. You see? When I’m gone, they’ll just find another monster. They have to, because they have to justify their wages. Our time has passed, John."
    yt_category = "27"
    yt_keywords = "ai,brainrot,reddit,fyp"
    yt_privacy = "public"

    post_to_instagram(video_file, insta_caption)
    post_to_youtube(video_file, yt_title, yt_description, yt_category, yt_keywords, yt_privacy)
