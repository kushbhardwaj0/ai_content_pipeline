# AI Content Pipeline

This project is an AI-powered content pipeline that generates "brainrot" style videos based on stories from the AITA (Am I the A-hole?) subreddit. The pipeline includes the following steps:

1.  **Text Generation**: Fine-tunes a language model on AITA stories to generate new, engaging content.
2.  **Text-to-Speech**: Converts the generated text into speech.
3.  **Video Generation**: Combines the audio with a background video and adds subtitles.
4.  **Social Media Posting**: Uploads the final video to Instagram and YouTube.

## Project Structure

-   `data/`: Should contain the input data, such as `aitah_stories.json` and the background video `brain_number_trimmed.mp4`.
-   `notebooks/`: Contains the original Colab notebook.
-   `scripts/`: Contains the Python scripts for each step of the pipeline.
-   `output/`: Contains the generated audio, video, and subtitle files.
-   `requirements.txt`: A list of the Python packages required to run this project.

## Notes from the original notebook:

-   Use a GPU for training.
-   Remember to import the `aitah_stories.json` file.
-   Remember to import the `brain_number_trimmed.mp4` file.
-   Enter the API key when prompted.
