import ffmpeg

# Path to your video file
video_path = "/Users/tonywang/Documents/hiwi/Aries/Code/Aries/data/recordings/20250206_Testing/Marshall/recording/videos/marshall_1_ch1_0065_0001.mp4"

# Extract metadata, including timecode
metadata = ffmpeg.probe(video_path)

# Locate the timecode in the metadata
timecode = None
for stream in metadata["streams"]:
    if "tags" in stream and "timecode" in stream["tags"]:
        timecode = stream["tags"]["timecode"]
        break

if timecode:
    print(f"SMPTE Timecode: {timecode}")
else:
    print("No SMPTE timecode found in metadata.")
