from pathlib import Path

import cv2
import pyarrow.dataset as ds
import rootutils

rootutils.setup_root(__file__, ".project-root", pythonpath=True, dotenv=True)

from lib.utils.image import create_collage, create_video_from_images


def process_video(
    orbbe_timestamps,
    input_dirs,
    output_path,
    skip_n_frames=2,
    fps=6,
):
    frames = []
    for index, row in orbbec_timestamps.iterrows():
        if index % skip_n_frames != 0:
            continue
        frame_num = int(row["frame_number"])
        collage = []
        for camera in input_dirs.values():
            multi_view_frames = camera.glob(f"color_{frame_num:06d}_camera*.jpg")
            multi_view_frames = sorted(multi_view_frames)
            images = [cv2.imread(str(frame)) for frame in multi_view_frames]
            # resize images to height 1080 if they are not already
            for i, image in enumerate(images):
                if image.shape[0] != 1080:
                    images[i] = cv2.resize(image, (1080, 1080))
            collage.extend(images)

        collage = create_collage(collage, format=[3, 4], downscale_factor=2, frame_index=frame_num)
        frames.append(collage)

    create_video_from_images(frames, fps, output_path)


if __name__ == "__main__":
    recording_name = "20250206_Testing"
    cameras = ["Aria", "Marshall", "Orbbec"]

    base_input_dir = Path("data", "recordings", recording_name)
    output_dir = base_input_dir / "visualization" / "color"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dirs = {
        "Aria": base_input_dir / "Aria" / "export" / "color",
        "Marshall": base_input_dir / "Marshall" / "recording" / "export",
        "Orbbec": base_input_dir / "Orbbec" / "color",
    }

    orbbec_timestamp_file = base_input_dir / "tables_timestamps.arrow"
    orbbec_timestamps = (
        ds.dataset(orbbec_timestamp_file, format="arrow").to_table().to_pandas()
    )

    process_video(
        orbbe_timestamps=orbbec_timestamps,
        input_dirs=input_dirs,
        output_path=output_dir / f"{'_'.join(cameras)}.mp4",
        skip_n_frames=2,
        fps=6,
    )
