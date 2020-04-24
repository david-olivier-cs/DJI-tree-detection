'''
Entry point to apply trunk recognition algo to a target video
The input video has N frames and a refresh rate R (in frames) is defined
The recognition algo will process every R frames.
Between every processed frame the last procesed frame will be maintained
'''

import os
import os.path

import cv2 as cv
from peeptree.processing import ImageProcessor

# defining necessary paths
output_video_name = "output.mp4"
trained_clf_path = "classifier.pickle"
input_video_name = "drone_capture_2.mp4"
video_folder = "/home/one_wizard_boi/Documents/Projects/DJI-tree-detection/Docs/"

# defining detection refresh variables
frame_counter = 6
detection_refresh = 5
latest_frame = None

if __name__ == "__main__":

    # defining the image processor
    processor = ImageProcessor(trained_clf_path, block_size=20)

    # opening target video
    input_video_path = os.path.join(video_folder, input_video_name)
    input_video = cv.VideoCapture(input_video_path)
    if not input_video.isOpened():
        raise ValueError("Failed to load target video")

    # defining output video writter
    output_video_path = os.path.join(video_folder, output_video_name)
    output_fps = input_video.get(cv.CAP_PROP_FPS)
    video_writter = cv.VideoWriter(output_video_path, cv.VideoWriter_fourcc(*'mp4v'),
                                  output_fps, (processor.resized_width, processor.resized_height))

    # going through all the frames of the video
    while(input_video.isOpened()):

        # fetching next video frame
        is_frame, frame = input_video.read()
        if not is_frame: break

        # managing refresh counter
        frame_counter += 1
        if frame_counter >= detection_refresh:
            frame_counter = 0

        # applying recognition
        if frame_counter == 0:
            try : latest_frame = processor.detect_object_segments(frame)
            except:
                raise ValueError("Failed to process video frame")

        # writing the latest frame to the output video
        if latest_frame is not None:
            video_writter.write(latest_frame)
                
    # releasing resources
    input_video.release()
    video_writter.release()