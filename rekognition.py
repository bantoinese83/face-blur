import logging
import time

import cv2
from botocore.exceptions import ClientError

from aws_config import rekognition_client


class LabelDetectionError(Exception):
    pass


class FaceDetectionError(Exception):
    pass


class FaceDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rekognition = rekognition_client()
        if not self.rekognition:
            self.logger.error("Failed to initialize Rekognition client")
            raise FaceDetectionError("Failed to initialize Rekognition client")

    def detect_faces_in_frame(self, frame):
        try:
            _, jpeg_frame = cv2.imencode('.jpg', frame)
            response = self.rekognition.detect_faces(
                Image={
                    'Bytes': jpeg_frame.tobytes(),
                },
                Attributes=['ALL']
            )
            return response['FaceDetails']
        except ClientError as e:
            self.logger.error(f"Error detecting faces in frame: {e}")
            raise FaceDetectionError(f"Error detecting faces in frame: {e}")


def censor_face(frame, left, top, width, height, pixelation_level=10):
    """Pixelate the area of the face in the frame."""
    # Extract the face region
    face_region = frame[top:top + height, left:left + width]

    # Resize the face region to a smaller size
    small_face = cv2.resize(face_region, (pixelation_level, pixelation_level), interpolation=cv2.INTER_LINEAR)

    # Scale it back up to the original size
    pixelated_face = cv2.resize(small_face, (width, height), interpolation=cv2.INTER_NEAREST)

    # Replace the face region in the frame with the pixelated version
    frame[top:top + height, left:left + width] = pixelated_face

    return frame


def main():
    face_detector = FaceDetector()

    # Initialize the video capture object
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open video device.")
        return

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            if not ret:
                print("Failed to capture image")
                break

            # Detect faces in the current frame
            faces = face_detector.detect_faces_in_frame(frame)
            print(faces)

            # Censor faces in the current frame
            for face in faces:
                bbox = face['BoundingBox']
                left = int(bbox['Left'] * frame.shape[1])
                top = int(bbox['Top'] * frame.shape[0])
                width = int(bbox['Width'] * frame.shape[1])
                height = int(bbox['Height'] * frame.shape[0])

                # Censor the face
                frame = censor_face(frame, left, top, width, height)

            # Display the resulting frame
            cv2.imshow('Video', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Add a short delay to avoid hitting the Rekognition API too frequently
            time.sleep(1)

    except KeyboardInterrupt:
        print("Video capture interrupted by user.")

    finally:
        # Release the capture and destroy windows
        video_capture.release()
        cv2.destroyAllWindows()
        print("Video capture released and windows destroyed.")


if __name__ == "__main__":
    main()
