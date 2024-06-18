import logging
import time
import cv2
from rekognition import FaceDetector, censor_face, FaceDetectionError


def main():
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Initialize FaceDetector
        face_detector = FaceDetector()
    except FaceDetectionError as e:
        logger.error(f"Initialization error: {e}")
        return

    # Initialize the video capture object
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        logger.error("Error: Could not open video device.")
        return

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            if not ret:
                logger.error("Failed to capture image")
                break

            try:
                # Detect faces in the current frame
                faces = face_detector.detect_faces_in_frame(frame)
                logger.info(f"Detected faces: {faces}")
            except FaceDetectionError as e:
                logger.error(f"Face detection error: {e}")
                continue

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
        logger.info("Video capture interrupted by user.")

    finally:
        # Release the capture and destroy windows
        video_capture.release()
        cv2.destroyAllWindows()
        logger.info("Video capture released and windows destroyed.")


if __name__ == "__main__":
    main()
