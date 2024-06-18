import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from botocore.exceptions import ClientError

from rekognition import FaceDetector, censor_face, FaceDetectionError


class TestFaceDetector(unittest.TestCase):
    @patch('rekognition.rekognition_client', return_value=MagicMock())
    def setUp(self, mock_rekognition_client):
        self.face_detector = FaceDetector()
        self.mock_rekognition_client = self.face_detector.rekognition
        self.mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_detect_faces_in_frame(self):
        self.face_detector.detect_faces_in_frame(self.mock_frame)
        self.mock_rekognition_client.detect_faces.assert_called_once()

    def test_detect_faces_in_frame_no_faces(self):
        self.mock_rekognition_client.detect_faces.side_effect = [{'FaceDetails': []}]
        faces = self.face_detector.detect_faces_in_frame(self.mock_frame)
        self.assertEqual(faces, [])

    def test_detect_faces_in_frame_error(self):
        self.mock_rekognition_client.detect_faces.side_effect = ClientError({}, 'DetectFaces')
        with self.assertRaises(FaceDetectionError):
            self.face_detector.detect_faces_in_frame(self.mock_frame)


class TestCensorFace(unittest.TestCase):
    def setUp(self):
        self.mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_censor_face_different_pixelation_levels(self):
        left, top, width, height = 10, 10, 50, 50
        for pixelation_level in range(1, 11):
            result = censor_face(self.mock_frame, left, top, width, height, pixelation_level)
            self.assertIsNotNone(result)

    def test_censor_face_different_face_sizes_and_positions(self):
        pixelation_level = 10
        for left, top, width, height in [(10, 10, 50, 50), (20, 20, 30, 30), (30, 30, 20, 20)]:
            result = censor_face(self.mock_frame, left, top, width, height, pixelation_level)
            self.assertIsNotNone(result)
