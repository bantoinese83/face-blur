# Face Blur Application

This application uses AWS Rekognition to detect faces in a video stream and censor them using pixelation. The main components are face detection using AWS Rekognition and face censoring using OpenCV.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Tests](#tests)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/bantoinese83/face-blur.git
    cd face-blur
    ```

2. **Create and activate a virtual environment:**

    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Configure AWS credentials:**

   Ensure that your AWS credentials are configured properly. You can do this by setting up the `~/.aws/credentials` file or using environment variables.

## Usage

1. **Run the main application:**

    ```sh
    python main.py
    ```

   The application will start capturing video from your default camera, detect faces, and censor them in real-time. Press `q` to quit the application.

## Tests

1. **Run the unit tests:**

    ```sh
    python -m unittest discover -s tests
    ```

   This will discover and run all the unit tests in the `tests` directory.

## Configuration

The AWS Rekognition client is configured in the `aws_config.py` file. Ensure that your AWS credentials and region are properly set up.

Example `aws_config.py`:

```python
import boto3

def rekognition_client():
    return boto3.client('rekognition', region_name='your-region')
