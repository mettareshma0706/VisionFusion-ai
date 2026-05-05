# VisionFusion Multi-Modal Insight

A FastAPI-based application that combines CLIP (Contrastive Language-Image Pre-Training) for image feature extraction with OpenAI's GPT models to provide intelligent insights from images.

## Features

- **Image Feature Extraction**: Uses OpenAI's CLIP model to extract rich visual features from images
- **AI-Powered Analysis**: Leverages GPT-4 to generate explanations and recommendations based on extracted features
- **REST API**: FastAPI-based endpoints for easy integration
- **Multi-Modal**: Combines visual and language understanding

## Project Structure

```
visionfusion/
├── main.py       # FastAPI application entry point
├── model.py      # CLIP model for image feature extraction
├── utils.py      # OpenAI integration for response generation
└── requirements.txt  # Python dependencies
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

## Usage

Start the API server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

- **GET /** - Health check
- **POST /analyze/** - Analyze image with prompt

### Example Request

```python
import requests

url = "http://localhost:8000/analyze/"
files = {"file": open("image.jpg", "rb")}
data = {"prompt": "What is in this image?"}

response = requests.post(url, files=files, data=data)
print(response.json())
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for faster inference)
- OpenAI API key

## License

MIT