# cpp Backend

## Overview

This service utilizes llama.cpp and whisper.cpp to provide an API compatible with OpenAI's endpoints. It enables you to run large language models and speech recognition models locally, offering features like text completion and audio transcription through familiar API calls.

## Setup instructions

1. Clone the repository

```bash
git clone https://github.com/EmanuelJr/cpp-backend.git
cd cpp-backend
```

2. Setup your environment

This script creates a virtual environment, figures out the build type, and installs the required dependencies.

```bash
./setup.sh
```

3. Set your API key _(optional)_

To protect your API, create an API key and set it as an environment variable:

```bash
export API_KEY=your-api-key-here
```

## Usage

### Running the server

1. Start the server

```bash
python3 run.py
```

By default, the server runs on [http://127.0.0.1:8000](http://127.0.0.1:8000).

You also can run to set host and port:

```bash
python3 run.py 127.0.0.1:8080
```

2. Verify the Server is Running

Visit http://127.0.0.1:8000/health in your browser or use curl:

```bash
curl http://127.0.0.1:8000/health
```

You should receive a response indicating the server is operational.

### API Endpoints

Feel free to inspect our Swagger at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Acknowledgments

- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for the library and the base server
- [whisper-cpp-python](https://github.com/carloscdias/whisper-cpp-python)
