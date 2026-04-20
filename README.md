# Triadic Classroom Conversational Agent

A multi-agent spoken dialogue system for educational environments. The system features a Teacher Agent and a Student Agent that interact in real-time, allowing a human learner to overhear and participate in their conversation.

## Overview

This project implements a triadic classroom setting where:
- **Teacher Agent** delivers structured lessons
- **Student Agent** asks clarifying questions
- **Human Learner** participates by overhearing and contributing

![System Architecture](report/figures/system.png)

## Tech Stack

- **ASR**: Whisper (OpenAI)
- **TTS**: Chatterbox-Turbo (ResembleAI)
- **LLM**: Llama 3.2 (via Ollama)
- **Protocol**: TCP sockets with JSON events

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally with Llama 3.2
- Audio input/output devices

### Running the System

1. **Start the Turn-Taking Controller (TTC)**:
   ```bash
   cd src/server/turn-taking-controller
   uv run src/main.py --mode teacher_student
   ```

2. **Start the Teacher client** (in a new terminal):
   ```bash
   cd src/client/teacher
   uv run src/main.py
   ```

3. **Start the Student client** (in a new terminal):
   ```bash
   cd src/client/student
   uv run src/main.py
   ```

## Demo

<video src='https://rawcdn.githack.com/F21CA-Disembodied/coursework/a035d697be8ce5c5b7dc2bb1180f712682758827/demo.mp4' width=180/>
See `demo.mp4` for a video demonstration of the system in action.

## Report

The full technical report is available in the repository under `report/build/report.pdf` or at [GitHub Releases](https://github.com/F21CA-Disembodied/coursework-gonzalo/releases/tag/v1.0.0).
