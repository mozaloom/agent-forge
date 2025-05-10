# AI Agents Course Repository

This repository contains the code and resources from the AI Agents Course, focusing on various agent frameworks and implementations including LangGraph, LlamaIndex, and SmolagentS.

## Overview

This repository serves as a collection of hands-on examples, implementations, and demonstrations of AI agent architectures explored during the Hugging Face AI Agents Course. It showcases different approaches to building autonomous AI systems using popular frameworks.

## Repository Structure

```
├── Core Agent Implementations
│   ├── simpleAgent.py           # Basic agent implementation
│   ├── multiAgent.py            # Multi-agent system
│   ├── codeAgents.py            # Agents for code-related tasks
│   ├── RAG.py                   # Retrieval Augmented Generation implementation
│   ├── documentAnalysis.py      # Document analysis capabilities
│   ├── visionBrowser.py         # Agent with web browsing and vision capabilities
│   └── mailSorting.py           # Email categorization agent

├── Framework-specific Implementations
│   ├── langGraphDemo.py         # LangGraph demonstration
│   ├── event-rag-agent/         # Event-driven RAG agents
│   │   ├── langgraph/           # LangGraph implementation
│   │   ├── llama-index/         # LlamaIndex implementation
│   │   └── smolagents/          # SmolagentS implementation

├── User Interface
│   └── Gradio_UI.py             # Gradio-based user interface

├── Tools & Utilities
│   ├── tools/                   # Agent tools
│   │   ├── final_answer.py      # Tool for providing final responses
│   │   ├── tempLib.py           # Template library
│   │   ├── visit_webpage.py     # Web browsing tool
│   │   └── web_search.py        # Web search tool
│   └── prompts.yaml             # Prompt templates

├── Notebooks
│   ├── LangGraph/               # LangGraph tutorials
│   ├── llama-Index/             # LlamaIndex tutorials
│   └── smolagents/              # SmolagentS tutorials
```

## Key Features

- **Multiple Agent Frameworks**: Implementations using LangGraph, LlamaIndex, and SmolagentS
- **RAG (Retrieval Augmented Generation)**: Enhanced agents with document retrieval capabilities
- **Multi-Agent Systems**: Collaborative agent architectures
- **Code Agents**: Specialized agents for code understanding and generation
- **Vision & Browsing**: Agents that can see and interact with web content
- **User Interface**: Gradio-based interface for easy interaction

## Prerequisites

To run the code in this repository, you need:

- Python 3.12+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mozaloom/agent-forge.git
   cd ai-agents-course
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Agent

```python
from simpleAgent import SimpleAgent

agent = SimpleAgent()
response = agent.run("What is the capital of France?")
print(response)
```

### RAG Agent

```python
from RAG import RAGAgent

agent = RAGAgent(documents_path="./your_documents/")
response = agent.query("Summarize the main points in these documents.")
print(response)
```

### Using the Gradio UI

```bash
python Gradio_UI.py
```

Then open your browser and navigate to the URL shown in the terminal (typically http://127.0.0.1:7860).

## Notebooks

The `notebooks/` directory contains Jupyter notebooks demonstrating various agent functionalities:

- **LangGraph**: Workflow-based agents
- **LlamaIndex**: Document-based agents and components
- **SmolagentS**: Various agent implementations including:
  - RAG agents
  - Code agents
  - Multi-agent systems
  - Vision agents

## Tools

The repository includes several tools that agents can use:

- `final_answer.py`: Tool for providing structured final responses
- `visit_webpage.py`: Web browsing capability
- `web_search.py`: Search engine queries

## Course Notes

For a comprehensive summary of the AI Agents Course, visit:
[AI Agents Course Notes](https://burnt-toothpaste-b35.notion.site/Agents-Course-Hugging-Face-1d5b72b9ff508025abf4c4fcc2047615?pvs=4)

## License

This project is licensed under the terms of the LICENSE file included in this repository.

## Contributions

Contributions are welcome! Feel free to submit issues or pull requests if you have improvements or bug fixes to suggest.