<p align="center">

  <h1 align="center">AskMyPDF</h1>

  <p align="center">
    A PDF Question-Answering application that lets you upload any PDF and chat with its content using AI-powered retrieval and language models. Powered by Streamlit for an intuitive UI.
    <br/>
    <br/>
    <a href="https://github.com/Yousinator/AskMyPDF/issues">Report Bug</a>
    .
    <a href="https://github.com/Yousinator/AskMyPDF/issues">Request Feature</a>
  </p>
</p>
<p align="center">
  <a href="https://github.com/Yousinator/AskMyPDF">
<img src="https://img.shields.io/github/downloads/Yousinator/AskMyPDF/total"> <img src ="https://img.shields.io/github/contributors/Yousinator/AskMyPDF?color=dark-green"> <img src ="https://img.shields.io/github/forks/Yousinator/AskMyPDF?style=social"> <img src ="https://img.shields.io/github/stars/Yousinator/AskMyPDF?style=social"> <img src ="https://img.shields.io/github/license/Yousinator/AskMyPDF">
  </a>
</p>

## Table Of Contents

- [About the Project](#about-the-project)
- [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)

## About the Project

AskMyPDF enables you to:

- **Upload any PDF** and instantly chat with its content.
- **Conversational Q&A** powered by Anthropic Claude via LangChain.
- **Semantic search** using FAISS and Sentence Transformers.
- **Source highlighting**: See exactly where answers come from in your PDF.
- **Modern Streamlit UI** with chat history and source expansion.

It's designed for students, researchers, and professionals who want to interactively analyze and extract knowledge from PDF documents.

## Built With

This tech stack powers the dynamic chat interface and retrieval-augmented Q&A in the project. Streamlit and st-chat provide an intuitive chat UI, while LangChain, FAISS, and Sentence Transformers handle the AI and search logic.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-E04E39?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-00b16a?style=for-the-badge&logo=chainlink&logoColor=white" />
  <img src="https://img.shields.io/badge/FAISS-009688?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Sentence_Transformers-5.0.0-6A1B9A?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Anthropic-000000?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Poetry-8C52FF?style=for-the-badge&logo=python" />
</p>

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/) (recommended for managing dependencies)
- Anthropic API Key (for Claude)

### Installation

1. Clone the repo

```bash
   git clone https://github.com/Yousinator/AskMyPDF
   cd AskMyPDF
```

2. **Install Poetry (if you haven’t)**

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install Dependencies**

   ```bash
   poetry install
   ```

4. **Activate the Virtual Environment**

   ```bash
   poetry shell
   ```

5. **Set Environment Variables**

   - `ANTHROPIC_API_KEY` (required)
   - `CLAUDE_MODEL` (e.g., `claude-sonnet-4-20250514`)
   - `EMBED_MODEL_NAME` (e.g., `all-MiniLM-L6-v2`)

   You can set these in your shell or in a `.env` file.

6. **Run the App**

   ```bash
   poetry run streamlit run src/main.py
   ```

## Usage

- Launch the app using `poetry run streamlit run src/main.py`.
- Upload a PDF using the file uploader.
- Ask any question about the document in the chat box.
- View answers and expand to see the exact PDF source chunks.
- Chat history is maintained for your session.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

- If you have suggestions for improvements or new features, feel free to [open an issue](https://github.com/Yousinator/AskMyPDF/issues/new) to discuss it, or directly create a pull request after you edit the _README.md_ file with necessary changes.
- Please make sure you check your spelling and grammar.
- Create individual PR for each suggestion.
- Please also read through the [Code Of Conduct](https://github.com/Yousinator/AskMyPDF/blob/main/CODEOFCONDUCT.md) before posting your first idea as well.

### Creating A Pull Request

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the `Apache 2.0` License. See [LICENSE](https://github.com/Yousinator/AskMyPDF/blob/main/LICENSE) for more information.

## Authors

- **Yousinator** - [Yousinator](https://github.com/Yousinator/)

<div align="center">

**Made with ❤️ by Yousinator**

</div>