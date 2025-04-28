# Project04_docchat: chat with your documents!

![test cases](https://github.com/ShawnWhyWander/Project04_docchat/workflows/tests/badge.svg)

This project allows you to interact with a document by asking spoken questions.
It supports PDF, JPG, HTML, and text files, transcribes your voice into text, and retrieves relevant answers using a retrieval-augmented generation (RAG) pipeline.

Example usage:

```
$ python docchat.py "https://en.wikipedia.org/wiki/World_Bank" 
🎤 Press Enter to start recording your question 
📝 You asked: What is this document about? 
💡 Answer: This document discusses the history, goals, and structure of the World Bank.

🎤 Press Enter to start recording your question 
📝 You asked: Who is the current CEO of the World Bank? 
💡 Answer: I'm sorry, the document does not specify the current CEO.

🎤 Press Enter to start recording your question 
📝 You asked: exit 
👋 Exiting. Goodbye!
```

![demo](Animation.gif)
