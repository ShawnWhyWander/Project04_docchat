# Project04_docchat: chat with your documents!

![test cases](https://github.com/ShawnWhyWander/Project04_docchat/workflows/tests/badge.svg)

This project allows you to interact with a document by asking spoken questions.
It supports PDF, JPG, HTML, and text files, transcribes your voice into text, and retrieves relevant answers using a retrieval-augmented generation (RAG) pipeline.

Example usage:

```
$ python docchat.py "https://en.wikipedia.org/wiki/World_Bank" 

>>
🧠 Voice Q&A System Starting...
✅ Document loaded.
🎤 Press Enter to start recording your question (or type 'exit' to quit).

🎙️ Recording... speak now
✅ Recording saved.
📝 You asked:  What is this document about?
💡 Answer:
 Based on the provided document, it appears to be about the World Bank Group, its mission, and its efforts to address climate change. The document discusses the World Bank's role in providing loans and assistance to developing countries, its mission to end extreme poverty and boost shared prosperity, and its concerns about climate change. It also mentions a World Bank report on climate change and the bank's increased aid for climate change adaptation.
🔊 Playing TTS response...
🎤 Press Enter to start recording your question (or type 'exit' to quit).

🎙️ Recording... speak now
✅ Recording saved.
📝 You asked:  Who is the current CEO of the world?
💡 Answer:
 The document does not mention the current CEO of the World Bank. It only discusses the tradition of having an American head the bank, implemented because the United States provides the majority of World Bank funding, and mentions that the president of the bank is responsible for chairing meetings of the boards of directors and for overall management of the bank.
🔊 Playing TTS response...
🎤 Press Enter to start recording your question (or type 'exit' to quit).
exit
Goodbye!
```

![demo](Animation.gif)
