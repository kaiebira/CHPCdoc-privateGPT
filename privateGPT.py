#!/usr/bin/env python3
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import chromadb
import os
import argparse
import sys

model = os.environ.get("MODEL", "llama3")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-mpnet-base-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))
context_window = int(os.environ.get('CONTEXT_WINDOW', 4000))

from constants import CHROMA_SETTINGS

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    llm = Ollama(model=model, callbacks=callbacks)

    # Initialize conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=5
    )

    # Custom prompt template
    prompt_template = """
    Human: {question} If you cannot answer the question based on the context, simply state "Unfortunately, I cannot answer your question at this time.", and politely refer the user to CHPC documentation at https://www.chpc.utah.edu/documentation. Make sure to specifically provide that link for the user. Do not refer to the context such as: "Based on...," "According to...," etc. Be detailed, clear, and concise. Do not format your answers as Markdown. Quote code blocks verbatim for the user.
    
    Assistant:

    {context}

    From CHPC documentation:

    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Create the conversational chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

    # Interactive questions and answers
    try:
        while True:
            query = input("\nEnter a query (or 'exit()' to quit): ")
            if query.lower() in ["exit()", "quit", "q"]:
                print("\nCleaning up and exiting.")
                break
            if query.strip() == "":
                continue

            # Get the answer from the chain using invoke
            res = qa.invoke({"question": query})
            answer = res['answer']

            print(answer)
    except KeyboardInterrupt:
        print("\n\nInterrupt received.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("\nCleaning up and exiting.")
        # Add any cleanup code here if needed

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

if __name__ == "__main__":
    main()
