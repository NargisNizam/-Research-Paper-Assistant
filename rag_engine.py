import os
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber

from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.documents import Document

from langchain_core.prompts import ChatPromptTemplate

from langchain_groq import ChatGroq

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from graph import build_rag_graph


class RAGEngine:

    def __init__(self):

        self.documents: List[Document] = []

        self.document_names = set()

        self.vectorizer = None

        self.matrix = None

        self.texts = []

        self.chunks = []

        self.splitter = RecursiveCharacterTextSplitter(

            chunk_size=900,

            chunk_overlap=150,

            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

        api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not api_key:

            raise RuntimeError(
                "GROQ_API_KEY is missing. "
                "Please add it to your .env file."
            )

        self.llm = ChatGroq(

            api_key=api_key,

            model=os.getenv(
                "GROQ_MODEL",
                "llama-3.3-70b-versatile"
            ),

            temperature=0
        )

        prompt = ChatPromptTemplate.from_messages([

            (
                "system",

                """
You are a Research Paper Assistant.

Your job is to answer questions ONLY
using the supplied context from the
uploaded research papers.

STRICT RULES:

1. Do not use outside knowledge.
2. Do not browse the internet.
3. Do not make assumptions.
4. Do not invent information.
5. If the answer is not present in
   the supplied context, respond:

"The answer is not available in
the uploaded documents."

6. If multiple documents are used,
clearly distinguish their information.

Context:

{context}
"""
            ),

            (
                "human",
                "{question}"
            )

        ])

        self.graph = build_rag_graph(
            self.llm,
            prompt
        )

    @property
    def chunk_count(self):

        return len(self.chunks)

    def _extract_pages(
        self,
        pdf_path: Path
    ):

        pages = []

        reader = PdfReader(
            str(pdf_path)
        )

        plumber_pages = []

        try:

            with pdfplumber.open(
                str(pdf_path)
            ) as pdf:

                plumber_pages = pdf.pages

        except Exception:

            plumber_pages = []

        for index, page in enumerate(
            reader.pages
        ):

            text = page.extract_text() or ""

            if (
                len(text.strip()) < 30
                and
                index < len(plumber_pages)
            ):

                try:

                    plumber_text = (
                        plumber_pages[index]
                        .extract_text()
                        or ""
                    )

                    if len(
                        plumber_text.strip()
                    ) > len(text.strip()):

                        text = plumber_text

                except Exception:

                    pass

            pages.append(
                text.strip()
            )

        return pages

    def add_pdf(
        self,
        pdf_path: Path
    ):

        pages = self._extract_pages(
            pdf_path
        )

        page_documents = []

        for page_number, text in enumerate(
            pages,
            start=1
        ):

            if not text.strip():
                continue

            page_documents.append(

                Document(

                    page_content=text,

                    metadata={

                        "source":
                        pdf_path.name,

                        "page":
                        page_number
                    }
                )
            )

        if not page_documents:

            raise ValueError(
                "No readable text was found "
                "in this PDF."
            )

        # Remove old chunks if same
        # document is uploaded again.

        self.chunks = [

            chunk

            for chunk in self.chunks

            if chunk.metadata.get(
                "source"
            ) != pdf_path.name
        ]

        new_chunks = (
            self.splitter
            .split_documents(
                page_documents
            )
        )

        self.chunks.extend(
            new_chunks
        )

        self.document_names.add(
            pdf_path.name
        )

        self._rebuild_index()

        return {

            "name":
            pdf_path.name,

            "pages_read":
            len(page_documents),

            "chunks_added":
            len(new_chunks)
        }

    def _rebuild_index(self):

        self.texts = [

            document.page_content

            for document in self.chunks
        ]

        if not self.texts:

            self.vectorizer = None
            self.matrix = None

            return

        self.vectorizer = (
            TfidfVectorizer(

                lowercase=True,

                stop_words="english",

                ngram_range=(1, 2),

                max_features=50000
            )
        )

        self.matrix = (
            self.vectorizer
            .fit_transform(
                self.texts
            )
        )

    def retrieve(
        self,
        question: str,
        top_k: int = 5
    ):

        if (
            self.vectorizer is None
            or
            self.matrix is None
        ):

            return []

        query_vector = (
            self.vectorizer
            .transform(
                [question]
            )
        )

        scores = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

        ranked_indices = (
            scores.argsort()[::-1]
        )

        results = []

        for index in ranked_indices[
            :top_k
        ]:

            if scores[index] < 0.08:
                continue

            original = self.chunks[
                index
            ]

            document = Document(

                page_content=
                original.page_content,

                metadata={

                    **original.metadata,

                    "score":
                    float(
                        scores[index]
                    )
                }
            )

            results.append(
                document
            )

        return results

    def ask(
        self,
        question: str
    ):

        retrieved = self.retrieve(
            question
        )

        if not retrieved:

            return {

                "answer":
                "The answer is not available "
                "in the uploaded documents.",

                "sources": []
            }

        context_parts = []

        for index, document in enumerate(
            retrieved,
            start=1
        ):

            context_parts.append(

                f"""
[Source {index}
Document: {document.metadata.get("source")}
Page: {document.metadata.get("page")}]

{document.page_content}
"""
            )

        context = "\n\n".join(
            context_parts
        )

        result = self.graph.invoke({

            "question":
            question,

            "context":
            context
        })

        sources = []

        seen = set()

        for document in retrieved:

            source = (
                document.metadata.get(
                    "source"
                )
            )

            page = (
                document.metadata.get(
                    "page"
                )
            )

            key = (
                source,
                page
            )

            if key not in seen:

                sources.append({

                    "document":
                    source,

                    "page":
                    page,

                    "score":
                    round(
                        document.metadata.get(
                            "score",
                            0
                        ),
                        3
                    )
                })

                seen.add(key)

        return {

            "answer":
            result["answer"],

            "sources":
            sources
        }

    def clear(self):

        self.documents = []

        self.document_names = set()

        self.vectorizer = None

        self.matrix = None

        self.texts = []

        self.chunks = []