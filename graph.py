from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate


class RAGState(TypedDict):

    question: str

    context: str

    answer: str


def build_rag_graph(
    llm,
    prompt: ChatPromptTemplate
):

    from langgraph.graph import (
        StateGraph,
        START,
        END
    )

    def generate(
        state: RAGState
    ):

        chain = prompt | llm

        response = chain.invoke({

            "question":
            state["question"],

            "context":
            state["context"]
        })

        return {

            "answer":
            response.content
        }

    workflow = StateGraph(
        RAGState
    )

    workflow.add_node(
        "generate",
        generate
    )

    workflow.add_edge(
        START,
        "generate"
    )

    workflow.add_edge(
        "generate",
        END
    )

    return workflow.compile()