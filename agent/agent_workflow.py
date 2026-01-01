from utils.model_loader import ModelLoader
from prompt_library.prompts import SYSTEM_PROMPT
from langgraph.graph import StateGraph, MessagesState, END, START



class build_graph():
    def __init__(self, model_provider: str = "groq"):
        self.model_provider = model_provider
        self.model_loader = ModelLoader(model_provider=self.model_provider)
        self.llm = self.model_loader.load_llm()

        self.graph = None
        self.system_prompt = SYSTEM_PROMPT

    def agent_function(self, state: MessagesState):
        """main function"""
        user_question = state["messages"]
        input_question = [self.system_prompt] + user_question
        response = self.llm.invoke(input_question)
        return {"messages": [response]}


    def build(self):
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("agent", self.agent_function)

        graph_builder.add_edge(START, "agent")
        graph_builder.add_edge("agent", END)

        self.graph = graph_builder.compile()
        return self.graph
    

    def __call__(self):
        return self.build()



