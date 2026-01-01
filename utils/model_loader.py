import os
from dotenv import load_dotenv
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field
from utils.common import read_yaml
from constant import config_path
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


load_dotenv(override=True)
class ConfigLoader:
    def __init__(self):
        print(f"Loading config from {config_path}...")
        self.config = read_yaml(config_path)
    
    def __getitem__(self, key):
        return self.config[key]

class ModelLoader(BaseModel):
    model_provider: Literal["groq", "openai"] = "groq"
    config: Optional[ConfigLoader] = Field(default=None, exclude=True)

    def model_post_init(self, __context: Any) -> None:
        # Jab ModelLoader banega, automatically config load hogi
        self.config = ConfigLoader()
    
    class Config:
        arbitrary_types_allowed = True
    
    def load_llm(self):
        """
        Load and return the LLM model.
        """
        print(f"LLM loading for provider: {self.model_provider}")
        
        if self.model_provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            # Debug: Check if key is loaded (sirf pehle 5 characters dikhayega security ke liye)
            if not api_key:
                raise ValueError("GROQ_API_KEY missing! Check your .env file.")
            
            model_name = self.config["llm"]["groq"]["model_name"]
            print(f"Using Groq Model: {model_name}")
            
            return ChatGroq(model=model_name, api_key=api_key)

        elif self.model_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY missing! Check your .env file.")
            
            # FIX: Hardcoded "o4-mini" ko hata kar config wala model use karen
            model_name = self.config["llm"]["openai"]["model_name"]
            print(f"Using OpenAI Model: {model_name}")
            
            return ChatOpenAI(model=model_name, api_key=api_key)
        
        else:
            raise ValueError(f"Unsupported provider: {self.model_provider}")