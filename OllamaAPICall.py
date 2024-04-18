from src.utils import GenericAPICall
import requests

class OllamaAPICall(GenericAPICall):
    
    def __init__(self, address) -> None:
        super().__init__(address)
        self.endpoint = f"{address}/api/generate/"
        
    def run(self, prompt):
        response = requests.post(self.endpoint, 
                                 json = {"model": "nous-hermes2-mixtral:latest", 
                                         "prompt": prompt,
                                         "options": {
                                             "temperature": 0 , 
                                             "num_predict": 200, 
                                             "num_gpu":33}, 
                                         "stream":False})
        model_out = response.json()['response']
        return model_out