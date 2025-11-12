class PromptManager:
    
    def render(self, **kwargs):
        file:str = kwargs.pop("file")
        
        with open(file, 'r', encoding='utf-8') as f:
            self.template = f.read()
            
        return self.template.format(**kwargs)
