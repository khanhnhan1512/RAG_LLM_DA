class Process:
    def __init__(self, llm_instance, settings, original_data):
        """
        """
        self.llm_instance = llm_instance
        self.settings = settings
        self.original_data = original_data
        self.data = {}
    
    def get_data_path(self):
