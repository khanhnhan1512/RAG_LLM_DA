class Process:
    def __init__(self, llm_instance, settings, original_data):
        """
        """
        self.llm_instance = llm_instance
        self.settings = settings
        self.original_data = original_data
        self.data = {}
    
    def get_data_path(self):
        """
        """
        task_result = ''
        print('1. Getting path to data...')

        for i, key in enumerate(self.settings):
            if 'output_vector_database' not in key:
                self.data[key] = {}
                self.data[key]['path'] = self.settings[key]
                task_result += f"- Path to data {key}: {self.data[key]['path']}.\n"
        print(task_result)
    
    def load_data(self):
        """
        """
        print('2. Loading data...')
        self.data, task_result = main_load_data(self.data, self.original_data, self.settings['output_vertor_database_build'])
        print(task_result)
        