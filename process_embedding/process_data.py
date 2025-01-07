from process_embedding.steps.main_load_data import main_load_data
from process_embedding.steps.prepare_vector_db_content import main_prepare_vector_db
from process_embedding.steps.create_vector_db import main_create_vector_db
from process_embedding.steps.add_attributes import add_attribute

class Process:
    def __init__(self, llm_instance, settings, original_data, data_loader):
        """
        """
        self.llm_instance = llm_instance
        self.settings = settings
        self.original_data = original_data
        self.data_loader = data_loader
        self.data = {}
    
    def get_data_path(self):
        """
        """
        task_result = ''
        print('1. Getting path to data...')

        for i, key in enumerate(self.settings):
            if key in ['facts', 'rules']:
                self.data[key] = {}
                self.data[key]['path'] = self.settings[key]
                task_result += f"- Path to data {key}: {self.data[key]['path']}.\n"
        print(task_result)
    
    def load_data(self):
        """
        """
        print('2. Loading data...')
        self.data, task_result = main_load_data(self.data, self.original_data, self.settings['output_vector_database_build'])
        print(task_result)
        
    def add_attributes(self):
        """
        """
        print("3. Adding attributes to data...")
        self.data, task_result = add_attribute(self.data, self.llm_instance, self.settings, self.data_loader)
        print(task_result)

    def prepare_vector_database_content(self):
        """
        """
        print("4. Preparing vector database content...")
        self.data, task_result = main_prepare_vector_db(self.data, self.llm_instance)
        print(task_result)

    def create_vector_database(self):
        """
        """
        print("5. Creating vector database...")
        self.data, task_result = main_create_vector_db(self.data, self.llm_instance, self.settings['output_vector_database_build'])
        print(task_result)

    def main(self):
        steps = [
            self.get_data_path,
            self.load_data,
            self.add_attributes,
            self.prepare_vector_database_content,
            self.create_vector_database
        ]
        for step in steps:
            step()
        print("Process completed.")

    