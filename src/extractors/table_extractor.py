import camelot
import pandas as pd

class TableExtractor:
    def __init__(self):
        self.parser = camelot

    def extract(self, pdf_path):
        """Extract tables from PDF"""
        # Read tables from PDF
        tables = self.parser.read_pdf(pdf_path, pages='all')
        
        extracted_tables = []
        for table in tables:
            df = table.df
            # Convert table to structured format
            table_data = {
                'content': df.to_dict(),
                'page': table.page,
                'coords': table.coords,
                'accuracy': table.accuracy
            }
            extracted_tables.append(table_data)
        
        return extracted_tables 