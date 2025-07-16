import pandas as pd
import ast

class FileHandler:
    def read_file(self, file):
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
        return None

    def validate_columns(self, df, required_columns):
        return all(col in df.columns for col in required_columns)

    def parse_test_cases(self, test_cases_str):
        try:
            return ast.literal_eval(test_cases_str)
        except:
            return []
