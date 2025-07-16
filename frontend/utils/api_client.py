import requests

class APIClient:
    def __init__(self):
        self.base_url = "http://localhost:8000"

    def login(self, username, password):
        response = requests.post(f"{self.base_url}/auth/login", json={"username": username, "password": password})
        return response.json()

    def get_projects(self, token):
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{self.base_url}/projects", headers=headers)
        return response.json()

    def get_project(self, project_id, token):
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{self.base_url}/projects/{project_id}", headers=headers)
        return response.json()
