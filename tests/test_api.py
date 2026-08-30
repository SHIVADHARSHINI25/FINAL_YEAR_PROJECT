import pytest
from fastapi.testclient import TestClient
from api.main import app
import json
import os

client = TestClient(app)

def test_create_pipeline():
    # create dummy file
    with open("dummy.csv", "w") as f:
        f.write("age,income,label\n25,50000,0\n")
        
    with open("dummy.csv", "rb") as f:
        response = client.post(
            "/pipelines",
            files={"file": ("dummy.csv", f, "text/csv")},
            data={
                "algorithm": "random_forest",
                "target_column": "label",
                "metrics": json.dumps(["accuracy", "f1"])
            }
        )
        
    os.remove("dummy.csv")
    
    assert response.status_code == 200
    assert "run_id" in response.json()

def test_list_pipelines():
    response = client.get("/pipelines")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_knowledge():
    response = client.get("/knowledge")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
