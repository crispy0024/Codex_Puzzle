from webapp.app import app


def test_home_page():
    client = app.test_client()
    resp = client.get('/')
    assert resp.status_code == 200
