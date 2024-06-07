import cv2
import base64
import requests
import datetime
from openai import OpenAI



def get_base64_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def interpret_pic():
    resp = requests.post('http://192.168.122.1:11434/api/generate', json={
        "model": "moondream:1.8b",
        "prompt": "Describe the person and their emotional state looking directly at you.",
        "stream": False,
        "images": [get_base64_image()]
    }).json()

    print(resp.content)
    return resp['response']

def interpret_image_description_ollama(text):
    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama',  # required, but unused
    )

    response = client.chat.completions.create(
        model="llama2",
        messages=[
            {"role": "system", "content": "Greet the person that is described in a funny way. The person is described as: " + text}
        ]
    )
    print(response.choices[0].message.content)

if __name__ == '__main__':
    print(interpret_pic())


