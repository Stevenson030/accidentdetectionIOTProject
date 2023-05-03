import cv2
import requests
import json
import numpy as np
import random
from shapely.geometry import Polygon, Point
from iotc.aio import IoTCClient, IOTCConnectType
from azure.iot.device import IoTHubDeviceClient, Message
import asyncio

# Set up your Azure Custom Vision API endpoint and key
ENDPOINT = 'https://accidentdetection-prediction.cognitiveservices.azure.com'
KEY = '9240cab6db9c4e0a89e4a005bf283fb1'
# Set up the API URL and headers
API_URL = ENDPOINT + "/customvision/v3.0/Prediction/5e839385-ea56-4adb-bd93-726d64025126/classify/iterations/Iteration8/image"
headers = {'Content-Type': 'application/octet-stream', 'Prediction-Key': KEY}


# Set up your Azure IoT Central device connection string and model ID
CONNECTION_STRING = 'HostName=iotc-7887bc43-759a-4e29-b958-e06653501ac7.azure-devices.net;DeviceId=test2;SharedAccessKey=yvyops7VNnEKsHm50bT7QfPnSVc9igFCF7iuuhUWkJg='
scopeId = '0ne009D545B'
device_id = 'test2'
sasKey = 'yvyops7VNnEKsHm50bT7QfPnSVc9igFCF7iuuhUWkJg=' # or use device key directly
# Create the IoT Central device client
iotc = IoTCClient(device_id, scopeId, IOTCConnectType.IOTC_CONNECT_DEVICE_KEY, sasKey)

poly = Polygon([(37.75850848099701, -122.50833008408812), (61.69742149786521, 98.45210397852014),(-81.20166845789012, 52.45632145697852),(65.45987563201458, -21.45978563214058)])
def polygon_random_points (poly, num_points):
    min_x, min_y, max_x, max_y = poly.bounds
    points = []
    while len(points) < num_points:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            points.append(random_point)
    return points





#Send Telemetry Data
async def send_telemetry(accident, confidence,timestamp):
    points = polygon_random_points(poly,1) 
    for p in points:
        gps = str(p.x)+","+str(p.y)
    if accident == "Accident":
        acc = True
    else:
        acc = False
    await iotc.connect()
    await iotc.send_telemetry({"Accident": acc , "Confidence": confidence * 100, "GPS_data": gps, "EventTime": timestamp})
    await iotc.disconnect()


# Start capturing video from the default camera
#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(cv2.CAP_DSHOW)
cap = cv2.VideoCapture('http://10.106.173.165:8080/video')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    # Read the next frame from the camera
    ret, frame = cap.read()
    # Convert the frame to a byte array
    image_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
    #print(image_bytes)
    try:
        
        # Call the Custom Vision API with the captured image
        response = requests.post(API_URL, headers=headers, data=image_bytes)
        response.raise_for_status()
        # Get the prediction result from the API response
        result = json.loads(response.content.decode('utf-8'))
        print(result)
        prediction = result['predictions'][0]['tagName']
        probability = result['predictions'][0]['probability']
        timestamp = result['created']
        asyncio.run(send_telemetry(prediction, probability,timestamp))
        
        #Draw the prediction result on the captured image
        cv2.putText(frame, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        print(e)
    
    # Show the captured image with the prediction result
    cv2.imshow('Camera', frame)
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()

