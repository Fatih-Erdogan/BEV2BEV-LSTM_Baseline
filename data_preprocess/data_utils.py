from PIL import Image
import json
import os


def resize_images(input_dir, target_size=(128, 128)):
    for filename in os.listdir(input_dir):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            file_path = os.path.join(input_dir, filename)
            with Image.open(file_path) as img:
                resized_img = img.resize(target_size)
                resized_img.save(file_path)


def process_and_overwrite_waypoints(input_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)

            waypoints = data["waypoints"]
            delta_waypoints = []

            for i in range(len(waypoints)-1, 0, -1):
                x_delta = round(waypoints[i][0] - waypoints[i-1][0], 6)
                y_delta = round(waypoints[i][1] - waypoints[i-1][1], 6)
                delta_waypoints.append([x_delta, y_delta])

            first_x_delta = round(waypoints[0][0] - data["x"], 6)
            first_y_delta = round(waypoints[0][1] - data["y"], 6)
            delta_waypoints.append([first_x_delta, first_y_delta])

            delta_waypoints.reverse()

            with open(file_path, 'w') as file:
                json.dump({"waypoints": delta_waypoints}, file, indent=4)

