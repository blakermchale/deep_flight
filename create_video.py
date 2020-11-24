import cv2
import os
from pathlib import Path

# Read in airsim recording data from test
import glob
import os

airsim_dir = Path.home().joinpath("Documents").joinpath("AirSim")  # .joinpath("2020-11-23-23-48-57")
list_of_files = glob.glob(str(airsim_dir.joinpath('*')) + '/') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)

image_folder = Path().joinpath(latest_file).joinpath("images")
video_name = 'videos\\' + latest_file.split('\\')[-2] + '.avi'


images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 20., (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

print(f'Placed video in {video_name}')