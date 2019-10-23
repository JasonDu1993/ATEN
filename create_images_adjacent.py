import os
import shutil
import time

t0 = time.time()
DATA_DIR = r"D:\dataset\VIP_tiny"
IMAGE_ADJACENT = r"D:\dataset\VIP_tiny\lists\image_adjacent_id.txt"
f = open(IMAGE_ADJACENT, "w")
for video_name in os.listdir(os.path.join(DATA_DIR, "adjacent_frames")):
    for image_id in os.listdir(os.path.join(DATA_DIR, "adjacent_frames", video_name)):
        save_dir = os.path.join(DATA_DIR, "ImagesAdjacent", video_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for image_name in os.listdir(os.path.join(DATA_DIR, "adjacent_frames", video_name, image_id)):
            print(image_name)
            path = os.path.join(DATA_DIR, "adjacent_frames", video_name, image_id, image_name)
            print(path)
            save_path = os.path.join(save_dir, image_name)
            print(save_path)
            shutil.copy(path, save_path)
            f.write(video_name + "/" + image_name.split(".")[0] + "\n")
print("total time", time.time() - t0, "s")
