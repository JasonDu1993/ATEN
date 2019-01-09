import shutil
import os

dir_dataset = "/home/sk49/workspace/dataset/VIP/"
dir_src = "/home/sk49/workspace/dataset/VIP/adjacent_frames"
videos = os.listdir(dir_src)
videos.sort()
for video in videos:
    video_path = os.path.join(dir_src, video)
    v_sequence = os.listdir(video_path)
    v_sequence.sort()
    for s in v_sequence:
        video_s_path = os.path.join(video_path, s)
        imgs = os.listdir(video_s_path)
        imgs.sort()
        for img in imgs:
            video_img_path = os.path.join(video_s_path, img)
            print(video_img_path)
            dst_path = os.path.join(dir_dataset, "videos", video, img)  # .jpg为你的文件类型，即后缀名，读者自行修改
            print(dst_path)
            shutil.copyfile(video_img_path, dst_path)
