import os
import subprocess
import time

# cmd = "aws s3 --no-sign-request sync s3://open-images-dataset/challenge2018 test_challenge_2018"
# cmd = "aws s3 --no-sign-request sync s3://open-images-dataset/validation validation"
cmd = "ps -aux|grep sk49|awk '{sum +=$3}END{print sum}'"
# cmd = "ls"
p = subprocess.Popen(args=cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     close_fds=True)  # 创建用于shell的子进程
errorcode = 10000  # errorcode=0表示子进程正常退出
while errorcode != 0:
    p = subprocess.Popen(args=cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
    while p.poll() is None:  # 判断子进程是否terminated
        line = p.stdout.readline()
        line = line.strip()
        print('output:{}'.format(line))  # 实时输出子进程的print信息
    errorcode = p.wait()  # python主进程等待子进程结束，返回returncode，如果为0表示正常退出
    if errorcode == 0:
        errorcode = 10000
        time.sleep(1)
    else:
        print('errorcode', errorcode)
print("end")
