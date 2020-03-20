# singularity shell - B /work/aashfaq /projects/singularity/images/
import subprocess
import os
start_point = "s01"
command = "/usr/local/bin/FeatureExtraction -f {file} -out_dir /work/aashfaq/datasets/deap/openFaceFeatures/processed/"
# command = "echo '{file}'; sleep 5"
video_features_path = "/work/aashfaq/datasets/deap/videos"
commands_to_execute = []
for root, subdirs, files in os.walk(video_features_path):
    # print(sorted(subdirs))
    for file in files:
        folder = root.split("/")[-1]
        if folder == start_point:
            filepath = root + "/" + file
            one_command = command.format(file = filepath)
            commands_to_execute.append(one_command)


print(commands_to_execute)


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]



for command_chunk in chunks(commands_to_execute, 5):
    processes = []
    for one_command in command_chunk:
        p = subprocess.Popen(one_command.split())
        processes.append(p)
    [p.communicate() for p in processes]
    print("done 1 iteration")
    # breaks