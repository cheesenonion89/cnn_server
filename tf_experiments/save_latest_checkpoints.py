import fnmatch
import os
from shutil import copyfile
from shutil import rmtree

path = '/home/markus/projects/cnn_server/model'

for subdir in os.listdir(path):
    if 'setting' not in subdir:
        continue
    bkp_dir_path = '%s_bkp' % os.path.join(path, subdir)
    os.makedirs(bkp_dir_path)
    for subsubdir in os.listdir(os.path.join(path, subdir)):
        if 'README' in subsubdir:
            continue
        ckpt_dir = os.path.join(path, subdir, subsubdir)
        bkp_bot_dir = os.path.join(bkp_dir_path, subsubdir)
        os.makedirs(bkp_bot_dir)

        with open(os.path.join(ckpt_dir, 'checkpoint'), 'r') as f:
            copyfile(os.path.join(ckpt_dir, 'checkpoint'), os.path.join(bkp_bot_dir, 'checkpoint'))
            ckpt_name = f.readline().split('/')[-1].replace('\"', '').replace('\n', '')
            for dirpath, dirnames, files in os.walk(ckpt_dir):
                for ckpt_file in fnmatch.filter(files, ckpt_name + '*'):
                    # copyfile(os.path.join(ckpt_dir, ckpt_file), os.path.join(bkp_bot_dir, ckpt_file))
                    print('')

    rmtree(os.path.join(path, subdir))

    # ckpt_name_stub = os.path.join(ckpt_dir, ckpt_name)
    # pattern = ckpt_name_stub

    # print(pattern)
    # ckpt_files = glob.glob(pattern)
    # print(ckpt_files)
