import os
import shutil


def copy_files(source_path, dest_path):
	for file in os.listdir(source_path):
		file_name = os.path.join(source_path, file)
		if os.path.isfile(file_name):
			shutil.copy(file_name, dest_path)
