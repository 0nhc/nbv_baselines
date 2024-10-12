import os
import pickle
import json

import numpy as np


class FileUtil:
    @staticmethod
    def get_path(file_name, target_dir=None):
        if target_dir is None:
            file_path = file_name
        else:
            file_path = os.path.join(target_dir, file_name)
        return file_path

    @staticmethod
    def load_pickle(file_name, target_dir=None):
        file_path = FileUtil.get_path(file_name, target_dir)
        with open(file_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def save_pickle(data, file_name, target_dir=None):
        file_path = FileUtil.get_path(file_name, target_dir)
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
            return True

    @staticmethod
    def load_json(file_name, target_dir=None):
        file_path = FileUtil.get_path(file_name, target_dir)
        with open(file_path, "r") as f:
            return json.load(f)

    @staticmethod
    def save_json(data, file_name, target_dir=None):
        file_path = FileUtil.get_path(file_name, target_dir)
        with open(file_path, "w") as f:
            json.dump(data, f)
            return True

    @staticmethod
    def save_np_txt(np_data, file_name, target_dir=None):
        if len(np_data.shape) > 2:
            raise ValueError("Only 2D arrays are supported.")
        file_path = FileUtil.get_path(file_name, target_dir)
        np.savetxt(file_path, np_data)

    @staticmethod
    def load_np_txt(file_name, target_dir=None, shuffle=False):
        file_path = FileUtil.get_path(file_name, target_dir)
        np_data = np.loadtxt(file_path)
        if shuffle:
            indices = np.arange(np_data.shape[0])
            np.random.shuffle(indices)
            np_data_shuffled = np_data[indices]
            return np_data_shuffled
        else:
            return np_data

    @staticmethod
    def find_object_models(path):
        obj_files = {}
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".obj"):
                    full_path = os.path.join(root, file)
                    modified_name = full_path.replace(path, "").replace(os.sep, "_").rstrip(".obj")
                    if modified_name.startswith("_"):
                        modified_name = modified_name[1:]
                    obj_files[modified_name] = full_path
        return obj_files


''' ------------ Debug ------------ '''
if __name__ == "__main__":
    arr2d = np.random.random((4, 3))
    print(arr2d)
    np.savetxt("test.txt", arr2d)
    loaded_arr2d = FileUtil.load_np_txt("test.txt")
    print()
    print(loaded_arr2d)
