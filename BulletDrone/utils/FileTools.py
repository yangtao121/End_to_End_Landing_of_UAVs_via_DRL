import os
from shutil import copy


def rename_itr(path):
    i = 0
    FileList = os.listdir(path)
    # print(FloatingPointError)
    for files in FileList:
        oldDirPath = os.path.join(path, files)
        if os.path.isdir(oldDirPath):
            rename_itr(oldDirPath)
        fileName = os.path.splitext(files)[0]
        fileType = os.path.splitext(files)[1]
        newDirPath = os.path.join(path, str(i) + fileType)
        os.rename(oldDirPath, newDirPath)
        i += 1


def copy_file(from_path, to_path, file_name, out_name=None):
    from_path_file = os.path.join(from_path, file_name)
    if out_name is None:
        to_path_file = to_path + '/' + file_name
    else:
        to_path_file = to_path + '/' + out_name
    if not os.path.exists(to_path):
        os.makedirs(to_path)
    copy(from_path_file, to_path_file)


if __name__ == "__main__":
    copy_file('', 'img', '1.png', '2.png')
    os.remove("1.png")
    # rename_itr('texture')
