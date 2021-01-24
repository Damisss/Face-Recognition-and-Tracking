import os

def DataLoader (path):
    try:
        imageFolders = [os.path.join(path, dir) for dir in os.listdir(path)]
        data = []
        for folder in imageFolders:
            imagePath = [os.path.join(folder, image) for image in os.listdir(folder)]
            for imagePath in imagePath:
                data.append(imagePath)

        return data

    except Exception as e:
        raise e
