import os
def get_paths(dir):
    all_path = []
    for fRoot, fDirs, fFiles in os.walk(dir):
        for ffile in fFiles:
            if ffile.endswith('.jpg') or ffile.endswith('.jpeg'):
                full_path = os.path.join(fRoot, ffile).replace('/', os.sep)
                all_path.append(full_path)
    return all_path

def check_matches(mkpts0, mkpts1, mask0, mask1, b_ids):

    pass