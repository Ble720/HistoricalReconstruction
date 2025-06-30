import os
import argparse

def rename_jpeg_to_jpg(directory):
    """Renames all .jpeg files to .jpg in the given directory."""
    for filename in os.listdir(directory):
        if filename.lower().endswith(".jpeg"):
            src = os.path.join(directory, filename)
            dst = os.path.join(directory, filename[:-5] + ".jpg")
            os.rename(src, dst)
            print(f"Renamed '{filename}' to '{filename[:-5] + '.jpg'}'")


# Call the function to rename the files
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='path to source images')
    args = parser.parse_args()
    rename_jpeg_to_jpg(args.path)