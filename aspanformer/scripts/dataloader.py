import os

class Dataloader:
    def __init__(self, dir_path, batch_size):
        self.dir_path = dir_path
        self.paths = self.im_paths()
        self.count = len(self.paths)
        self.index = 0
        self.batch_size = batch_size

    def im_paths(self):
        return os.listdir(self.dir_path)

    def __len__(self):
        return self.count

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.count:
            raise StopIteration

        next_index = int(round(self.index + self.batch_size / 2))

        if next_index > self.count:
            next_index = self.count

        next_paths = [self.dir_path + '/' + p for p in self.paths[self.index:next_index]]
        curr_index = list(range(self.index, next_index))
        self.index = next_index
        return curr_index, next_paths

    def overflow(self, index):
        #print('overflow index reset:', index)
        self.index = index