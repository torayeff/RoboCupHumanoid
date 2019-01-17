"""
Helper functions.
"""
import os
import glob
import shutil
import numpy as np
import scipy.stats
import xml.etree.ElementTree
from skimage import io
import torch
from torch.utils.data import Dataset


def get_teacher_signal(height, width, bndboxes, sigma=4, downsample=4):
    """Creates teacher signal for the image.

    Args:
        height: Height of the image.
        width: Width of the image.
        bndboxes: Coordinates of bounding boxes around objects as a list of tuples.
        sigma: Standard deviation for Gaussian.
        downsample: Downsampling ratio.

    Returns:
        2D feature map of image.
    """

    signal = np.zeros((int(height) // downsample, int(width) // downsample))

    for box in bndboxes:
        xmin = int(box[0]) // downsample
        ymin = int(box[1]) // downsample
        xmax = int(box[2]) // downsample
        ymax = int(box[3]) // downsample

        c_x = xmin + (xmax - xmin) // 2
        c_y = ymin + (ymax - ymin) // 2

        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                signal[y, x] = scipy.stats.multivariate_normal.pdf([y, x], [c_y, c_x], [sigma, sigma])

    return signal


def get_csv_lines(filename, labels):
    """Creates CSV lines from PASCAL VOC formatted XML file."""
    tree = xml.etree.ElementTree.parse(filename)

    root = tree.getroot()

    size = root.find('size')
    width = size.find('width').text
    height = size.find('height').text
    fn = filename.split("/")[-1].split(".")[0] + ".jpg"
    filedata = [fn, width, height]

    lines = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text

        if label in labels:
            objdata = [label, xmin, ymin, xmax, ymax]
            line = filedata + objdata
            lines.append(line)

    return lines


def create_csv_folder(xml_folder, csv_folder, labels, sep=','):
    """Creates CSV folder from XML PASCAL VOC folder."""

    strdata = ""
    for fn in glob.glob(xml_folder + "*.xml"):
        lines = get_csv_lines(fn, labels)

        for line in lines:
            strdata += sep.join(line) + "\n"

            # copy to csv_folder
            src = xml_folder + line[0]
            dst = csv_folder + line[0]
            shutil.copy(src, dst)

    with open(csv_folder + "data.csv", "w") as f:
        f.write("image_file,width,height,label,xmin,ymin,xmax,ymax\n")
        f.write(strdata.strip())


class SoccerBallDataset(Dataset):
    """Soccer Balls dataset."""

    def __init__(self, csv_file, root_dir, transform=None, sigma=4, downsample=4):
        """
        Args:
            csv_file: Path to csv file.
            root_dir: Directory with all images.
            transform: Optional transform to be applied.
        """

        self.sigma = sigma
        self.downsample = downsample

        self.dset = {}
        with open(csv_file, "r") as f:
            for line in f.readlines()[1:]:
                data = line.strip().split(",")
                filename = data[0]
                if filename not in self.dset.keys():
                    self.dset[filename] = []

                self.dset[filename].append(data[1:])

        self.filenames = list(self.dset.keys())
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """Fetches image with corresponding teacher signal."""

        image_name = self.filenames[idx]
        image_path = os.path.join(self.root_dir, image_name)
        image = io.imread(image_path).transpose(2, 0, 1)
        image = image / 255.
        image = torch.from_numpy(image)

        width = self.dset[image_name][0][0]
        height = self.dset[image_name][0][1]
        bndboxes = []

        for obj in self.dset[image_name]:
            bndboxes.append(obj[3:])

        # transform only images
        if self.transform:
            image = self.transform(image)

        signal = get_teacher_signal(height, width, bndboxes, self.sigma, self.downsample)

        sample = {'image': image, 'signal': signal, 'img_name': image_name, 'boxes': bndboxes}

        return sample

