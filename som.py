import tkinter.filedialog
import secrets
import os
import sys
from PIL import Image
from numba import jit, njit
import numpy as np
import matplotlib.pyplot as plt

PROGRAM_DIR = os.path.dirname(os.path.abspath(__file__))

# Image file extension
IMG_EXT = [".jpg", ".JPG"]
# For one pixel
RGB_DICT = {"red": 0, "green": 1, "blue": 2}


# A3 (300dpi): 4,093 x 5,787 px
# RESIZE_HEIGHT = 4093
# RESIZE_WIDTH = 5787

# A3 (100dpi): 1,169 x 1,654 px
RESIZE_HEIGHT = 1169
RESIZE_WIDTH = 1654


# Program for extended multiple exposure, which overlays multiple images into one image
class StackOfMemories(object):
	def __init__(self):
		self.path = tkinter.filedialog.askdirectory(initialdir=PROGRAM_DIR)
		print("Parent Directory: " + self.path)
		self.subdir_list = []
		self.all_files = []

	# Get random "count" numbers of ["start", "end")
	# "dup": Whether to allow duplication
	@staticmethod
	def get_rand(start, end, count, dup=False):
		res = []
		while not len(res) == count:
			buf = secrets.randbelow(end - start) + start
			if dup or ((not dup) and res.count(buf) == 0):
				res.append(buf)
		return res

	# Get all subdirectories
	def get_subdir(self):
		dir_and_file_list = os.listdir(self.path)
		dir_list = [f for f in dir_and_file_list if not os.path.isfile(os.path.join(self.path, f))]
		self.subdir_list = dir_list

	# Get specified percentage of image files from "subdir_list" respectively
	# "file_percentage": Percentage of images acquired from each directory
	def get_files(self, file_percentage):
		for buf_path in self.subdir_list:
			files_list = os.listdir(os.path.join(self.path, buf_path))
			img_list = [e for e in files_list if e[-4:] in IMG_EXT]

			# If image files are not found
			if len(img_list) == 0:
				print("Image files are not found in ", os.path.join(self.path, buf_path))
				continue

			# "file_percentage" % of all files in the directory
			rnd = self.get_rand(0, len(img_list), len(img_list) // (100/file_percentage))
			self.all_files.extend([os.path.join(self.path, buf_path, img_list[e]) for e in rnd])

		with open(self.path + "\\files.txt", mode="w", encoding="UTF-8") as f:
			# Separate by \n and write to "files.txt"
			f.write("\n".join(self.all_files))

	# Process each of image files
	@jit(parallel=True)
	def process_each(self):
		rgb = [np.zeros((RESIZE_HEIGHT, RESIZE_WIDTH, 256), dtype="uint32"),
			np.zeros((RESIZE_HEIGHT, RESIZE_WIDTH, 256), dtype="uint32"),
			np.zeros((RESIZE_HEIGHT, RESIZE_WIDTH, 256), dtype="uint32")]

		length = len(self.all_files)
		print("File Length: ", length)

		if length == 0:
			print("Image files are not found")
			sys.exit(1)

		for e in self.all_files:
			print("Processing... ", e, "-", round((self.all_files.index(e) + 1) * 100 / length, 1), "%")
			img_arr = self._change_size(e)
			rgb = self._sum_rgb(img_arr, rgb)

		self._save_rgbarr(rgb)
		print("Resulting data are saved!")

	# Resize the image and return an array of the image
	@staticmethod
	def _change_size(file):
		with Image.open(file) as img:
			img_width, img_height = img.size

			# e.g. 3:2
			if img_width / img_height > 1.41:
				# Resize height
				resize_width = RESIZE_HEIGHT / img_height * img_width

				img = img.resize((int(resize_width), RESIZE_HEIGHT))

				# Resize width
				img_width, img_height = img.size
				img = img.crop(((img_width - RESIZE_WIDTH) // 2, 0, (img_width + RESIZE_WIDTH) // 2, RESIZE_HEIGHT))

			# e.g. 4:3
			else:
				# Resize width
				resize_heigth = RESIZE_WIDTH / img_width * img_height
				img = img.resize((RESIZE_WIDTH, int(resize_heigth)))

				# Resize heigth
				img_width, img_height = img.size
				img = img.crop((0, (img_height - RESIZE_HEIGHT) // 2, RESIZE_WIDTH, (img_height + RESIZE_HEIGHT) // 2))

			return np.array(img)

	# Update "rgb" with "img_arr"
	def _sum_rgb(self, img_arr, rgb):
		for color in RGB_DICT:
			# If an image is in monochrome
			if not img_arr.shape[2:3]:
				print("This is in monochrome!")
				return rgb

			buf = img_arr[:, :, RGB_DICT[color]]

			rgb[RGB_DICT[color]] = self._add(rgb[RGB_DICT[color]], buf, len(rgb[RGB_DICT[color]]),
				len(rgb[RGB_DICT[color]][0]))
		return rgb

	# Append the item to the array
	@staticmethod
	@njit("u4[:,:,:](u4[:,:,:], u1[:,:], u2, u2)")
	def _add(loaded, new, len_loaded, len_line):
		for i in range(len_loaded):
			for j in range(len_line):
				loaded[i, j, new[i, j]] += 1
		return loaded

	# Save results array in binary format
	def _save_rgbarr(self, rgb):
		rgb_dir = self.path + "\\rgb_files\\"
		for color in RGB_DICT:
			if not os.path.isdir(rgb_dir):
				os.makedirs(rgb_dir)
			buf_path = rgb_dir + color + ".npy"
			np.save(buf_path, rgb[RGB_DICT[color]])

	# Aggrigate results
	def aggrigate(self):
		print("Aggrigating results...")
		res_mean = np.zeros((RESIZE_HEIGHT, RESIZE_WIDTH, 3), dtype="uint8")
		res_mode = np.zeros((RESIZE_HEIGHT, RESIZE_WIDTH, 3), dtype="uint8")

		for color in RGB_DICT:
			rgb_path = self.path + "\\rgb_files\\" + color

			buf = np.zeros((RESIZE_HEIGHT, RESIZE_WIDTH, 256), dtype="uint32")
			buf += np.load(rgb_path + ".npy")

			np.save(rgb_path + "_all.npy", buf)

			mode = np.argmax(buf, axis=2).astype("uint8")
			mean = self._mean(buf).astype("uint8")

			np.save(rgb_path + "_agg_mode.npy", mode)
			np.save(rgb_path + "_agg_mean.npy", mean)

			res_mode[:, :, RGB_DICT[color]] += mode
			res_mean[:, :, RGB_DICT[color]] += mean
		print("Aggrigating is completed!")

	# Computing the mean
	@staticmethod
	@jit
	def _mean(buf):
		res = np.zeros((RESIZE_HEIGHT, RESIZE_WIDTH), dtype="uint8")
		for i in range(buf.shape[0]):
			for j in range(buf.shape[1]):
				sum = 0
				for k in range(buf.shape[2]):
					sum += k * buf[i, j, k]
				res[i, j] = int(sum / np.sum(buf[i, j, :]))
		return res

	# Show images of results
	def show_agg_img(self):
		print("Showing images of results...")

		(load_mode, load_mean) = self._load_agg()

		fig = plt.figure()
		plt.subplot(2, 1, 1)
		plt.title("mode")
		plt.imshow(load_mode)

		plt.subplot(2, 1, 2)
		plt.title("mean")
		plt.imshow(load_mean)
		plt.draw()

	# Load data of results
	def _load_agg(self):
		res_mean = np.zeros((RESIZE_HEIGHT, RESIZE_WIDTH, 3), dtype="uint8")
		res_mode = np.zeros((RESIZE_HEIGHT, RESIZE_WIDTH, 3), dtype="uint8")

		for color in RGB_DICT:
			rgb_path = self.path + "\\rgb_files\\" + color

			mode = np.load(rgb_path + "_agg_mode.npy")
			mean = np.load(rgb_path + "_agg_mean.npy")

			res_mode[:, :, RGB_DICT[color]] += mode
			res_mean[:, :, RGB_DICT[color]] += mean

		return res_mode, res_mean

	# Save images of results
	def save_agg_img(self):
		print("Saving images of results...")

		(load_mode, load_mean) = self._load_agg()

		Image.fromarray(load_mode).save(self.path + "mode.bmp")
		Image.fromarray(load_mean).save(self.path + "mean.bmp")

		print("Saving images of resulting is completed!")

	# Show histograms of mode and mean resulting images relative to the center point of the image
	def show_histogram(self):
		print("Showing histograms...")
		histogram = []
		for color in RGB_DICT:
			rgb_path = self.path + "\\rgb_files\\" + color
			buf = np.load(rgb_path + "_all.npy")
			histogram.append(buf[RESIZE_HEIGHT // 2, RESIZE_WIDTH // 2, :])

		fig = plt.figure()
		plt.subplot(1, 3, 1)
		plt.title("Red")
		plt.xlim([-2, 257])
		plt.plot(histogram[RGB_DICT["red"]])

		plt.subplot(1, 3, 2)
		plt.title("Green")
		plt.xlim([-2, 257])
		plt.plot(histogram[RGB_DICT["green"]])

		plt.subplot(1, 3, 3)
		plt.title("Blue")
		plt.xlim([-2, 257])
		plt.plot(histogram[RGB_DICT["blue"]])
		plt.draw()


if __name__ == "__main__":
	app = StackOfMemories()

	app.get_subdir()
	app.get_files(100) # Percentage of images acquired from each directory
	app.process_each()
	app.aggrigate()
	app.show_agg_img()
	app.save_agg_img()
	app.show_histogram()

	plt.show()
