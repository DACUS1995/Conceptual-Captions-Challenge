import zipfile
import tensorflow as tf
from matplotlib import plt
import numpy as np
from PIL import Image

def unzip(path_to_zip_file, directory_to_extract_to):
	zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
	zip_ref.extractall(directory_to_extract_to)
	zip_ref.close()


def load_image(image_path, resize_dim):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, resize_dim)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def plot_attention(image, result, attention_plot):
	temp_image = np.array(Image.open(image))

	fig = plt.figure(figsize=(10, 10))

	len_result = len(result)

	plot_columns = min(5, len_result)
	plot_lines = int(np.ceil(len_result / plot_columns))
	for l in range(len_result):
		temp_att = np.resize(attention_plot[l], (8, 8))
		ax = fig.add_subplot(plot_lines, plot_columns, l + 1)
		ax.set_title(result[l])
		img = ax.imshow(temp_image)
		ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

	plt.tight_layout()
	plt.show()


image_model = tf.keras.applications.InceptionV3(include_top=False,
                                              weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
