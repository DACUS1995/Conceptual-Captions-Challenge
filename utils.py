import zipfile
import tensorflow as tf
from matplotlib import pyplot as plt
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

def save_fig():
	plt.savefig(str(plt.gcf().number))

def plot_image_attention(title, image, result, attention_plot):
	temp_image = np.array(Image.open(image))

	fig = plt.figure(figsize=(16, 10))

	len_result = len(result)

	plot_columns = min(5, len_result)
	plot_lines = int(np.ceil(len_result / plot_columns))
	for l in range(len_result):
		temp_att = np.resize(attention_plot[l], (8, 8))
		ax = fig.add_subplot(plot_lines, plot_columns, l + 1)
		ax.set_title(result[l])
		img = ax.imshow(temp_image)
		ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

	fig.suptitle(title)
	#     plt.tight_layout()
	# plt.savefig()
	save_fig()

def plot_multi_head_text_attention(title, result, attention):
	fig = plt.figure(figsize=(16, 8))

	no_heads = attention.shape[0]
	plot_columns = min(5, no_heads)
	plot_lines = int(np.ceil(no_heads / plot_columns))

	for head in range(attention.shape[0]):
		ax = fig.add_subplot(plot_lines, plot_columns, head + 1)

		# plot the attention weights
		ax.matshow(attention[head][:, :], cmap='viridis')

		fontdict = {'fontsize': 10}

		ax.set_xticks(range(len(result)))
		ax.set_yticks(range(len(result)))

		ax.set_ylim(len(result) - 1.5, -0.5)

		ax.set_xticklabels(result, fontdict=fontdict, rotation=90)

		ax.set_yticklabels(result, fontdict=fontdict)

		ax.set_xlabel('Head {}'.format(head + 1))

	fig.suptitle(title)
	plt.tight_layout()
	# plt.show()
	save_fig()

def plot_multi_head_image_attention(layer, image, result, attention):
  for head_i, head_attention in enumerate(attention):
    title = layer + "head " + str(head_i)
    plot_image_attention(title, image, result, head_attention)


image_model = tf.keras.applications.InceptionV3(include_top=False,
                                              weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
