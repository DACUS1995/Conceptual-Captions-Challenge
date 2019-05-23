import tensorflow as tf
import numpy as np
from utils import *

"""## Model

Fun fact, the decoder below is identical to the one in the example for [Neural Machine Translation with Attention](../sequences/nmt_with_attention.ipynb).

The model architecture is inspired by the [Show, Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf) paper.

* In this example, we extract the features from the lower convolutional layer of InceptionV3 giving us a vector of shape (8, 8, 2048).
* We squash that to a shape of (64, 2048).
* This vector is then passed through the CNN Encoder(which consists of a single Fully connected layer).
* The RNN(here GRU) attends over the image to predict the next word.
"""


class BahdanauAttention(tf.keras.Model):
	def __init__(self, units):
		super(BahdanauAttention, self).__init__()
		self.W1 = tf.keras.layers.Dense(units)
		self.W2 = tf.keras.layers.Dense(units)
		self.V = tf.keras.layers.Dense(1)

	def call(self, features, hidden):
		# features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

		# hidden shape == (batch_size, hidden_size)
		# hidden_with_time_axis shape == (batch_size, 1, hidden_size)
		hidden_with_time_axis = tf.expand_dims(hidden, 1)

		# score shape == (batch_size, 64, hidden_size)
		score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

		# attention_weights shape == (batch_size, 64, 1)
		# we get 1 at the last axis because we are applying score to self.V
		attention_weights = tf.nn.softmax(self.V(score), axis=1)

		# context_vector shape after sum == (batch_size, hidden_size)
		context_vector = attention_weights * features
		context_vector = tf.reduce_sum(context_vector, axis=1)

		return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
	# Since we have already extracted the features and dumped it using pickle
	# This encoder passes those features through a Fully connected layer
	def __init__(self, embedding_dim):
		super(CNN_Encoder, self).__init__()
		# shape after fc == (batch_size, 64, embedding_dim)
		self.fc = tf.keras.layers.Dense(embedding_dim)

	def call(self, x):
		x = self.fc(x)
		x = tf.nn.relu(x)
		return x


class RNN_Decoder(tf.keras.Model):
	def __init__(self, embedding_dim, units, vocab_size):
		super(RNN_Decoder, self).__init__()
		self.units = units
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = tf.keras.layers.GRU(self.units,
									   return_sequences=True,
									   return_state=True,
									   recurrent_initializer='glorot_uniform')
		self.fc1 = tf.keras.layers.Dense(self.units)
		self.fc2 = tf.keras.layers.Dense(vocab_size)

		self.attention = BahdanauAttention(self.units)

	def call(self, x, features, hidden):
		# defining attention as a separate model
		context_vector, attention_weights = self.attention(features, hidden)
		# x shape after passing through embedding == (batch_size, 1, embedding_dim)

		x = self.embedding(x)
		# x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
		x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
		# passing the concatenated vector to the GRU
		output, state = self.gru(x)

		# shape == (batch_size, 1, hidden_size)
		x = self.fc1(output)

		# x shape == (batch_size, hidden_size)
		x = tf.reshape(x, (-1, x.shape[2]))

		# output shape == (batch_size, vocab)
		x = self.fc2(x)

		return x, state, attention_weights

	def reset_state(self, batch_size):
		return tf.zeros((batch_size, self.units))


class AttentionDecoderEncoder():
	def __init__(self, embedding_dim, units, vocabulary_size, start_token, batch_size, max_train_len, max_val_len):
		self.encoder = CNN_Encoder(embedding_dim)
		self.decoder = RNN_Decoder(embedding_dim, units, vocabulary_size)
		self.start_token = start_token
		self.batch_size = batch_size

		self.optimizer = tf.keras.optimizers.Adam()
		self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
			from_logits=True, reduction='none')
		self.max_train_len = max_train_len
		self.max_val_len = max_val_len

	def get_ckpt_config(self):
		ckpt = tf.train.Checkpoint(encoder=self.encoder,
								   decoder=self.decoder,
								   optimizer=self.optimizer)
		return ckpt

	def get_valid_mask(self, real):
		mask = tf.math.logical_not(tf.math.equal(real, 0))
		return mask

	def loss_function(self, real, pred, mask):
		loss_ = self.loss_object(real, pred)
		mask = tf.cast(mask, dtype=loss_.dtype)
		loss_ *= mask
		return tf.reduce_mean(loss_)

	@tf.function
	def train_step(self, img_tensor, target, metrics):
		loss = 0

		for metric in metrics:
			metrics[metric].reset_states()

		# initializing the hidden state for each batch
		# because the captions are not related from image to image
		hidden = self.decoder.reset_state(self.batch_size)


		dec_input = tf.expand_dims([self.start_token] * self.batch_size, 1)

		with tf.GradientTape() as tape:
			features = self.encoder(img_tensor)

			for i in range(1, self.max_train_len):
				# passing the features through the decoder
				predictions, hidden, _ = self.decoder(dec_input, features, hidden)

				valid_mask = self.get_valid_mask(target[:, i])
				current_loss = self.loss_function(target[:, i], predictions, valid_mask)
				loss += current_loss
				# using teacher forcing
				dec_input = tf.expand_dims(target[:, i], 1)

				metrics['loss'](current_loss)

		trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
		gradients = tape.gradient(loss, trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, trainable_variables))

	@tf.function
	def val_step(self, img_tensor, target, metrics):
		for metric in metrics:
			metrics[metric].reset_states()

		loss = 0

		hidden = self.decoder.reset_state(batch_size=self.batch_size)

		dec_input = tf.expand_dims([self.start_token] * self.batch_size, 1)

		features = self.encoder(img_tensor)

		for i in range(1, self.max_val_len):
			# passing the features through the decoder
			predictions, hidden, _ = self.decoder(dec_input, features, hidden)

			valid_mask = self.get_valid_mask(target[:, i])
			current_loss = self.loss_function(target[:, i], predictions, valid_mask)
			loss += current_loss
			# using teacher forcing
			dec_input = tf.expand_dims(target[:, i], 1)

			metrics['loss'](current_loss)

	def evaluate(self, image, max_seq_length, attention_features_shape, input_resize_dim, tokenizer):
		attention_plot = np.zeros((max_seq_length, attention_features_shape))

		hidden = self.decoder.reset_state(batch_size=1)

		temp_input = tf.expand_dims(load_image(image, input_resize_dim)[0], 0)
		img_tensor_val = image_features_extract_model(temp_input)
		img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

		features = self.encoder(img_tensor_val)

		dec_input = tf.expand_dims([self.start_token], 0)
		result = []

		for i in range(max_seq_length):
			predictions, hidden, attention_weights = self.decoder(dec_input, features, hidden)

			attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

			predicted_id = tf.argmax(predictions[0]).numpy()
			result.append(tokenizer.index_word[predicted_id])

			if tokenizer.index_word[predicted_id] == '<end>':
				return result, attention_plot

			dec_input = tf.expand_dims([predicted_id], 0)

		attention_plot = attention_plot[:len(result), :]
		return result, attention_plot
