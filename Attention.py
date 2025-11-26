from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda, AveragePooling2D, BatchNormalization, UpSampling2D
from keras.layers import Dense, Layer, Dropout
from keras import backend as K
from keras.activations import sigmoid
import tensorflow as tf

def custom_attention(input_feature, ratio=8, kernel_size=7):
    """
    Custom Attention Module: Combines Spatial Attention and Channel Attention (SE).
    """
    # Get channel dimension
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = K.int_shape(input_feature)[channel_axis]
    assert channel is not None, "Input feature map must have a defined channel dimension."

    # ----- Condition A: Spatial Attention -----
    # Global Average Pooling
    avg_pool = Lambda(lambda x: K.mean(x, axis=channel_axis, keepdims=True))(input_feature)
    assert K.int_shape(avg_pool)[channel_axis] == 1, "Average pooling should reduce channel axis to 1."
   
    # Global Max Pooling
    max_pool = Lambda(lambda x: K.max(x, axis=channel_axis, keepdims=True))(input_feature)
    assert K.int_shape(max_pool)[channel_axis] == 1, "Max pooling should reduce channel axis to 1."
   
    # Concatenate along channel axis
    concat = Concatenate(axis=channel_axis)([avg_pool, max_pool])
    assert K.int_shape(concat)[channel_axis] == 2, "Concatenation should result in 2 channels (Avg + Max)."
   
    # Convolution to produce spatial attention weights
    spatial_attention = Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid')(concat)
    assert K.int_shape(spatial_attention)[channel_axis] == 1, "Spatial attention should have 1 channel after Conv2D."
   
    # Apply spatial attention weights
    F_s = multiply([input_feature, spatial_attention])
    assert K.int_shape(F_s) == K.int_shape(input_feature), "Spatially refined feature map should match input dimensions."

    # ----- Condition B: Channel Attention (SE Block) -----
    # Squeeze: Global Average Pooling
    gap = GlobalAveragePooling2D()(input_feature)
    assert K.int_shape(gap)[-1] == channel, "Global Average Pooling should reduce spatial dimensions but keep channels."
   
    # First Dense Layer (Reduction)
    dense1 = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)(gap)
    assert K.int_shape(dense1)[-1] == channel // ratio, "First Dense layer should reduce channels by 'ratio'."
   
    # Second Dense Layer (Expansion)
    dense2 = Dense(channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=True)(dense1)
    assert K.int_shape(dense2)[-1] == channel, "Second Dense layer should restore the original channel size."
   
    # Reshape to apply channel attention
    channel_attention = Reshape((1, 1, channel))(dense2)
    assert K.int_shape(channel_attention)[1:] == (1, 1, channel), "Channel attention should have shape (1, 1, channels)."
   
    # Apply channel attention weights
    F_c = multiply([input_feature, channel_attention])
    assert K.int_shape(F_c) == K.int_shape(input_feature), "Channel-refined feature map should match input dimensions."

    # ----- Combine Spatial and Channel Attention -----
    output = Add()([F_s, F_c])
    assert K.int_shape(output) == K.int_shape(input_feature), "Final output should match the input dimensions."

    return output

# class MultiHeadSelfAttention(Layer):
#     def __init__(self, num_heads, key_dim=None, dropout=0.1, **kwargs):
#         super(MultiHeadSelfAttention, self).__init__(**kwargs)
#         self.num_heads = num_heads
#         self.key_dim = key_dim
#         self.dropout = Dropout(dropout)
        
#     def build(self, input_shape):
#         embedding_dim = input_shape[-1]
        
#         # Calculate key_dim if not specified and ensure divisibility
#         if self.key_dim is None:
#             if embedding_dim % self.num_heads != 0:
#                 raise ValueError(f"Embedding dimension {embedding_dim} must be divisible by num_heads {self.num_heads}.")
#             self.key_dim = embedding_dim // self.num_heads
            
#         self.query_dense = Dense(self.num_heads * self.key_dim, use_bias=False)
#         self.key_dense = Dense(self.num_heads * self.key_dim, use_bias=False)
#         self.value_dense = Dense(self.num_heads * self.key_dim, use_bias=False)
#         self.output_dense = Dense(embedding_dim, use_bias=False)
#         super(MultiHeadSelfAttention, self).build(input_shape)
    
#     def call(self, inputs):
#         batch_size = tf.shape(inputs)[0]
        
#         # Project inputs to queries, keys, and values
#         query = self.query_dense(inputs)
#         key = self.key_dense(inputs)
#         value = self.value_dense(inputs)
        
#         # Split into heads
#         query = self._split_heads(query, batch_size)
#         key = self._split_heads(key, batch_size)
#         value = self._split_heads(value, batch_size)
        
#         # Scaled dot-product attention
#         attention_scores = tf.matmul(query, key, transpose_b=True)
#         attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
#         attention_weights = tf.nn.softmax(attention_scores, axis=-1)
#         attention_weights = self.dropout(attention_weights)

#         # Compute attention context
#         context = tf.matmul(attention_weights, value)
#         context = self._combine_heads(context, batch_size)
        
#         # Final linear projection
#         output = self.output_dense(context)
#         return output

#     def _split_heads(self, x, batch_size):
#         """Split the last dimension into (num_heads, key_dim)."""
#         x = tf.reshape(x, (batch_size, -1, self.num_heads, self.key_dim))
#         return tf.transpose(x, perm=[0, 2, 1, 3])

#     def _combine_heads(self, x, batch_size):
#         """Combine the heads into the original shape."""
#         x = tf.transpose(x, perm=[0, 2, 1, 3])
#         return tf.reshape(x, (batch_size, -1, self.num_heads * self.key_dim))
    
#     def get_config(self):
#         config = super(MultiHeadSelfAttention, self).get_config()
#         config.update({"num_heads": self.num_heads, "key_dim": self.key_dim})
#         return config    

# class LayerNormalization(Layer):
#     def __init__(self, epsilon=1e-6, **kwargs):
#         self.epsilon = epsilon
#         super(LayerNormalization, self).__init__(**kwargs)

#     def build(self, input_shape):
#         self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer='ones', trainable=True)
#         self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer='zeros', trainable=True)
#         super(LayerNormalization, self).build(input_shape)

#     def call(self, inputs):
#         mean = K.mean(inputs, axis=-1, keepdims=True)
#         std = K.std(inputs, axis=-1, keepdims=True)
#         return self.gamma * (inputs - mean) / (std + self.epsilon) + self.beta
    
#     def get_config(self):
#         config = super(LayerNormalization, self).get_config()
#         config.update({"epsilon": self.epsilon})
#         return config

def attach_attention_module(net, attention_module):
  if attention_module == 'se_block': # SE_block
    net = se_block(net)
  elif attention_module == 'cbam_block': # CBAM_block
    net = cbam_block(net)
  elif attention_module == 'self_attention_block': # self_attention_block
    net = self_attention_block(net)
  else:
    raise Exception("'{}' is not supported attention module!".format(attention_module))

  return net

def se_block(input_feature, ratio=8):
	"""Contains the implementation of Squeeze-and-Excitation(SE) block.
	As described in https://arxiv.org/abs/1709.01507.
	"""
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature.shape[channel_axis]

	se_feature = GlobalAveragePooling2D()(input_feature)
	se_feature = Reshape((1, 1, channel))(se_feature)
	assert se_feature.shape[1:] == (1,1,channel)
	se_feature = Dense(channel // ratio,
					   activation='relu',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature.shape[1:] == (1,1,channel//ratio)
	se_feature = Dense(channel,
					   activation='sigmoid',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature.shape[1:] == (1,1,channel)
	if K.image_data_format() == 'channels_first':
		se_feature = Permute((3, 1, 2))(se_feature)

	se_feature = multiply([input_feature, se_feature])
	return se_feature

def cbam_block(cbam_feature, ratio=8):
	"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	As described in https://arxiv.org/abs/1807.06521.
	"""
	
	cbam_feature = channel_attention(cbam_feature, ratio)
	cbam_feature = spatial_attention(cbam_feature)
	return cbam_feature

def channel_attention(input_feature, ratio=8):
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature.shape[channel_axis]
	
	shared_layer_one = Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	
	avg_pool = GlobalAveragePooling2D()(input_feature)    
	avg_pool = Reshape((1,1,channel))(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel)
	
	max_pool = GlobalMaxPooling2D()(input_feature)
	max_pool = Reshape((1,1,channel))(max_pool)
	assert max_pool.shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	assert max_pool.shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	assert max_pool.shape[1:] == (1,1,channel)
	
	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('sigmoid')(cbam_feature)
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
	
	return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
	kernel_size = 7
	
	if K.image_data_format() == "channels_first":
		channel = input_feature.shape[1]
		cbam_feature = Permute((2,3,1))(input_feature)
	else:
		channel = input_feature.shape[-1]
		cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	assert avg_pool.shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	assert max_pool.shape[-1] == 1
	concat = Concatenate(axis=3)([avg_pool, max_pool])
	assert concat.shape[-1] == 2
	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='sigmoid',
					kernel_initializer='he_normal',
					use_bias=False)(concat)	
	assert cbam_feature.shape[-1] == 1
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
		
	return multiply([input_feature, cbam_feature])


from keras.layers import Conv2D, BatchNormalization, Activation, Reshape, Permute, multiply, AveragePooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, Lambda
import tensorflow.keras.backend as K

def self_attention_block(input_tensor, reduction_ratio=4):
  
    
    # Step 1: Downsample spatially to reduce memory usage
    reduced_input = AveragePooling2D(pool_size=(reduction_ratio, reduction_ratio), padding='same')(input_tensor)
    
    # Extract spatial dimensions and channels
    shape = K.int_shape(reduced_input)
    height, width, channels = shape[1], shape[2], shape[3]
    
    # Flatten the spatial dimensions for query, key, and value
    query = Reshape((height * width, channels))(reduced_input)
    key = Reshape((height * width, channels))(reduced_input)
    value = Reshape((height * width, channels))(reduced_input)
    
    # Step 3: Calculate attention matrix with axes adjustment
    attention_scores = K.batch_dot(query, key, axes=[2, 2]) / K.sqrt(K.cast(channels, K.floatx()))
    attention_scores = Activation('softmax')(attention_scores)
    
    # Step 4: Multiply attention scores with value
    attention_output = K.batch_dot(attention_scores, value, axes=[1, 1])
    
    # Step 5: Reshape attention output back to reduced input shape
    attention_output = Reshape((height, width, channels))(attention_output)
    
    # Step 6: Upsample to match input tensor's dimensions
    attention_output = UpSampling2D(size=(reduction_ratio, reduction_ratio), interpolation='bilinear')(attention_output)
    attention_output = Conv2D(K.int_shape(input_tensor)[-1], (1, 1), padding='same', activation='sigmoid')(attention_output)

    # Step 7: Adjust dimensions if there's a mismatch
    input_shape = K.int_shape(input_tensor)
    output_shape = K.int_shape(attention_output)
    height_diff = input_shape[1] - output_shape[1]
    width_diff = input_shape[2] - output_shape[2]

    # Apply padding or cropping to align dimensions
    if height_diff > 0 or width_diff > 0:
        # Pad attention_output to match input_tensor dimensions
        padding_height = (height_diff // 2, height_diff - height_diff // 2)
        padding_width = (width_diff // 2, width_diff - width_diff // 2)
        attention_output = ZeroPadding2D(padding=(padding_height, padding_width))(attention_output)
    elif height_diff < 0 or width_diff < 0:
        # Crop attention_output to match input_tensor dimensions
        cropping_height = (abs(height_diff) // 2, abs(height_diff) - abs(height_diff) // 2)
        cropping_width = (abs(width_diff) // 2, abs(width_diff) - abs(width_diff) // 2)
        attention_output = Cropping2D(cropping=(cropping_height, cropping_width))(attention_output)

    # Step 8: Multiply input and attention output (residual connection)
    output = multiply([input_tensor, attention_output])
    
    return output
