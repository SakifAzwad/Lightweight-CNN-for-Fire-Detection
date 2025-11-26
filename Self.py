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
    # attention_scores = K.batch_dot(query, key, axes=[2, 2]) / K.sqrt(K.cast(channels, K.floatx()))
    # attention_scores = Activation('softmax')(attention_scores)

    key_T = Permute((2, 1))(key)
    attention_scores = K.batch_dot(query, key_T)
    attention_scores = attention_scores / K.sqrt(K.cast(channels, K.floatx()))
    attention_scores = Activation('softmax')(attention_scores)

    
    # Step 4: Multiply attention scores with value
    # attention_output = K.batch_dot(attention_scores, value, axes=[1, 1])
    attention_output = K.batch_dot(attention_scores, value)

    
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