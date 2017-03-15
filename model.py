from keras.layers import Input, LSTM, Reshape, Activation
from keras.layers import Embedding, merge, Lambda
from keras.models import Model


def get_model(word_to_idx, embedding_dim, max_context_len, max_ending_len, lstm_size, dropout, embeddings=None):
    """
    Builds the keras model for training or inference.
    :param word_to_idx: needed for the input dimension of the embedding layer
    :param embedding_dim: dimension of one embedding
    :param max_context_len: number of indices in the longest context input
    :param max_ending_len: number of indices in the longest ending input
    :param lstm_size: Size of the lstm output layer
    :param dropout: Float between 0 and 1. Fraction of the input units to drop for input gates and recurrent connections
    :param embeddings: embedding weights
    :return: initialized model
    """

    story_input = Input(shape=(max_context_len + max_ending_len,))
    # during inference, the embedding weights will be set later, so init randomly for now
    if embeddings is not None:
        embedding_layer = Embedding(input_dim=len(word_to_idx)+1, output_dim=embedding_dim, weights=[embeddings],
                                    trainable=False, name='embeddings', mask_zero=False)
    else:
        embedding_layer = Embedding(input_dim=len(word_to_idx)+1, output_dim=embedding_dim,
                                    trainable=False, name='embeddings', mask_zero=False)

    lstm = LSTM(output_dim=lstm_size, name='lstm', return_sequences=True,
                dropout_W=dropout, dropout_U=dropout,
                activation='sigmoid')

    lstm_sequence = lstm(embedding_layer(story_input))
    # context embedding is the output of the lstm after the 'end context' flag
    context_embedding = Lambda(lambda x: x[:, max_context_len, :], name='context_embedding',
                               output_shape=lambda s: (s[0], s[2]))(lstm_sequence)
    # context embedding is the last output of the lstm
    ending_embedding = Lambda(lambda x: x[:, -1, :], name='ending_embedding',
                              output_shape=lambda s: (s[0], s[2]))(lstm_sequence)

    # compute cosine between context embedding and ending embedding
    cos_distance = merge([context_embedding, ending_embedding], mode='cos', dot_axes=1)
    cos_similarity = Lambda(lambda x: 1-x, output_shape=lambda s: s)(cos_distance)
    cos_similarity = Reshape(target_shape=(1,))(cos_similarity)

    activation = Activation(activation='sigmoid')(cos_similarity)

    model = Model(input=[story_input], output=activation)
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adagrad')

    return model

