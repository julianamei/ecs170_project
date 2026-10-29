from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_ORL(Method_CNN):
    input_shape = (1, 112, 92)
    output_dim = 40
    conv_channels = [16, 32, 64]
    hidden_dims = [128]
    dropout = 0.5
    batch_size = 32
    max_epoch = 30