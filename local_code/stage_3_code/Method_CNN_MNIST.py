from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_MNIST(Method_CNN):
    input_shape = (1, 28, 28)
    output_dim = 10
    conv_channels = [32, 64]
    hidden_dims = [128]
    max_epoch = 10