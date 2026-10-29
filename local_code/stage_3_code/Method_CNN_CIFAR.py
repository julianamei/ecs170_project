from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_CIFAR(Method_CNN):
    input_shape = (3, 32, 32)
    output_dim = 10
    conv_channels = [32, 64, 128]
    hidden_dims = [256]
    max_epoch = 15