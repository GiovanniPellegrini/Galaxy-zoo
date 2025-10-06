from galaxy_classification.model.galaxy_cnn import GalaxyClassificationCNN, GalaxyClassificationCNNConfig, GalaxyRegressionCNN, GalaxyRegressionCNNConfig


def build_network(
    input_image_shape: tuple[int, int],
    config: GalaxyClassificationCNNConfig| GalaxyRegressionCNNConfig
):
    match config:
        case GalaxyClassificationCNNConfig(
            channel_count_hidden=channel_count_hidden,
            convolution_kernel_size=convolution_kernel_size,
            mlp_hidden_unit_count=mlp_hidden_unit_count,
        ):
            return GalaxyClassificationCNN(
                input_image_shape,
                channel_count_hidden,
                convolution_kernel_size,
                mlp_hidden_unit_count,
            )
        case GalaxyRegressionCNNConfig(
            channel_count_hidden=channel_count_hidden,
            convolution_kernel_size=convolution_kernel_size,
            mlp_hidden_unit_count=mlp_hidden_unit_count,
        ):
            return GalaxyRegressionCNN(
                input_image_shape,
                channel_count_hidden,
                convolution_kernel_size,
                mlp_hidden_unit_count,
            )    
        case _:
            raise ValueError("Invalid network configuration")
        