from mocrograd import tensor, matrix

from . import modules


alias LossFunction = fn (
    input: List[tensor.Tensor],
    target: List[tensor.Tensor],
    parameters_dict: modules.ParametersDict,
) raises -> tensor.Tensor


alias AccuracyFunction = fn (
    input: List[tensor.Tensor],
    target: List[tensor.Tensor],
) raises -> Float32


fn cross_entropy_loss(
    input: List[tensor.Tensor],
    target: List[tensor.Tensor],
    parameters_dict: modules.ParametersDict,
) raises -> tensor.Tensor:
    if len(input) != len(target):
        raise "InputTargetLenError"

    var n_outputs = 0
    var summed_losses = tensor.Tensor(0.0)
    for i in range(len(input)):
        var loss = target[i] @ input[i]
        summed_losses = summed_losses + loss
        n_outputs += 1

    var data_loss = -summed_losses / n_outputs

    # L2 regularization
    # var reg_loss = get_reg_loss(parameters_dict)
    # return data_loss + reg_loss
    return data_loss


fn get_reg_loss(
    parameters_dict: modules.ParametersDict,
) raises -> tensor.Tensor:
    """L2 norm  places an outsize penalty on large components of the weight vector.
    This biases our learning algorithm towards models that **distribute weight evenly
    across a larger number of features**."""
    var alpha = Float32(1e-4)
    var summed_squared_parameters = Float32(0.0)

    for parameters_sequence in parameters_dict.values():
        for parameters in parameters_sequence[]:
            summed_squared_parameters += (parameters[] ** 2).sum().item()
    return tensor.Tensor(alpha * summed_squared_parameters)


fn calculate_accuracy(
    input: List[tensor.Tensor],
    target: List[tensor.Tensor],
) raises -> Float32:
    if len(input) != len(target):
        raise "InputTargetLenError"

    var correct_predictions = 0
    var all_predations = 0
    for i in range(len(input)):
        # Step 1: Convert probabilities to predicted classes
        var predicted_classes = input[i].data.argmax()

        # Step 2: Convert one-hot encoded targets to class labels
        true_classes = target[i].data.argmax()

        # Step 3: Calculate accuracy
        if predicted_classes == true_classes:
            correct_predictions += 1
        all_predations += 1

    return Float32(correct_predictions) / Float32(all_predations)
