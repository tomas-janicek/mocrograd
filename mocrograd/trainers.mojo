import time

from mocrograd import nn, tensor, datasets


struct Trainer[ModuleT: nn.Module, OptimizerT: nn.Optimizer]:
    var model: ModuleT
    var optimizer: OptimizerT
    var loss_function: nn.LossFunction
    var accuracy_function: nn.AccuracyFunction

    fn __init__(
        out self,
        owned model: ModuleT,
        owned optimizer: OptimizerT,
        loss_function: nn.LossFunction,
        accuracy_function: nn.AccuracyFunction,
    ):
        self.model = model^
        self.optimizer = optimizer^
        self.loss_function = loss_function
        self.accuracy_function = accuracy_function

    fn fit(
        self,
        epochs: UInt,
        dataloader: datasets.Dataloader,
    ) raises -> None:
        var start = time.perf_counter_ns()
        for epoch in range(epochs):
            loss, accuracy = self.do_training(dataloader)
            print(
                String("Training Epoch   {}, loss {}, accuracy {}%").format(
                    epoch, loss, accuracy * 100
                )
            )

            loss, accuracy = self.do_validation(dataloader)
            print(
                String("Validation Epoch {}, loss {}, accuracy {}%").format(
                    epoch, loss, accuracy * 100
                )
            )

        var end = time.perf_counter_ns()
        var duration_in_seconds = (end - start) * 1e-9
        print(String("Training took {} seconds.").format(duration_in_seconds))

    fn do_training(
        self, dataloader: datasets.Dataloader
    ) raises -> Tuple[Float32, Float32]:
        var sum_losses = Float32(0.0)
        var sum_accuracies = Float32(0.0)
        var batches = 0

        for inputs_targets in dataloader.get_train_dataloader():
            inputs, targets = inputs_targets
            loss, accuracy = self.get_loss(inputs, targets)

            sum_losses += loss.item()
            sum_accuracies += accuracy
            batches += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        var loss_mean = sum_losses / batches
        var accuracies_mean = sum_accuracies / batches
        return loss_mean, accuracies_mean

    fn do_validation(
        self, dataloader: datasets.Dataloader
    ) raises -> Tuple[Float32, Float32]:
        var sum_losses = Float32(0.0)
        var sum_accuracies = Float32(0.0)
        var batches = 0

        for inputs_targets in dataloader.get_validation_dataloader():
            inputs, targets = inputs_targets
            loss, accuracy = self.get_loss(inputs, targets)

            sum_losses += loss.item()
            sum_accuracies += accuracy
            batches += 1

        var loss_mean = sum_losses / batches
        var accuracies_mean = sum_accuracies / batches
        return loss_mean, accuracies_mean

    fn get_loss(
        self,
        inputs: List[tensor.Tensor],
        targets: List[tensor.Tensor],
    ) raises -> Tuple[tensor.Tensor, Float32]:
        var predictions = List[tensor.Tensor]()

        for input in inputs:
            predictions.append(self.model(input[]))

        loss = self.loss_function(
            input=predictions,
            target=targets,
            parameters_dict=self.model.parameters(),
        )

        accuracy = self.accuracy_function(input=predictions, target=targets)
        return loss, accuracy
