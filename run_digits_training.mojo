import sys

from mocrograd import datasets, nn, trainers


fn train_digits(*, model_type: String, epochs: UInt, length: UInt) raises:
    if model_type == "normal":
        var model = nn.MLPDigits()
        _train_digits_on_model(epochs=epochs, length=length, model=model^)
    elif model_type == "longer":
        var model = nn.MLPDigitsLonger(n_layers=30)
        _train_digits_on_model(epochs=epochs, length=length, model=model^)
    elif model_type == "bigger":
        var model = nn.MLPDigitsBigger()
        _train_digits_on_model(epochs=epochs, length=length, model=model^)
    else:
        raise "BadDigitsModelType"


fn _train_digits_on_model[
    ModuleT: nn.Module
](*, owned model: ModuleT, epochs: UInt, length: UInt) raises:
    var optimizer = nn.SGD(
        learning_rate=0.01, parameters_dict=model.parameters()
    )
    var trainer = trainers.Trainer(
        model=model^,
        optimizer=optimizer^,
        loss_function=nn.cross_entropy_loss,
        accuracy_function=nn.calculate_accuracy,
    )
    var data = datasets.DigitsData(length=length)
    var dataloader = datasets.Dataloader(dataset=data^, batch_size=32)
    trainer.fit(epochs=epochs, dataloader=dataloader^)


fn main() raises:
    var arguments = sys.argv()
    if len(arguments) != 3:
        raise "BadArguments: Should be run_digits_training.mojo <epochs> <length>"
    var epochs = Int(arguments[1])
    var length = Int(arguments[2])
    print("\nTraining on Normal Model")
    train_digits(model_type="normal", epochs=epochs, length=length)
    print("\nTraining on Longer Model")
    train_digits(model_type="longer", epochs=epochs, length=length)
    print("\nTraining on Bigger Model")
    train_digits(model_type="bigger", epochs=epochs, length=length)
