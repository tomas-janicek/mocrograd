from mocrograd import datasets, nn, trainers


fn train_digits(training_type: String, epochs: UInt) raises:
    if training_type == "normal":
        var model = nn.MLPDigits()
        _train_digits_on_model(epochs=epochs, model=model^)
    elif training_type == "longer":
        var model = nn.MLPDigitsLonger(n_layers=30)
        _train_digits_on_model(epochs=epochs, model=model^)
    elif training_type == "bigger":
        var model = nn.MLPDigitsBigger()
        _train_digits_on_model(epochs=epochs, model=model^)
    else:
        raise "BadDigitsModelType"


fn _train_digits_on_model[
    ModuleT: nn.Module
](owned model: ModuleT, epochs: UInt) raises:
    var optimizer = nn.SGD(
        learning_rate=0.01, parameters_dict=model.parameters()
    )
    var trainer = trainers.Trainer(
        model=model^,
        optimizer=optimizer^,
        loss_function=nn.cross_entropy_loss,
        accuracy_function=nn.calculate_accuracy,
    )
    var data = datasets.DigitsData(length=1)
    var dataloader = datasets.Dataloader(dataset=data^, batch_size=32)
    trainer.fit(epochs=epochs, dataloader=dataloader^)


fn main() raises:
    print("Training on Normal Model")
    train_digits("normal", 5)
    print("Training on Longer Model")
    train_digits("longer", 5)
    print("Training on Bigger Model")
    train_digits("bigger", 5)
