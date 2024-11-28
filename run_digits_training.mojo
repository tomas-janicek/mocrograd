from mocrograd import datasets, nn, trainers


fn main() raises:
    var epochs = 1
    var model = nn.MLPDigitsLonger()
    var optimizer = nn.SGD(
        learning_rate=0.01, parameters_dict=model.parameters()
    )
    var trainer = trainers.Trainer[nn.MLPDigitsLonger, nn.SGD](
        model=model^,
        optimizer=optimizer^,
        loss_function=nn.cross_entropy_loss,
        accuracy_function=nn.calculate_accuracy,
    )
    var data = datasets.DigitsData(length=1000)
    var dataloader = datasets.Dataloader(dataset=data^, batch_size=32)
    trainer.fit(epochs=epochs, dataloader=dataloader^)
