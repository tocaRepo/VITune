from vit_finetuner import VITFineTuner

def main():
    #make sure to have 1 folder called 'train' in the dataset folder, VITune will automatically create a test dataset off that
    dataset_path = "./dataset"
    fine_tuner = VITFineTuner(dataset_path)
    fine_tuner.load_dataset()
    fine_tuner.prepare_dataset()
    fine_tuner.setup_model()
    fine_tuner.setup_training_args()
    fine_tuner.setup_trainer()
    fine_tuner.train()
    fine_tuner.plot_metrics()

if __name__ == "__main__":
    main()
