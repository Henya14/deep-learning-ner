from training import load_model, NERModel, LABEL_COUNT





if __name__ == '__main__':
    
    model = NERModel(LABEL_COUNT)
    model = load_model(model, "name")
    model.eval()
    model()

