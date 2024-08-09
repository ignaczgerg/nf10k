import argparse
from chemprop.args import PredictArgs
from chemprop.train import make_predictions
from data_processing import InputGenerator

def gnn_prediction(smiles, membrane, solvent, configuration):
    print("Calling gnn_prediction")
    
    feature_generator = InputGenerator(smiles, [membrane], [solvent], configuration)
    feature_generator.generate_features()
    feature_generator.dump(rf"temp_generated_features.csv") 

    arguments = [
        '--test_path', 'temp_generated_features.csv', 
        '--preds_path', 'temp_predictions.csv',
        '--checkpoint_dir', 'models/',
        '--smiles_columns', 'full_smiles',
        '--features_generator', 'custom',  
        '--no_cuda'
    ]

    args = PredictArgs().parse_args(arguments)
    preds = make_predictions(args=args)
    rounded_preds = round(preds[0][0], 3)
    if rounded_preds > 1.0:
        rounded_preds = 1.0

    retrieved_features = {
        "smiles": smiles,
        "membrane": membrane,
        "solvent": solvent,
        "configuration": configuration,
        "rejection": rounded_preds
    }

    return retrieved_features

def main():
    parser = argparse.ArgumentParser(description="GNN Prediction CLI")
    parser.add_argument('--smiles', type=str, required=True, help="SMILES string of the molecule")
    parser.add_argument('--membrane', type=str, required=True, help="Membrane material")
    parser.add_argument('--solvent', type=str, required=True, help="Solvent name")
    parser.add_argument('--configuration', type=str, required=True, help="Configuration type")

    args = parser.parse_args()

    print("Prediction under progress, please wait.")
    results = gnn_prediction(args.smiles, args.membrane, args.solvent, args.configuration)
    print("Prediction finished")
    print(results)

if __name__ == "__main__":
    main()
