import argparse
from chemprop.args import PredictArgs
from chemprop.train import make_predictions
from data_processing import InputGenerator
import pandas as pd



def gnn_prediction(paper_id, smiles, membrane, solvent, configuration):
    feature_generator = InputGenerator(smiles, [membrane], [solvent], configuration)
    feature_generator.generate_features()
    feature_generator.dump(rf"results/_temp/temp_generated_features.csv") 

    arguments = [
        '--test_path', 'results/_temp/temp_generated_features.csv', 
        '--preds_path', 'results/_temp/temp_predictions.csv',
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
        "paper_id": paper_id,
        "smiles": smiles,
        "membrane": membrane,
        "solvent": solvent,
        "configuration": configuration,
        "rejection": preds[0],
        "corrected_rejection": rounded_preds,
    }

    return retrieved_features

def main():
    results_df = pd.DataFrame(columns=["paper_id", "smiles", "membrane", "solvent", "configuration", "rejection", "corrected_rejection"])
    df = pd.read_csv("data/smiles.csv")
    for _, row in df.iterrows():
        paper_id = row["paper_id"]
        smiles = row["smiles"]
        membrane = row["membrane"]
        solvent = row["solvent"]
        configuration = row["configuration"]
        results = pd.DataFrame(gnn_prediction(paper_id, smiles, membrane, solvent, configuration))
        results_df = results_df.append(results, ignore_index=True)

    results_df.to_csv("results/paper_predictions.csv", index=False)
    

if __name__ == "__main__":
    main()
