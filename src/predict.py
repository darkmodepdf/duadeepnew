import os
import torch
import pandas as pd
from network import DuaDeepImproved

def predict_interactions(ab_seqs, ag_seqs, model_path="checkpoints/best_model.pth"):
    """
    Inference script mapping unknown Antibody & Antigen pairings via the trained Interaction head.
    Outputs binding probabilities mimicking the original prediction capabilities of DuaDeep.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading trained DuaDeep Improved architecture to {device}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing Trained Model File: {model_path}. Execute training First.")
        
    model = DuaDeepImproved()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    results = []
    
    # Process sequentially due to dynamic constraints (For prod environments implement batching via dataloaders.)
    print(f"Performing inference across {len(ab_seqs)} inputs.")
    with torch.no_grad():
        for i, (ab, ag) in enumerate(zip(ab_seqs, ag_seqs)):
            # Add batch dimension to string lists.
            ab_t = [ab]
            ag_t = [ag]
            
            logits = model(ab_t, ag_t)
            prob = torch.softmax(logits, dim=1)[:, 1].item()
            predicted_class = "Binding" if prob > 0.5 else "Non-Binding"
            
            results.append({
                "Antibody": ab[:20] + "..." if len(ab) > 20 else ab,
                "Antigen": ag[:20] + "..." if len(ag) > 20 else ag,
                "Binding_Prob": f"{prob:.4f}",
                "Prediction": predicted_class
            })
            
    df_results = pd.DataFrame(results)
    print("\n--- Interaction Predictions ---")
    print(df_results.to_markdown(index=False))
    return df_results

if __name__ == "__main__":
    # Generic Test Sequence Data
    ab_demo = ["EVQLVESGGGLVQPGGSLRLSCAASGFTFSTYAMNWVRQAPGKGLEWVGRIRSK", "DIVMTQSPSSLSASVGDRVTITCRASQGISNYLAWYQQKPGKAPKLLIYAASTL"]
    ag_demo = ["SSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHLTQDLFLPFFSNVTWF", "MDVNPTLLFLKVPAQNAISTTFPYSSRRHRLTLFDLNGLLNGQ"]
    
    predict_interactions(ab_demo, ab_demo)
