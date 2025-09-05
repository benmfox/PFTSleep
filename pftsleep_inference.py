import yaml, glob, os, torch, sys

from pftsleep.inference import infer_on_edf_dataset, EDFDataset, download_pftsleep_models
from torch.utils.data import DataLoader
from pathlib import Path
import getpass

yaml_path = sys.argv[1]
with open(yaml_path, 'r') as f:
    yaml_data = yaml.safe_load(f)


if __name__ == "__main__":

    if os.path.exists(os.path.join(yaml_data['models_dir'], yaml_data['encoder_model_name'])) and os.path.exists(os.path.join(yaml_data['models_dir'], yaml_data['classifier_model_name'])):
        print(f"Encoder and classifier models found in {yaml_data['models_dir']}")
    else:
        print(f"Encoder and classifier models not found in {yaml_data['models_dir']}")
        print(f"Downloading encoder and classifier models to {yaml_data['models_dir']}...")
        if not os.path.exists(yaml_data['models_dir']):
            os.makedirs(yaml_data['models_dir'])
        try:
            download_pftsleep_models(write_dir=yaml_data['models_dir'], token=getpass.getpass("Enter your Hugging Face token: "))
        except Exception as e:
            raise ValueError(f"Error downloading encoder and classifier models: {e}")

    edf_directory_or_file_path = yaml_data['edf_directory_or_file_path']
    assert os.path.exists(edf_directory_or_file_path), f"EDF directory or file path not found in {edf_directory_or_file_path}"
    if Path(edf_directory_or_file_path).is_dir():
        edf_file_paths = glob.glob(os.path.join(edf_directory_or_file_path, '*.edf')) + glob.glob(os.path.join(edf_directory_or_file_path, '*.EDF'))
    elif Path(edf_directory_or_file_path).is_file():
        assert Path(edf_directory_or_file_path).suffix.lower() in ['.edf', '.EDF'], f"The file {edf_directory_or_file_path} is not an EDF file"
        edf_file_paths = [edf_directory_or_file_path]
    else:
        raise ValueError(f"EDF directory or file path not found in {edf_directory_or_file_path}")

    assert len(edf_file_paths) > 0, "No EDF files found in {edf_directory_or_file_path}"
    dataset = EDFDataset(edf_file_paths=edf_file_paths, 
                        eeg_channel=yaml_data['eeg_channel'], 
                        left_eog_channel=yaml_data['left_eog_channel'], 
                        chin_emg_channel=yaml_data['chin_emg_channel'],
                        ecg_channel=yaml_data['ecg_channel'], 
                        spo2_channel=yaml_data['spo2_channel'], 
                        abdomen_rr_channel=yaml_data['abdomen_rr_channel'], 
                        thoracic_rr_channel=yaml_data['thoracic_rr_channel'],
                        eeg_reference_channel=yaml_data['eeg_reference_channel'],
                        left_eog_reference_channel=yaml_data['left_eog_reference_channel'],
                        chin_emg_reference_channel=yaml_data['chin_emg_reference_channel'],
                        ecg_reference_channel=yaml_data['ecg_reference_channel'],
                        **yaml_data['process_edf_kwargs']
                        )
    data_loader = DataLoader(dataset, batch_size=yaml_data['batch_size'], pin_memory=yaml_data['pin_memory'], persistent_workers=yaml_data['persistent_workers'], num_workers=yaml_data['num_workers'])
    preds = infer_on_edf_dataset(edf_dataloader=data_loader, 
                                device=yaml_data['device'],
                                models_dir=yaml_data['models_dir'],
                                encoder_model_name=yaml_data['encoder_model_name'],
                                classifier_model_name=yaml_data['classifier_model_name']
                                )

    torch.save(preds, yaml_data['preds_output_path'])

