import json
import os
import sys
import re
from multiprocessing import Pool

# Add the current directory to sys.path to allow imports from NN_datasets
sys.path.append(os.path.dirname(__file__))


from AttackType import AttackType


# Helper functions for parallel processing (must be at module level for pickling)
def _load_single_json_file(file_info):
    """Load and parse a single JSON file."""
    filepath, vehicle_id = file_info
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            content = f.read()
            fixed_content = re.sub(r',(\s*[\]}])', r'\1', content)
            return (vehicle_id, json.loads(fixed_content))
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return (vehicle_id, None)


def _parse_json_content(content_info):
    """Parse JSON content from string."""
    content, vehicle_id = content_info
    try:
        fixed_content = re.sub(r',(\s*[\]}])', r'\1', content)
        return (vehicle_id, json.loads(fixed_content))
    except Exception as e:
        print(f"Error parsing JSON for {vehicle_id}: {e}")
        return (vehicle_id, None)


class VeReMiNextGen():
    def __init__(
            self, 
            scenario=["highway", "urban"], 
            traffic_density=[2,7], 
            attack_types=[1,2,3,4,5,6,7,8,9,10,11,12,13,14], 
            data_type="train", 
            dataset_type="VeReMiNextGen",
            use_sliding_window=False,
            ground_truth_only=False
             ):
        self.data = []
        self.NN_input = []

        self.scenario = scenario
        self.traffic_density = traffic_density
        self.attack_types = attack_types
        self.data_type = data_type
        self.dataset_type = dataset_type
        self.use_sliding_window = use_sliding_window
        self.ground_truth_only = ground_truth_only
        self._load_data()

    def _load_data(self):
        for scenario in self.scenario:
            for td in self.traffic_density:
                if (self.ground_truth_only):
                    dataset_name = f"InTAS_{scenario}_{td}"

                    if (self.dataset_type == "VeReMiNextGen"):
                        dataset_dir = f"datasets/veremi_next_gen_time_split/{dataset_name}/{self.data_type}"
                        simulation_dict = self._load_data_from_path(dataset_dir)
                    else:
                        dataset_dir = f"ground_truth/{dataset_name}/{self.data_type}/"
                        simulation_dict = self._load_data_from_zip(f"{dataset_dir}/{dataset_name}_enriched.zip")

                    simulation_dict["scenario"] = scenario
                    simulation_dict["td"] = td
                    simulation_dict["at"] = "_"
                    self.data.append(simulation_dict)
                    self._prepare_sim(simulation_dict)
                    print(f"Loaded VeReMi NextGen data: Scenario={scenario}, Traffic Density={td}, Data Type={self.data_type}")

                else:
                    if self.data_type == "Optimization":
                        data_type_str = "Validation"
                    else:
                        data_type_str = self.data_type
                    for at in self.attack_types:
                        print(f"Loading VeReMi NextGen data: Scenario={scenario}, Traffic Density={td}, Attack Type={at}, Data Type ={data_type_str}")
                        dataset_name = f"InTAS_{scenario}_{td}_{AttackType(at).name}"
                        
                        if (self.dataset_type == "VeReMiNextGen"):
                            dataset_dir = f"datasets/veremi_next_gen_time_split/{dataset_name}/{data_type_str}"
                            simulation_dict = self._load_data_from_path(dataset_dir)
                        else:
                            dataset_dir = f"dataset/{dataset_name}/{data_type_str}/"
                            simulation_dict = self._load_data_from_zip(f"{dataset_dir}/{dataset_name}.zip")

                        
                        simulation_dict["scenario"] = scenario
                        simulation_dict["td"] = td
                        simulation_dict["at"] = at
                        self.data.append(simulation_dict)
                        self._prepare_sim(simulation_dict)

    def _load_data_from_path(self, file_path):
        # Dispatcher: prefer directory loader if path is a directory, otherwise treat as zip
        file_path = os.path.normpath("./" + file_path)
        if os.path.isdir(file_path):
            return self._load_data_from_dir(file_path)
        else:
            return self._load_data_from_zip(file_path)

    def _load_data_from_dir(self, dir_path):
        """Load vehicle JSON files from a directory (parallelized for speed)."""
        import time
        start = time.time()
        
        dir_path = os.path.normpath(dir_path)
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Directory not found: '{dir_path}'")
        
        # Get all JSON files
        json_files = []
        for file in os.listdir(dir_path):
            filename = os.fsdecode(file)
            if filename.endswith('.json'):
                filepath = os.path.join(dir_path, filename)
                vehicle_id = filename.split('.')[0]
                json_files.append((filepath, vehicle_id))
        
        # Parallel loading with 8 workers (reduced for I/O-bound operations)
        data = {}
        if len(json_files) > 0:
            with Pool(8) as pool:
                results = pool.map(_load_single_json_file, json_files)
            data = {vehicle_id: content for vehicle_id, content in results if content is not None}
        
        elapsed = time.time() - start
        print(f"  → JSON loading from dir took {elapsed:.1f}s ({len(json_files)} files)")
        return {"vehicle_data": data}

    def _load_data_from_zip(self, zip_path):
        """Load vehicle JSON files from exactly one zip file (parallelized for speed)."""
        import zipfile
        import time
        start = time.time()
        
        zip_path = os.path.normpath(zip_path)
        if not os.path.isfile(zip_path):
            raise FileNotFoundError(f"ZIP file not found: '{zip_path}'")
        if not zipfile.is_zipfile(zip_path):
            raise ValueError(f"Expected a .zip file containing vehicle JSONs: '{zip_path}'")

        # Extract all JSON entries
        json_entries = []
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if name.endswith('.json') and not name.endswith('/'):
                    vehicle_id = os.path.splitext(os.path.basename(name))[0]
                    content = z.read(name).decode('utf-8')
                    json_entries.append((content, vehicle_id))
        
        # Parallel JSON parsing with 8 workers (reduced for better performance)
        data = {}
        if len(json_entries) > 0:
            with Pool(8) as pool:
                results = pool.map(_parse_json_content, json_entries)
            data = {vehicle_id: content for vehicle_id, content in results if content is not None}
        
        elapsed = time.time() - start
        print(f"  → JSON loading from zip took {elapsed:.1f}s ({len(json_entries)} files)")
        return {"vehicle_data": data}

    def _save_NN_input(self):
        pass
    
    def _prepare_sim(self, simulation_dict):
        # needs to be implemented by child classes
        pass