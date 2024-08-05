import numpy as np 
import pandas as pd 
import joblib
import json


class single_input_generator(object):
    membrane_dictionary = {'AK': np.array(['LPRO', 'AK', 150, 50.0, -1.0], dtype=object),
                        'BW30': np.array(['RO', 'BW30', 100, 59.8, -1.0], dtype=object),
                        'BW440': np.array(['RO', 'BW440', 100, 56.8, -4.49], dtype=object),
                        'CK': np.array(['NF', 'CK', 225, 54.2, -1.0], dtype=object),
                        'CPA3': np.array(['RO', 'CPA3', 100, 73.0, -1.0], dtype=object),
                        'Crosslined_RC': np.array(['RO', 'Crosslined_RC', 300, 26.0, -1.0], dtype=object),
                        'DDK': np.array(['OSN', 'DDK', 150, -1.0, -1.0], dtype=object),
                        'DF30': np.array(['NF', 'DF30', 400, 15.2, -75.0], dtype=object),
                        'DK': np.array(['NF', 'DK', 250, 40.6, -1.0], dtype=object),
                        'DM1000': np.array(['OSN', 'DM1000', 1000, 59.0, -1.0], dtype=object),
                        'DM1000X': np.array(['OSN', 'DM1000X', 1000, 59.0, -1.0], dtype=object),
                        'DM150': np.array(['OSN', 'DM150', 150, 59.0, -1.0], dtype=object),
                        'DM200': np.array(['OSN', 'DM200', 200, 59.0, -20.0], dtype=object),
                        'DM300': np.array(['OSN', 'DM300', 300, 59.0, -37.5], dtype=object),
                        'DM500': np.array(['OSN', 'DM500', 500, 59.0, -31.0], dtype=object),
                        'DM700': np.array(['OSN', 'DM700', 700, 59.0, -1.0], dtype=object),
                        'Desal51HL': np.array(['NF', 'Desal51HL', 190, 47.0, -13.0], dtype=object),
                        'Desal5DL': np.array(['NF', 'Desal5DL', 260, 44.0, -17.0], dtype=object),
                        'ES10C': np.array(['RO', 'ES10C', 65, 54.4, -30.0], dtype=object),
                        'ES20': np.array(['LPRO', 'ES20', 200, 40.0, -20.0], dtype=object),
                        'ESNA': np.array(['LPRO', 'ESNA', 250, 47.3, -11.0], dtype=object),
                        'ESPA': np.array(['LPRO', 'ESPA', 200, 47.0, -1.0], dtype=object),
                        'ESPA1': np.array(['LPRO', 'ESPA1', 200, 22.6, -30.6], dtype=object),
                        'ESPA2': np.array(['LPRO', 'ESPA2', 200, 43.0, -38.0], dtype=object),
                        'ESPA3': np.array(['LPRO', 'ESPA3', 100, 60.6, -24.8], dtype=object),
                        'GMT-oNF-1': np.array(['OSN', 'GMT-oNF-1', 600, -1.0, -1.0], dtype=object),
                        'GMT-oNF-2': np.array(['OSN', 'GMT-oNF-2', 350, 87.0, -15.0], dtype=object),
                        'GMT-oNF-3': np.array(['OSN', 'GMT-oNF-3', 600, -1.0, -1.0], dtype=object),
                        'HL': np.array(['NF', 'HL', 190, 26.8, -11.0], dtype=object),
                        'HR95': np.array(['NF', 'HR95', 200, 40.0, -20.0], dtype=object),
                        'LC2': np.array(['NF', 'LC2', 437, 31.3, -9.6], dtype=object),
                        'LE440': np.array(['LPRO', 'LE440', 100, 42.0, -23.0], dtype=object),
                        'LF10': np.array(['RO', 'LF10', 150, 50.0, -24.0], dtype=object),
                        'LFC1': np.array(['RO', 'LFC1', 100, 78.0, 4.0], dtype=object),
                        'MPF44': np.array(['OSN', 'MPF44', 250, -1.0, -1.0], dtype=object),
                        'MPF50': np.array(['OSN', 'MPF50', 700, -1.0, -1.0], dtype=object),
                        'NE90': np.array(['NF', 'NE90', 200, 52.0, -23.6], dtype=object),
                        'NF': np.array(['NF', 'NF', 300, 53.8, -5.48], dtype=object),
                        'NF200': np.array(['NF', 'NF200', 300, 30.3, -24.0], dtype=object),
                        'NF270': np.array(['loose NF', 'NF270', 340, 33.0, -21.6], dtype=object),
                        'NF70': np.array(['NF', 'NF70', 250, 8.5, -36.4], dtype=object),
                        'NF90': np.array(['NF', 'NF90', 118, 41.4, -7.0], dtype=object),
                        'NTR 7450': np.array(['NF', 'NTR 7450', 600, 70.0, -17.0], dtype=object),
                        'NTR729HF': np.array(['NF', 'NTR729HF', 150, 49.6, -30.0], dtype=object),
                        'NTR7450': np.array(['NF', 'NTR7450', 310, 70.0, 1.0], dtype=object),
                        'PA1': np.array(['OSN', 'PA1', 350, -1.0, -1.0], dtype=object),
                        'PBI': np.array(['OSN', 'PBI', 200, 34.0, -15.0], dtype=object),
                        'PDMS': np.array(['OSN', 'PDMS', 250, -1.0, -1.0], dtype=object),
                        'PIP': np.array(['RO', 'PIP', 250, 47.1, -12.8], dtype=object),
                        'PM280': np.array(['OSN', 'PM280', 280, -1.0, -1.0], dtype=object),
                        'PMS380': np.array(['OSN', 'PMS380', 600, -1.0, -1.0], dtype=object),
                        'PMS600': np.array(['OSN', 'PMS600', 600, -1.0, -1.0], dtype=object),
                        'PV46': np.array(['OSN', 'PV46', 600, -1.0, -1.0], dtype=object),
                        'RE-BLR': np.array(['RO', 'RE-BLR', 100, 47.0, -20.9], dtype=object),
                        'SB50': np.array(['RO', 'SB50', 152, 63.0, -13.0], dtype=object),
                        'SM122': np.array(['OSN', 'SM122', 220, -1.0, -1.0], dtype=object),
                        'SM240': np.array(['OSN', 'SM240', 400, -1.0, -1.0], dtype=object),
                        'SR2': np.array(['NF', 'SR2', 460, 40.1, -11.5], dtype=object),
                        'SR3': np.array(['NF', 'SR3', 165, 44.6, -17.0], dtype=object),
                        'SS126': np.array(['OSN', 'SS126', 300, -1.0, -1.0], dtype=object),
                        'SS126S': np.array(['OSN', 'SS126S', 300, -1.0, -1.0], dtype=object),
                        'SS136': np.array(['OSN', 'SS136', 500, -1.0, -1.0], dtype=object),
                        'SS336': np.array(['OSN', 'SS336', 500, 82.0, -1.0], dtype=object),
                        'SS375': np.array(['OSN', 'SS375', 500, -1.0, -1.0], dtype=object),
                        'SS815': np.array(['OSN', 'SS815', 500, -1.0, -1.0], dtype=object),
                        'SS911': np.array(['OSN', 'SS911', 350, -1.0, -1.0], dtype=object),
                        'SS981': np.array(['OSN', 'SS981', 350, -1.0, -1.0], dtype=object),
                        'SW30XLE': np.array(['RO', 'SW30XLE', 100, 55.0, -23.0], dtype=object),
                        'SWC1': np.array(['RO', 'SWC1', 100, 41.7, -31.0], dtype=object),
                        'TFC-SR2': np.array(['NF', 'TFC-SR2', 460, 57.0, -9.5], dtype=object),
                        'TS80': np.array(['tight NF', 'TS80', 200, 48.0, -14.0], dtype=object),
                        'UTC20': np.array(['RO', 'UTC20', 180, 36.0, -1.0], dtype=object),
                        'UTC60': np.array(['RO', 'UTC60', 150, 49.6, -30.0], dtype=object),
                        'UTC70': np.array(['LPRO', 'UTC70', 65, 54.4, -30.0], dtype=object),
                        'VNF1': np.array(['loose NF', 'VNF1', 240, 36.4, -44.0], dtype=object),
                        'VNF2': np.array(['tight NF', 'VNF2', 150, 79.4, -52.0], dtype=object),
                        'X20': np.array(['RO', 'X20', 200, 55.0, -87.0], dtype=object),
                        'XLE': np.array(['RO', 'XLE', 96, 59.6, -19.0], dtype=object),
                        'XLE440': np.array(['LPRO', 'XLE440', 150, 39.8, -19.0], dtype=object)}

    solvent_dictionary = {'2-propanol': np.array(['2-propanol', 'CC(O)C', 21.79, 60.1, 0.624, 1.96, 0.786, 1.68,
                                22.0, 11.5, 0.26, 7.7, 3.0, 8.0, '2-propanol'], dtype=object),
                        'Acetone': np.array(['Acetone', 'CC(C)=O', 23.32, 58.1, 0.618, 0.3, 0.784, 2.85, 21.0,
                                10.0, -0.24, 7.6, 5.1, 3.4, 'Acetone'], dtype=object),
                        'Acetonitrile': np.array(['Acetonitrile', 'CC#N', 19.1, 41.1, 0.55, 0.37, 0.786, 3.5, 37.5,
                                11.9, -0.34, 7.5, 8.8, 3.0, 'Acetonitrile'], dtype=object),
                        'Cyclohexane': np.array(['Cyclohexane', 'C1CCCCC1', 24.98, 84.16, 0.6, 0.99, 0.774, 0.0,
                                1.88, 8.2, 4.15, 8.2, 0.0, 0.1, 'Cyclohexane'], dtype=object),
                        'Dichloromethane': np.array(['Dichloromethane', 'ClCCl', 28.12, 84.9, 0.47, 0.44, 1.326, 1.14,
                                8.93, 9.7, 1.25, 8.9, 3.1, 3.0, 'Dichloromethane'], dtype=object),
                        'Dimethyl acetamide': np.array(['Dimethyl acetamide', 'CC(N(C)C)=O', 32.43, 87.12, 0.72, 2.14,
                                0.942, 3.72, 37.78, 11.0, -0.7, 8.2, 5.6, 5.0,
                                'Dimethyl acetamide'], dtype=object),
                        'Dimethyl formamide': np.array(['Dimethyl formamide', 'O=CN(C)C', 36.76, 73.1, 0.666, 0.8, 0.937,
                                3.8, 37.0, 12.1, -0.74, 8.5, 6.7, 5.5, 'Dimethyl formamide'],
                        dtype=object),
                        'Ethanol': np.array(['Ethanol', 'CCO', 22.32, 46.1, 0.52, 1.07, 1.044, 1.7, 24.0,
                                13.4, -0.32, 7.7, 4.3, 9.5, 'Ethanol'], dtype=object),
                        'Ethyl acetate': np.array(['Ethyl acetate', 'CC(OCC)=O', 23.75, 88.1, 0.677, 0.46, 0.902,
                                1.78, 6.0, 9.1, 0.73, 7.7, 2.6, 3.5, 'Ethyl acetate'], dtype=object),
                        'Heptane': np.array(['Heptane', 'CCCCCCC', 20.3, 86.2, 0.748, 0.39, 0.655, 0.0, 0.39,
                                7.5, 3.5, 7.5, 0.0, 0.0, 'Heptane'], dtype=object),
                        'Methanol': np.array(['Methanol', 'CO', 22.55, 32.04, 0.505, 0.55, 0.792, 1.6, 33.0,
                                14.5, -0.82, 7.4, 6.0, 10.9, 'Methanol'], dtype=object),
                        'Methyl ethyl ketone': np.array(['Methyl ethyl ketone', 'CC(CC)=O', 24.0, 72.11, 0.525, 0.43,
                                0.85, 2.78, 18.5, 9.3, 0.29, 7.8, 4.4, 2.5,
                                'Methyl ethyl ketone'], dtype=object),
                        'Methyl tert butyl ether': np.array(['Methyl tert butyl ether', 'CC(C)(C)OC', 19.4, 88.2, 0.72, 0.36,
                                0.74, 1.4, 2.6, 7.4, 1.05, 14.8, 4.3, 5.0,
                                'Methyl tert butyl ether'], dtype=object),
                        'Methyl tetrahydrofuran': np.array(['Methyl tetrahydrofuran', 'CC1CCCO1', 26.4, 86.1, 0.685, 0.46,
                                0.854, 1.38, 6.97, 16.9, 1.01, 16.9, 5.0, 4.3,
                                'Methyl tetrahydrofuran'], dtype=object),
                        'Tetrahyrofuran': np.array(['Tetrahyrofuran', 'C1CCCO1', 26.4, 72.11, 0.6, 0.55, 0.887, 1.69,
                                7.58, 9.1, 0.46, 8.2, 2.8, 3.9, 'Tetrahyrofuran'], dtype=object),
                        'Toluene': np.array(['Toluene', 'CC1=CC=CC=C1', 28.53, 92.1, 0.697, 0.59, 0.867, 0.36,
                                2.4, 8.9, 2.69, 8.8, 0.7, 1.0, 'Toluene'], dtype=object),
                        'Water': np.array(['Water', 'O', 72.8, 18.0, 0.275, 1.0, 0.998, 1.87, 80.1, 23.5,
                                -1.38, 7.6, 7.8, 20.7, 'Water'], dtype=object),
                        'n-Hexane': np.array(['n-Hexane', 'CCCCCCC', 17.91, 86.18, 0.251, 0.31, 0.659, 0.08,
                                1.9, 6.9, 3.8, 7.3, 0.0, 0.0, 'n-Hexane'], dtype=object)}
    
    
    category_names = ['role', 'membrane', 'process_configuration']
    
    
    numerical_names = ['mwco', 'zeta_potential', 'contact_angle', 'pressure', 'surface_tension', 'solvent_mw', 'solvent_diameter',
       'solvent_viscosity', 'density', 'solvent_dipole_moment',
       'solvent_dielectric_constant', 'solvent_hildebrand', 'solvent_logp',
       'solvent_dt', 'solvent_dp', 'solvent_dh', 'permeance', 'temperature', 'ph']
    
    
    def __init__(self):
        self.ALLOWED_MEMBRANES = {"DM300","GMT-oNF-2","PBI","NF90","PMS600","SM122","NF270"}
        self.ALLOWED_SOLVENTS = {'Water', 'Toluene', 'Methyl tetrahydrofuran', 'Methanol', 
                    'Ethanol', 'Dimethyl formamide', 'Acetonitrile', 'Acetone', 'Ethyl acetate'}
        self.ALLOWED_ROLES = {'OSN', 'Loose NF', 'NF'}
        


    def generate_features(self, row):
        import joblib
        import pandas as pd
        import numpy as np
        import json
        self.generator_data = row
        if not isinstance(row, list):        
            self.membrane = self.generator_data['membrane']
            self.solvent = self.generator_data['solvent']
            self.ph = self.generator_data['ph']
            self.temperature = self.generator_data['temperature']
            self.pressure = self.generator_data['pressure']
            self.process_configuration = self.generator_data['process_configuration']
        else:
            self.membrane = self.generator_data[3]
            self.solvent = self.generator_data[2]
            self.ph = self.generator_data[6]
            self.temperature = self.generator_data[4]
            self.pressure = self.generator_data[7]
            self.process_configuration = self.generator_data[5]

        membrane_parameters = self.membrane_dictionary[self.membrane]
        solvent_parameters = self.solvent_dictionary[self.solvent]
        #self.full_smiles = smiles + '.' + solvent_parameters[1]
        #self.full_smiles_df = pd.DataFrame([self.full_smiles], columns=['full_smiles'])
        categorical = [membrane_parameters[0], membrane_parameters[1], self.process_configuration]
        
        f = open('permeances.json')
        permeances = json.load(f)
        
        numerical = [
            membrane_parameters[2],
            membrane_parameters[4], # contact angle
            membrane_parameters[3], # zeta potential
            self.pressure, # pressure
            solvent_parameters[2],
            solvent_parameters[3],
            solvent_parameters[4],
            solvent_parameters[5],
            solvent_parameters[6],
            solvent_parameters[7],
            solvent_parameters[8],
            solvent_parameters[9],
            solvent_parameters[10],
            solvent_parameters[11],
            solvent_parameters[12],
            solvent_parameters[13],
            permeances[self.membrane][self.solvent],
            self.temperature, # temperature
            self.ph # ph
        ]
        encoder = joblib.load('gnn_predictor/chemprop/one_hot_encoder.joblib')
        one_hot_array = encoder.transform([categorical]).toarray()
        columns = [f"one_hot_{x}" for x in range(len(one_hot_array[0]))]
        self.one_hot_df = pd.DataFrame(one_hot_array, columns=columns)
        self.numerical_df = pd.DataFrame([numerical], columns=self.numerical_names)
        self.features = pd.concat([self.numerical_df, self.one_hot_df], axis=1)
        return self.features.to_numpy(dtype=float)

