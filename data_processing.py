import os
import pandas as pd
import numpy as np
from joblib import load
from utils import MEMBRANE_DICTIONARY, SOLVENT_DICTIONARY, ORIGINAL_PERMEANCES, CATEGORY_NAMES, NUMERICAL_NAMES

class InputGenerator:
    ALLOWED_MEMBRANES = (
        "DM300",
        "GMT-oNF-2",
        "PBI",
        "NF90",
        "PMS600",
        "SM122",
        "NF270",
    )
    ALLOWED_SOLVENTS = (
        "Water",
        "Toluene",
        "Methyl tetrahydrofuran",
        "Methanol",
        "Ethanol",
        "Dimethyl formamide",
        "Acetonitrile",
        "Acetone",
        "Ethyl acetate",
    )
    ALLOWED_ROLES = ("OSN", "Loose NF", "NF")

    def __init__(
        self,
        input_data,
        membranes,
        solvents,
        roles,
        pressure=20,
        temperature=20.0,
        process_configuration="CF",
        ph=7.0,
    ):
        # feaure for TODO: allow to load from user uploaded .csv
        # if isinstance(input_data, pd.DataFrame):
        #     self.dataframe = input_data
        # else:
        #     if not os.path.isfile(input_data) or not input_data.endswith(".csv"):
        #         raise ValueError("Invalid input: File not found or invalid file extension.")

        #     try:
        #         with open(input_data, 'r', encoding='utf-8') as file:
        #             self.dataframe = pd.read_csv(file)
        #     except OSError:
        #         raise ValueError("Error reading .csv file. Check extensions and UTF-8 coding.")
        self.dataframe = pd.DataFrame(
            {'name': ['User SMILES'],
             'smiles': [input_data]}
        )
        self.membranes = [membranes] if not isinstance(membranes, list) else membranes
        self.solvents = [solvents] if not isinstance(solvents, list) else solvents
        self.roles = [roles] if not isinstance(roles, list) else roles

        for membrane, solvent, role in zip(self.membranes, self.solvents, self.roles):
            if membrane not in self.ALLOWED_MEMBRANES:
                raise ValueError(
                    f"The membrane '{membrane}' is not supported. Try: {self.ALLOWED_MEMBRANES}"
                )

            if solvent not in self.ALLOWED_SOLVENTS:
                raise ValueError(
                    f"The solvent '{solvent}' is not supported. Try: {self.ALLOWED_SOLVENTS}"
                )

            if role not in self.ALLOWED_ROLES:
                raise ValueError(
                    f"The role '{role}' is not supported. Try: {self.ALLOWED_ROLES}"
                )

        self.pressure = check_and_assign(
            pressure,
            (float, int),
            f"Only a single pressure is accepted. The value {pressure} is not accepted.",
        )

        self.temperature = check_and_assign(
            temperature,
            (float, int),
            f"Only a single temperature is accepted. The value {temperature} is not accepted.",
        )

        self.process_configuration = check_and_assign(
            process_configuration,
            str,
            f"Invalid process configuration: {process_configuration}",
        )

        self.ph = check_and_assign(ph, (float, int), f"Invalid pH value: {ph}")

        self.new_columns = [s.lower() for s in self.dataframe.columns]
        if "smiles" not in self.new_columns:
            raise ValueError(
                "SMILES/smiles column is not provided. Please provide a SMILES/smiles column in the input file"
            )
        else:
            self.dataframe.rename(
                columns={
                    new_col: old_col
                    for (new_col, old_col) in zip(
                        self.dataframe.columns, self.new_columns
                    )
                },
                inplace=True,
            )

        self.smiles = list(self.dataframe.smiles)
        self.dataframe = self.generate_permutations()

    def generate_permutations(self):
        permutations = len(self.membranes) * len(self.solvents) * len(self.roles)

        new_df = pd.concat([self.dataframe] * permutations, ignore_index=True)
        if len(self.membranes) >= len(self.solvents):
            new_df["membrane"] = np.concatenate(
                [
                    ([i] * int(len(new_df) / len(self.membranes)))
                    for i in self.membranes
                ],
                axis=0,
            )
            new_df["solvent"] = self.solvents * int(len(new_df) / len(self.solvents))
        else:
            new_df["membrane"] = self.membranes * int(len(new_df) / len(self.membranes))
            new_df["solvent"] = np.concatenate(
                [([i] * int(len(new_df) / len(self.solvents))) for i in self.solvents],
                axis=0,
            )

        new_df["role"] = self.roles * int(len(new_df) / len(self.roles))
        new_df["temperature"] = self.temperature
        new_df["pressure"] = self.pressure
        new_df["process_configuration"] = self.process_configuration
        new_df["ph"] = self.ph

        return new_df

    def generate_features(self):
        self.membrane = self.dataframe["membrane"]
        self.solvent = self.dataframe["solvent"]
        self.smiles = self.dataframe["smiles"]
        self.names = self.dataframe["name"]
        self.process_parameters = self.dataframe.loc[
            :, ["temperature", "process_configuration", "ph", "pressure"]
        ]

        self.membrane_parameters = pd.DataFrame(
            columns=["role", "membrane", "mwco", "contact_angle", "zeta_potential"]
        )
        self.solvent_parameters = pd.DataFrame(
            columns=[
                "solvent_name",
                "solvent_smiles",
                "surface_tension",
                "solvent_mw",
                "solvent_diameter",
                "solvent_viscosity",
                "density",
                "solvent_dipole_moment",
                "solvent_dielectric_constant",
                "solvent_hildebrand",
                "solvent_logp",
                "solvent_dt",
                "solvent_dp",
                "solvent_dh",
            ]
        )
        self.full_smiles = pd.DataFrame(columns=["full_smiles"])
        self.permeances = pd.DataFrame(columns=["permeance"])

        for mem, solv, smile in zip(self.membrane, self.solvent, self.smiles):
            mem_series = pd.DataFrame(
                [MEMBRANE_DICTIONARY[mem]],
                columns=self.membrane_parameters.columns,
            )
            self.membrane_parameters = pd.concat(
                [self.membrane_parameters, mem_series], ignore_index=True, axis=0
            )

            solv_series = pd.DataFrame(
                [SOLVENT_DICTIONARY[solv][:-1]],
                columns=self.solvent_parameters.columns,
            )
            self.solvent_parameters = pd.concat(
                [self.solvent_parameters, solv_series], ignore_index=True, axis=0
            )
            full_smile = pd.DataFrame(
                [SOLVENT_DICTIONARY[solv][1] + "." + smile],
                columns=["full_smiles"],
            )
            self.full_smiles = pd.concat(
                [self.full_smiles, full_smile], ignore_index=True, axis=0
            )

            permeance = pd.DataFrame(
                [ORIGINAL_PERMEANCES[mem][solv]], columns=["permeance"]
            )
            self.permeances = pd.concat(
                [self.permeances, permeance], ignore_index=True, axis=0
            )

        self.full = pd.concat(
            [
                self.membrane_parameters,
                self.solvent_parameters,
                self.full_smiles,
                self.process_parameters,
                self.permeances,
            ],
            axis=1,
        )

        self.categorical = self.full.loc[:, CATEGORY_NAMES]
        self.numerical = self.full.loc[:, NUMERICAL_NAMES]

        encoder = load(
            r"one_hot_encoder.joblib"
        )
        one_hot_array = encoder.transform(self.categorical.to_numpy()).toarray()

        columns = [f"one_hot_{x}" for x in range(len(one_hot_array[0]))]
        self.one_hot_df = pd.DataFrame(one_hot_array, columns=columns)
        self.features = pd.concat([self.names, self.full_smiles, self.solvent, self.categorical, self.numerical, self.one_hot_df], axis=1)
        # self.names_smiles = pd.concat(
        #     [
        #         self.names,
        #         self.full_smiles,
        #         self.solvent,
        #         self.membrane,
        #         self.process_parameters,
        #         self.permeances,
        #     ],
        #     axis=1,
        # )

    def dump(self, save_name: str):
        self.features.to_csv(
            save_name,
            index=False,
        )
        # self.names_smiles.to_csv(
        #     rf"temp_generated_inputs\generated_full_smiles_{save_name}.csv",
        #     index=False,
        # )


def check_and_assign(value, accepted_types, error_message):
    if isinstance(value, accepted_types):
        return value
    else:
        raise ValueError(error_message)