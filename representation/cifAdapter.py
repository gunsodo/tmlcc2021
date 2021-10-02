# OS command
import os

# Sure, we need `pandas`
import pandas as pd

# Typing
from typing import (
    List,
    Tuple
)

# Abstract classes we're going to use them  in the section
from abc import ABC, abstractmethod

# Base model
from pydantic import BaseModel

class AbstractAdapter(ABC):
    @abstractmethod
    def extract(self):
        """
        Extract input data
        """

    def tranform(self):
        """
        Tranformed data into desired format
        """

    def load(self):
        """
        Load data to next step
        """

    @abstractmethod
    def apply(self):
        """
        Apply data ETL process
        """
        pass

class CIF2PandasAdapterOutput(BaseModel):
    """
    Set up desired output
    """
    
    metadata: pd.DataFrame
    loops: List[pd.DataFrame]

    class Config:
        """
        Set some config to allow pd.DataFrame to BaseModel
        """
        arbitrary_types_allowed = True

class CIF2PandasAdapter(AbstractAdapter):
    """
    An adapter to convert CIF into Pandas DataFrames
    """

    @staticmethod
    def load_cif(cif_filepath: str) -> List[str]:
        """
        Read CIF file as String and split looping sections
        """
        with open(cif_filepath) as f:
            filename = f.readline().strip()
            dataframes = []
            dataframe = []
            for line in f.readlines():
                columns = [
                    l.strip() for l in line.strip().split(" ") if l and (l != "")
                ]
                if columns:
                    if columns[0] == "loop_":
                        dataframes.append(dataframe)
                        dataframe = []
                    else:
                        if "fapswitch" not in columns and columns != [""]:
                            dataframe.append(columns)
            dataframes.append(dataframe)

        return dataframes

    @staticmethod
    def get_metadata(dataframes: List[List[str]]) -> pd.DataFrame:
        """
        Get metadata of a CIF file
        """
        return pd.DataFrame(dataframes[0])

    @staticmethod
    def get_loops(dataframes: List[List[str]]) -> List[pd.DataFrame]:
        """
        Get loops
        """
        loops = []
        for dataframe in dataframes[1:]:
            loop = pd.DataFrame(dataframe)
            loop_fixed = loop[loop[1].notna()]
            loop_fixed.columns = loop[loop[1].isna()][0]
            loops.append(loop_fixed)

        return loops

    def extract(self, cif_filepath: str) -> List[str]:
        """
        Read filepath and return sectioned string
        """
        return self.load_cif(cif_filepath)

    def transform(self, cif_list: List[str]) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """
        Transform CIF text into datafram
        :params:
        cif_list: List of sectioned CIF text
        :return:
        Tuple of :
        (
            dataframe of CIF metadata,
            Tuple of (
                dataframe of loop_1,
                dataframe of loop_2,
            )
        )
        """
        metadata = self.get_metadata(cif_list)
        extract_loops = self.get_loops(cif_list)
        return metadata, extract_loops

    def load(
        self, metadata: pd.DataFrame, extract_loops: List[pd.DataFrame]
    ) -> CIF2PandasAdapterOutput:
        """
        Load tranformed data into desired output
        """
        output = CIF2PandasAdapterOutput(metadata=metadata, loops=extract_loops)
        return output

    def apply(self, cif_filepath: str) -> List[pd.DataFrame]:
        """
        Apply ETL pipeline
        """
        # Extract
        cif_list = self.extract(cif_filepath)

        # Transform
        metadata = self.get_metadata(cif_list)
        extract_loops = self.get_loops(cif_list)

        # Load
        output = self.load(metadata, extract_loops)

        return output