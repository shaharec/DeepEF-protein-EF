NANO_TO_ANGSTROM = 0.1

AMINO_ACID_MAPPER = {'A': '0', 'C': '1', 'D': '2', 'E': '3', 'F': '4', 'G': '5', 'H': '6', 'I': '7', 'K': '8', 'L': '9',
                     'M': '10', 'N': '11', 'P': '12', 'Q': '13', 'R': '14', 'S': '15', 'T': '16', 'V': '17', 'W': '18',
                     'Y': '19', '-': '20'}

# Define the list of atom types to save
atom_types = ['N', 'CA', 'C', 'CB']
atom_mapper = {'N': 'CoordN', 'C': 'CoordC', 'CA': 'CoordAlpha', 'CB': 'CoordBeta'}


amino_acid_types = {
    "Positive": ["R", "H", "K"],
    "Negative": ["D", "E"],
    "Polar": ["S", "T", "N", "Q"],
    "Special": ["C", "G", "P"],
    "Hydrophobic": ["A", "V", "I", "L", "M", "F", "Y", "W"]
}

FIRST = 0
MISSING_COORD = [0, 0, 0]
OTHER_ACID = '-'

AA_SEQ_COL = 'aa_seq'
