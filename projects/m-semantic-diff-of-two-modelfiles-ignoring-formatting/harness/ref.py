MF1 = "FROM llama3\nPARAMETER temperature 0.7\nSYSTEM 'hello'\n"
MF2 = "FROM llama3\nPARAMETER temperature 0.8\nSYSTEM 'hello'\n"
MF3 = "FROM llama3\nPARAMETER temperature 0.7\nSYSTEM 'world'\n"

VALID_MF = "FROM llama3\nPARAMETER temperature 0.0\n"
INVALID_MF = "FROM llama3\nINVALID_CMD foo\n"
INVALID_PARAM_MF = "FROM llama3\nPARAMETER badparam 1.0\n"

BASE_MODEL = "llama3"
