"""Fixed transfer registry. Author: Moh'd Khier Al Kfari."""
from dataclasses import dataclass
@dataclass(frozen=True)
class Transfer:
    family:str; source:str; target:str
    @property
    def name(self): return f"{self.source}_to_{self.target}"
TRANSFERS=[*(Transfer("cross_dataset",a,b) for a,b in (("PAMAP2","RealWorld"),("RealWorld","PAMAP2"),("PAMAP2","MHEALTH"),("MHEALTH","PAMAP2"),("RealWorld","MHEALTH"),("MHEALTH","RealWorld"))),*(Transfer("cross_gender",a,b) for a,b in (("PAMAP2_male","PAMAP2_female"),("PAMAP2_female","PAMAP2_male"),("RealWorld_male","RealWorld_female"),("RealWorld_female","RealWorld_male"))),Transfer("cross_age","RealWorld_below30","RealWorld_age30plus"),Transfer("cross_age","RealWorld_age30plus","RealWorld_below30")]
def resolve_group(name):
    parts=name.split("_",1); return parts[0], parts[1] if len(parts)==2 else None
